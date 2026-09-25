import argparse
import asyncio
import ipaddress
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

# Resolver ruta raíz del proyecto ORION
PROJECT_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.parent.name == "versions" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debe fijarse ANTES de que MiDaS ejecute cualquier operación en MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# En Windows, usar SelectorEventLoopPolicy evita errores [WinError 10054] al cerrar pestañas o refrescar el navegador
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except ImportError:
    _HAS_SIMPLEAUDIO = False


MODEL_TYPE_DEFAULT = "MiDaS_small"


# ---------------------------------------------------------------------------
# Torch / MiDaS Core (Identical to main-prueba.py)
# ---------------------------------------------------------------------------

def configure_torch(threads: int = 0) -> None:
    if threads > 0:
        try:
            torch.set_num_threads(int(threads))
            torch.set_num_interop_threads(max(1, min(2, int(threads))))
        except RuntimeError:
            pass

    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_built = getattr(backend, "is_built", lambda: True)()
    is_available = getattr(backend, "is_available", lambda: False)()
    return bool(is_built and is_available)


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").lower()

    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if preferred == "mps":
        return torch.device("mps") if _mps_available() else torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_midas_torch(model_type: str, device: torch.device):
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    try:
        model.to(device)
    except (RuntimeError, NotImplementedError) as exc:
        print(f"⚠️ No se pudo mover el modelo a {device}: {exc}. Usando CPU.")
        device = torch.device("cpu")
        model.to(device)

    model.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = (
        midas_transforms.dpt_transform
        if model_type in ("DPT_Large", "DPT_Hybrid")
        else midas_transforms.small_transform
    )
    return model, transform, device


@torch.inference_mode()
def infer_depth_torch(
    model,
    transform,
    device,
    frame_bgr: np.ndarray,
    use_fp16_mps: bool = False,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device, non_blocking=True)

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(input_batch)
    elif device.type == "mps" and use_fp16_mps and not infer_depth_torch._mps_fp16_failed:
        try:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                prediction = model(input_batch)
        except RuntimeError as exc:
            infer_depth_torch._mps_fp16_failed = True
            print(f"\n⚠️ Autocast FP16 no soportado en MPS ({exc}); usando FP32.")
            prediction = model(input_batch)
    else:
        prediction = model(input_batch)

    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).squeeze(0)

    return prediction.float().cpu().numpy()


infer_depth_torch._mps_fp16_failed = False


def normalize_depth(depth_raw: np.ndarray, invert: bool) -> np.ndarray:
    min_value = float(depth_raw.min())
    max_value = float(depth_raw.max())
    span = max_value - min_value

    if span < 1e-6:
        normalized = np.zeros_like(depth_raw, dtype=np.float32)
    else:
        normalized = (depth_raw - min_value) / span

    if invert:
        normalized = 1.0 - normalized

    return normalized


def analyze_depth(
    depth_norm: np.ndarray,
    near_threshold: float = 0.6,
    sample_step: int = 4,
):
    depth = depth_norm[::sample_step, ::sample_step] if sample_step > 1 else depth_norm

    _, width = depth.shape[:2]
    third = width // 3
    center = depth[:, third:2 * third]

    center_mean = float(center.mean())
    overall_mean = float(depth.mean())

    action = "STOP" if overall_mean > 0.9 or center_mean > near_threshold else "FORWARD"

    if action == "STOP":
        reference = max(center_mean, overall_mean)
        span = max(1e-6, 1.0 - near_threshold)
        intensity = min(1.0, max(0.0, (reference - near_threshold) / span))
    else:
        intensity = 0.0

    return action, intensity, center_mean, overall_mean


def colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    depth_uint8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)


# ---------------------------------------------------------------------------
# Audio System (Local + ESP32)
# ---------------------------------------------------------------------------

def _generate_tone_pcm(
    frequency: float,
    duration: float,
    volume: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    fade_len = max(1, int(sample_rate * 0.01))
    envelope = np.ones_like(tone)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    tone = tone * envelope * volume
    return np.int16(np.clip(tone, -1.0, 1.0) * 32767)


class LocalAlarmSound:
    LEVELS = [
        (0.00, 700.0, 0.12, 0.55, 0.55),
        (0.30, 950.0, 0.14, 0.38, 0.70),
        (0.60, 1200.0, 0.16, 0.24, 0.85),
        (0.85, 1500.0, 0.18, 0.15, 1.00),
    ]

    def __init__(self, enabled: bool = True, sample_rate: int = 44100):
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.backend = self._detect_backend() if enabled else None
        self._tmp_files = []
        self._level_data = []

        if self.backend is not None:
            self._prepare_levels()

    def _detect_backend(self):
        if _HAS_SIMPLEAUDIO:
            return "simpleaudio"
        if sys.platform == "darwin" and shutil.which("afplay"):
            return "afplay"
        return None

    def _prepare_levels(self):
        for _, freq, dur, _interval, vol in self.LEVELS:
            pcm = _generate_tone_pcm(freq, dur, vol, self.sample_rate)
            if self.backend == "simpleaudio":
                self._level_data.append(("pcm", pcm))
            else:
                path = self._write_wav_tempfile(pcm)
                self._tmp_files.append(path)
                self._level_data.append(("file", path))

    def _write_wav_tempfile(self, pcm: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="orion_alarm_")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return path

    def level_for_intensity(self, intensity: float) -> int:
        level = 0
        for i, (min_intensity, *_rest) in enumerate(self.LEVELS):
            if intensity >= min_intensity:
                level = i
        return level

    def interval_for_level(self, level: int) -> float:
        return self.LEVELS[level][3]

    def play(self, level: int) -> None:
        if self.backend is None or not self.enabled:
            return
        kind, data = self._level_data[level]
        try:
            if kind == "pcm":
                sa.play_buffer(data, 1, 2, self.sample_rate)
            else:
                subprocess.Popen(
                    ["afplay", data],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def cleanup(self) -> None:
        for path in self._tmp_files:
            try:
                os.remove(path)
            except Exception:
                pass


class AlarmManager:
    def __init__(
        self,
        esp_controller,
        local_sound: "LocalAlarmSound",
        buzzer_duty_max: int = 255,
        buzzer_duration_ms: int = 150,
    ):
        self.esp = esp_controller
        self.local_sound = local_sound
        self.buzzer_duty_max = max(0, min(255, int(buzzer_duty_max)))
        self.buzzer_duration_ms = max(0, int(buzzer_duration_ms))
        self.esp_buzzer_enabled = True

        self._lock = threading.Lock()
        self._active = False
        self._intensity = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "AlarmManager":
        self._thread.start()
        return self

    def update(self, active: bool, intensity: float) -> None:
        with self._lock:
            self._active = bool(active)
            self._intensity = max(0.0, min(1.0, float(intensity)))

    def _snapshot(self):
        with self._lock:
            return self._active, self._intensity

    def _run(self) -> None:
        was_active = False

        while not self._stop_event.is_set():
            active, intensity = self._snapshot()

            if not active:
                if was_active and self.esp is not None and self.esp_buzzer_enabled:
                    self.esp.set_buzzer(False)
                was_active = False
                time.sleep(0.05)
                continue

            was_active = True

            if self.local_sound is not None and self.local_sound.enabled:
                level = self.local_sound.level_for_intensity(intensity)
                interval = self.local_sound.interval_for_level(level)
                self.local_sound.play(level)
            else:
                interval = 0.55 - 0.40 * intensity

            if self.esp is not None and self.esp_buzzer_enabled:
                duty = int(self.buzzer_duty_max * (0.4 + 0.6 * intensity))
                self.esp.pulse_buzzer(duty, self.buzzer_duration_ms)

            time.sleep(max(0.05, interval))

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

        if self.esp is not None:
            self.esp.set_buzzer(False)

        if self.local_sound is not None:
            self.local_sound.cleanup()


# ---------------------------------------------------------------------------
# Capture Readers
# ---------------------------------------------------------------------------

class LatestFrameReader:
    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass


class MjpegReader:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.resp = None
        self.buffer = bytearray()
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.resp = self.session.get(
            self.url,
            stream=True,
            timeout=(3, 10),
            headers={"Connection": "keep-alive"},
        )
        self.resp.raise_for_status()
        self.thread.start()
        return self

    def _run(self):
        for chunk in self.resp.iter_content(chunk_size=8192):
            if self.stopped:
                break
            if not chunk:
                continue

            self.buffer.extend(chunk)

            while not self.stopped:
                start = self.buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(self.buffer) > 1_000_000:
                        del self.buffer[:-2]
                    break

                end = self.buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start > 0:
                        del self.buffer[:start]
                    break

                jpg = self.buffer[start:end + 2]
                del self.buffer[:end + 2]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is not None:
                    with self.lock:
                        self.latest = frame

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def release(self):
        self.stopped = True
        try:
            if self.resp is not None:
                self.resp.close()
        except Exception:
            pass


class SocketMjpegReader:
    def __init__(self, url: str):
        parsed = urlparse(url)
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path or "/stream"
        self.sock = None
        self.buffer = bytearray()
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(2.0)

        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            "Connection: keep-alive\r\n\r\n"
        ).encode()
        self.sock.sendall(request)

        header = bytearray()
        while b"\r\n\r\n" not in header:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("ESP32 cerró la conexión MJPEG")
            header.extend(chunk)

        body = bytes(header).split(b"\r\n\r\n", 1)[1]
        self.buffer.extend(body)
        self.thread.start()
        return self

    def _decode_available(self):
        while not self.stopped:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                if len(self.buffer) > 1_000_000:
                    del self.buffer[:-2]
                return

            end = self.buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start > 0:
                    del self.buffer[:start]
                return

            jpg = self.buffer[start:end + 2]
            del self.buffer[:end + 2]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is not None:
                with self.lock:
                    self.latest = frame

    def _run(self):
        while not self.stopped:
            try:
                data = self.sock.recv(8192)
                if not data:
                    time.sleep(0.002)
                    continue
                self.buffer.extend(data)
                self._decode_available()
            except socket.timeout:
                continue
            except Exception:
                if not self.stopped:
                    time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def release(self):
        self.stopped = True
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass


class RawLenSocketReader:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(2.0)
        self.thread.start()
        return self

    def _recv_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n and not self.stopped:
            try:
                chunk = self.sock.recv(n - len(data))
            except socket.timeout:
                continue
            if not chunk:
                return bytes(data)
            data.extend(chunk)
        return bytes(data)

    def _run(self):
        while not self.stopped:
            try:
                header = self._recv_exact(4)
                if len(header) != 4:
                    time.sleep(0.005)
                    continue

                (length,) = struct.unpack(">I", header)
                if length <= 0 or length > 5_000_000:
                    continue

                jpg = self._recv_exact(length)
                if len(jpg) != length:
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is not None:
                    with self.lock:
                        self.latest = frame
            except Exception:
                if not self.stopped:
                    time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def release(self):
        self.stopped = True
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Network / ESP32
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def get_network_range():
    local_ip = get_local_ip()
    network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    return [str(ip) for ip in network.hosts()]


def check_esp32_port(ip: str, port: int = 81, timeout: float = 0.35) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((ip, port)) == 0
    except OSError:
        return False


def check_esp32_stream(ip: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(
            f"http://{ip}:81/stream",
            timeout=timeout,
            stream=True,
        )
        if response.status_code == 200:
            response.close()
            return True
    except requests.RequestException:
        pass
    return False


def check_esp32_control(ip: str, timeout: float = 1.0) -> bool:
    try:
        response = requests.get(f"http://{ip}/control", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def auto_find_esp32() -> str | None:
    print("🔍 Escaneando red local en busca de ESP32-CAM...")
    known_ips = [
        "10.35.132.231",
        "172.16.121.9",
        "192.168.1.100",
        "192.168.0.100",
        "192.168.1.1",
    ]

    for ip in known_ips:
        if check_esp32_port(ip, 81):
            if check_esp32_stream(ip) or check_esp32_control(ip):
                print(f"🎉 ESP32-CAM encontrado en {ip}")
                return ip

    network_ips = get_network_range()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(check_esp32_port, ip, 81, 0.35): ip
            for ip in network_ips
        }

        candidates = []
        for future in as_completed(futures):
            if future.result():
                candidates.append(futures[future])

    for ip in candidates:
        if check_esp32_stream(ip) or check_esp32_control(ip):
            print(f"🎯 ESP32-CAM encontrado: {ip}")
            return ip

    print("❌ No se encontró ningún ESP32-CAM")
    return None


def infer_esp_base_from_src(src: str) -> str:
    if not src.startswith(("http://", "https://")):
        return ""
    parsed = urlparse(src)
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    port = parsed.port
    if port in (None, 81):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


class EspController:
    def __init__(self, base_url: str, buzzer_duty: int = 200):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.buzzer_duty = max(0, min(255, int(buzzer_duty)))
        self.last_buzzer_state = None
        self.lock = threading.Lock()

    def request(self, path: str, timeout=(0.2, 0.6)):
        if not self.base_url:
            return None
        try:
            return self.session.get(f"{self.base_url}{path}", timeout=timeout)
        except requests.RequestException:
            return None

    def set_param(self, var: str, value: int):
        self.request(f"/control?var={var}&val={int(value)}", timeout=(0.2, 0.8))

    def set_buzzer(self, on: bool):
        if not self.base_url:
            return
        state = bool(on)
        with self.lock:
            if state == self.last_buzzer_state:
                return
            self.last_buzzer_state = state
        self.request(
            f"/buzzer?on={1 if state else 0}&duty={self.buzzer_duty}",
            timeout=(0.15, 0.5),
        )

    def pulse_buzzer(self, duty: int, duration_ms: int = 150):
        if not self.base_url:
            return
        duty = max(0, min(255, int(duty)))
        with self.lock:
            self.last_buzzer_state = True
        self.request(
            f"/buzzer?on=1&duty={duty}&duration={max(0, int(duration_ms))}",
            timeout=(0.15, 0.5),
        )

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


def open_capture(source: str, bufsize: int = 1):
    if source.isdigit():
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
    else:
        cap = cv2.VideoCapture(source)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(bufsize))
    except Exception:
        pass

    return cap


# ---------------------------------------------------------------------------
# ORION Web Engine (Background Processing Core)
# ---------------------------------------------------------------------------

class OrionEngine:
    """
    Orquesta la ingesta de video, inferencia MiDaS, alarmas y entrega de
    frames/telemetría al servidor web sin bloquear llamadas HTTP.
    """

    def __init__(self, args):
        self.args = args
        self.near_threshold = float(args.near_threshold)
        self.running = False
        self.lock = threading.Lock()

        # State Telemetry
        self.last_action = "FORWARD"
        self.last_intensity = 0.0
        self.last_center_prox = 0.0
        self.last_overall_prox = 0.0
        self.fps = 0.0
        self.frame_count = 0

        # Latest frames buffers
        self.latest_raw_jpeg = None
        self.latest_depth_jpeg = None
        self.latest_combined_jpeg = None

        # ML Core initialization
        configure_torch(args.torch_threads)
        if args.cv_threads > 0:
            cv2.setNumThreads(args.cv_threads)

        self.device = resolve_device(args.device)
        print(f"🧠 Inicializando MiDaS en device: {self.device}...")
        self.model, self.transform, self.device = load_midas_torch(args.model_type, self.device)
        print(f"✅ MiDaS listo ({self.device.type})")

        # ESP Controller
        esp_base = args.esp_base.strip() or infer_esp_base_from_src(args.src)
        self.esp = EspController(esp_base, args.buzzer_duty) if esp_base else None
        if self.esp:
            self.esp.set_param("quality", args.esp_quality)
            self.esp.set_param("framesize", args.esp_framesize)

        # Audio Alarms
        self.local_sound = LocalAlarmSound(enabled=not args.no_local_sound)
        self.alarm = AlarmManager(
            self.esp,
            self.local_sound,
            buzzer_duty_max=args.buzzer_duty,
            buzzer_duration_ms=args.buzzer_duration,
        ).start()

        # Capture Reader
        self.capture = None
        self.async_reader = None
        self.stream_reader = None
        self._init_reader()

        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)

    def _init_reader(self):
        args = self.args
        if args.raw_socket:
            base = args.esp_base.strip() or infer_esp_base_from_src(args.src)
            host = urlparse(base or args.src).hostname
            if host:
                try:
                    self.stream_reader = RawLenSocketReader(host, args.raw_port).start()
                except Exception as exc:
                    print(f"⚠️ Raw socket error: {exc}")

        elif args.socket_mjpeg and args.src.startswith(("http://", "https://")):
            try:
                self.stream_reader = SocketMjpegReader(args.src).start()
            except Exception as exc:
                print(f"⚠️ Socket MJPEG error: {exc}")

        elif args.mjpeg_reader and args.src.startswith(("http://", "https://")):
            try:
                self.stream_reader = MjpegReader(args.src).start()
            except Exception as exc:
                print(f"⚠️ MJPEG error: {exc}")

        if self.stream_reader is None:
            self.capture = open_capture(args.src, args.cap_bufsize)
            if not self.capture.isOpened():
                print(f"❌ Error al abrir fuente de video: {args.src}")
                return
            if args.async_capture and not args.sync_capture:
                self.async_reader = LatestFrameReader(self.capture).start()

    def start(self):
        self.running = True
        self.worker_thread.start()
        return self

    def _run_loop(self):
        args = self.args
        prev_depth = None
        last_depth = None
        last_depth_norm = None

        fps_counter = 0
        fps_time = time.perf_counter()

        mps_cache_supported = (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        )

        while self.running:
            if self.stream_reader is not None:
                ok, frame_bgr = self.stream_reader.read()
            elif self.async_reader is not None:
                ok, frame_bgr = self.async_reader.read()
            elif self.capture is not None:
                ok, frame_bgr = self.capture.read()
            else:
                ok, frame_bgr = False, None

            if not ok or frame_bgr is None:
                time.sleep(0.003)
                continue

            self.frame_count += 1
            proc_frame = frame_bgr

            # Resize before inference
            if args.proc_width > 0 and frame_bgr.shape[1] > args.proc_width:
                scale = args.proc_width / frame_bgr.shape[1]
                new_size = (args.proc_width, max(1, int(frame_bgr.shape[0] * scale)))
                proc_frame = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

            if args.deband and args.deband_ksize >= 3 and args.deband_ksize % 2 == 1:
                proc_frame = cv2.GaussianBlur(proc_frame, (1, args.deband_ksize), 0)

            # Inference
            run_inference = (
                last_depth is None
                or args.inference_every <= 1
                or self.frame_count % args.inference_every == 0
            )

            if run_inference:
                last_depth = infer_depth_torch(
                    self.model,
                    self.transform,
                    self.device,
                    proc_frame,
                    use_fp16_mps=args.mps_fp16,
                )
                depth_norm = normalize_depth(last_depth, invert=args.invert_depth)

                if 0.0 < args.temporal_alpha < 1.0:
                    if prev_depth is None or prev_depth.shape != depth_norm.shape:
                        prev_depth = depth_norm.copy()
                    else:
                        alpha = float(args.temporal_alpha)
                        prev_depth *= 1.0 - alpha
                        prev_depth += alpha * depth_norm
                    depth_used = prev_depth
                else:
                    depth_used = depth_norm

                last_depth_norm = depth_used

                action, intensity, center_prox, overall_prox = analyze_depth(
                    depth_used,
                    near_threshold=self.near_threshold,
                )

                self.alarm.update(active=(action == "STOP"), intensity=intensity)

                with self.lock:
                    self.last_action = action
                    self.last_intensity = intensity
                    self.last_center_prox = center_prox
                    self.last_overall_prox = overall_prox

            else:
                depth_used = last_depth_norm

            if (
                mps_cache_supported
                and args.mps_empty_cache_every > 0
                and self.frame_count % args.mps_empty_cache_every == 0
            ):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            # Visual Renderings (Clean lane overlays without messy text burn)
            depth_vis = colorize_depth(depth_used)
            h, w = proc_frame.shape[:2]
            third = w // 3

            # Draw sleek lane divider lines on visual outputs
            raw_vis = proc_frame.copy()
            cv2.line(raw_vis, (third, 0), (third, h), (0, 200, 200), 1)
            cv2.line(raw_vis, (2 * third, 0), (2 * third, h), (0, 200, 200), 1)

            cv2.line(depth_vis, (third, 0), (third, h), (0, 200, 200), 1)
            cv2.line(depth_vis, (2 * third, 0), (2 * third, h), (0, 200, 200), 1)

            combined_vis = np.hstack((raw_vis, depth_vis))

            # Encode JPEGs in memory
            _, raw_jpg = cv2.imencode(".jpg", raw_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            _, depth_jpg = cv2.imencode(".jpg", depth_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            _, combined_jpg = cv2.imencode(".jpg", combined_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            with self.lock:
                self.latest_raw_jpeg = raw_jpg.tobytes()
                self.latest_depth_jpeg = depth_jpg.tobytes()
                self.latest_combined_jpeg = combined_jpg.tobytes()

            # FPS calculation
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                with self.lock:
                    self.fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

    def get_jpeg(self, mode: str = "combined") -> bytes | None:
        with self.lock:
            if mode == "raw":
                return self.latest_raw_jpeg
            elif mode == "depth":
                return self.latest_depth_jpeg
            return self.latest_combined_jpeg

    def get_telemetry(self) -> dict:
        with self.lock:
            return {
                "action": self.last_action,
                "center_prox": round(self.last_center_prox, 4),
                "overall_prox": round(self.last_overall_prox, 4),
                "intensity": round(self.last_intensity, 4),
                "near_threshold": round(self.near_threshold, 3),
                "fps": round(self.fps, 1),
                "device": self.device.type,
                "frame_count": self.frame_count,
                "local_sound": self.local_sound.enabled,
                "esp_buzzer": self.alarm.esp_buzzer_enabled,
                "timestamp": time.time(),
            }

    def update_settings(self, settings: dict):
        with self.lock:
            if "near_threshold" in settings:
                self.near_threshold = max(0.05, min(0.95, float(settings["near_threshold"])))
            if "local_sound_enabled" in settings:
                self.local_sound.enabled = bool(settings["local_sound_enabled"])
            if "esp_buzzer_enabled" in settings:
                self.alarm.esp_buzzer_enabled = bool(settings["esp_buzzer_enabled"])

    def stop(self):
        self.running = False
        if self.async_reader:
            self.async_reader.stop()
        if self.capture:
            self.capture.release()
        if self.stream_reader:
            self.stream_reader.release()
        self.alarm.stop()
        if self.esp:
            self.esp.close()


# ---------------------------------------------------------------------------
# FastAPI Web App
# ---------------------------------------------------------------------------

app = FastAPI(title="ORION Web Dashboard")
engine: OrionEngine | None = None
server_instance: uvicorn.Server | None = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_candidates = [
        Path(__file__).parent / "templates" / "index.html",
        PROJECT_ROOT / "templates" / "index.html",
    ]
    template_path = next((p for p in template_candidates if p.exists()), None)
    if template_path and template_path.exists():
        return HTMLResponse(template_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ORION Web Dashboard</h1><p>templates/index.html no encontrado</p>")


@app.get("/video_feed")
async def video_feed(mode: str = "combined"):
    """MJPEG streaming endpoint for HTML <img> tags."""
    async def frame_generator():
        try:
            while engine and engine.running and not getattr(server_instance, "should_exit", False):
                jpg = engine.get_jpeg(mode)
                if jpg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                await asyncio.sleep(0.033)  # ~30 FPS stream cap
        except (asyncio.CancelledError, ConnectionResetError, OSError):
            pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time WebSocket telemetry channel."""
    await websocket.accept()
    try:
        while engine and engine.running and not getattr(server_instance, "should_exit", False):
            data = engine.get_telemetry()
            await websocket.send_json(data)
            await asyncio.sleep(0.04)  # ~25 Hz telemetry updates
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError, ConnectionResetError, OSError):
        pass


@app.get("/api/status")
async def api_status():
    if engine:
        return JSONResponse(engine.get_telemetry())
    return JSONResponse({"status": "error", "message": "Engine not initialized"})


@app.post("/api/settings")
async def api_settings(request: Request):
    if engine:
        payload = await request.json()
        engine.update_settings(payload)
        return JSONResponse({"status": "ok", "settings": engine.get_telemetry()})
    return JSONResponse({"status": "error"})


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ORION Web Service: MiDaS Depth Navigation & HTML5 Dashboard"
    )

    # Server Args
    parser.add_argument("--host", default="0.0.0.0", help="Host IP del servidor web (0.0.0.0 para acceso en red local)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP del servidor web (default 8000)")

    # Video Source
    parser.add_argument("--src", default="0", help="Cámara (0) o URL de stream MJPEG")
    parser.add_argument("--near-threshold", type=float, default=0.6, help="Umbral relativo de proximidad (0.0 a 1.0)")
    parser.add_argument("--invert-depth", action="store_true")
    parser.add_argument("--model-type", default=MODEL_TYPE_DEFAULT, choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"])
    parser.add_argument("--esp-base", default="", help="Base URL del ESP32-CAM")
    parser.add_argument("--auto-find", action="store_true", help="Buscar ESP32-CAM automáticamente en red local")

    # Hardware & Audio
    parser.add_argument("--buzzer-duty", type=int, default=200)
    parser.add_argument("--buzzer-duration", type=int, default=150)
    parser.add_argument("--no-local-sound", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mps-fp16", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--cv-threads", type=int, default=0)
    parser.add_argument("--mps-empty-cache-every", type=int, default=300)

    # Processing & Stream
    parser.add_argument("--proc-width", type=int, default=320)
    parser.add_argument("--inference-every", type=int, default=1)
    parser.add_argument("--temporal-alpha", type=float, default=0.65)
    parser.add_argument("--deband", action="store_true")
    parser.add_argument("--deband-ksize", type=int, default=5)

    # Capture Readers
    parser.add_argument("--async-capture", action="store_true", default=True)
    parser.add_argument("--sync-capture", action="store_true")
    parser.add_argument("--cap-bufsize", type=int, default=1)
    parser.add_argument("--mjpeg-reader", action="store_true")
    parser.add_argument("--socket-mjpeg", action="store_true")
    parser.add_argument("--raw-socket", action="store_true")
    parser.add_argument("--raw-port", type=int, default=3333)
    parser.add_argument("--esp-framesize", type=int, default=6)
    parser.add_argument("--esp-quality", type=int, default=15)

    return parser.parse_args()


def find_available_port(host: str, preferred_port: int) -> int:
    """
    Verifica si el puerto preferido está libre. Si está reservado/bloqueado
    (común en Windows con Hyper-V/WSL2 en el puerto 8000), busca automáticamente
    el siguiente puerto disponible (8080, 8088, 8501, 5000, etc.).
    """
    candidates = [preferred_port, 8080, 8088, 8501, 5000, 3000, 8888]
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main():
    global engine
    args = parse_args()

    if args.auto_find:
        esp_ip = auto_find_esp32()
        if not esp_ip:
            sys.exit(1)
        args.src = f"http://{esp_ip}:81/stream"
        args.esp_base = f"http://{esp_ip}"
        print(f"🎯 Usando ESP32-CAM: {esp_ip}")

    # Start Background ORION Engine
    engine = OrionEngine(args).start()

    local_ip = get_local_ip()
    chosen_port = find_available_port(args.host, args.port)

    if chosen_port != args.port:
        print(f"⚠️ El puerto {args.port} está reservado u ocupado en tu sistema. Usando puerto alternativo libre: {chosen_port}")

    # Guardar el puerto elegido para que el script de inicio
    # pueda abrir automáticamente el navegador en la URL correcta.
    port_file = Path(__file__).parent / ".orion_port"

    try:
        port_file.write_text(str(chosen_port), encoding="utf-8")
    except OSError as exc:
        print(f"⚠️ No se pudo guardar el puerto de ORION: {exc}")

    print("\n" + "=" * 60)
    print("🌌 ORION Web Dashboard iniciado exitosamente!")
    print(f"   👉 Acceso Local:        http://localhost:{chosen_port}")
    print(f"   👉 Acceso en Red WiFi:  http://{local_ip}:{chosen_port}")
    print("   Presiona 'Ctrl + C' en la terminal para detener el servidor.")
    print("=" * 60 + "\n")

    # Run Uvicorn HTTP/WebSocket Server with instant shutdown
    global server_instance
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=chosen_port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=0.1,
    )
    server_instance = uvicorn.Server(config)



    try:
        server_instance.run()
    except (KeyboardInterrupt, SystemExit):
        print("\nDeteniendo servidor...")
    finally:
        if engine:
            engine.stop()
        print("✅ ORION Web Service detenido.")


if __name__ == "__main__":
    main()
