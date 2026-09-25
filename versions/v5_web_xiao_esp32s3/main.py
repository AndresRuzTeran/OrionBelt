import argparse
import asyncio
import ipaddress
import json
import io
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
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

# Resolver ruta raíz del proyecto ORION
PROJECT_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.parent.name == "versions" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debe fijarse ANTES de que PyTorch/MiDaS ejecute cualquier operación en MPS (Apple Silicon)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Reconfigurar salida de consola a UTF-8 para evitar UnicodeEncodeError en Windows CP1252
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(*args, **kwargs):
    """Impresión segura en consola protegida contra UnicodeEncodeError en terminales Windows."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        clean_args = [
            a.encode("ascii", errors="replace").decode("ascii") if isinstance(a, str) else a
            for a in args
        ]
        print(*clean_args, **kwargs)


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

try:
    from utils.xiao_discovery import XiaoDiscoveryManager
except ImportError:
    from xiao_discovery import XiaoDiscoveryManager


MODEL_TYPE_DEFAULT = "MiDaS_small"


# ---------------------------------------------------------------------------
# Torch / MiDaS Core (Inferencia optimizada en Apple Silicon MPS / CUDA)
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
# Audio System (Local Mac Sound + Control Remoto Buzzer XIAO ESP32S3)
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
        self._afplay_proc = None

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
                if self._afplay_proc is not None:
                    try:
                        self._afplay_proc.poll()
                    except Exception:
                        pass
                self._afplay_proc = subprocess.Popen(
                    ["afplay", data],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def cleanup(self) -> None:
        if self._afplay_proc is not None:
            try:
                self._afplay_proc.terminate()
            except Exception:
                pass
            self._afplay_proc = None
        for path in self._tmp_files:
            try:
                os.remove(path)
            except Exception:
                pass


class XiaoBuzzerController:
    """Controlador HTTP del Buzzer en el módulo XIAO ESP32S3."""
    def __init__(self, base_url: str, default_duty: int = 200):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.default_duty = max(0, min(255, int(default_duty)))
        self.last_buzzer_state = None
        self.lock = threading.Lock()

    def update_base_url(self, new_base: str):
        with self.lock:
            self.base_url = new_base.rstrip("/")

    def request(self, path: str, timeout=(0.15, 0.4)):
        if not self.base_url:
            return None
        try:
            return self.session.get(f"{self.base_url}{path}", timeout=timeout)
        except requests.RequestException:
            return None

    def set_param(self, var: str, value: int):
        self.request(f"/control?var={var}&val={int(value)}", timeout=(0.2, 0.5))

    def set_buzzer(self, on: bool):
        if not self.base_url:
            return
        state = bool(on)
        with self.lock:
            if state == self.last_buzzer_state:
                return
            self.last_buzzer_state = state
        self.request(
            f"/buzzer?on={1 if state else 0}&duty={self.default_duty}",
            timeout=(0.12, 0.35),
        )

    def pulse_buzzer(self, duty: int, duration_ms: int = 150, freq: int | None = None):
        if not self.base_url:
            return
        duty = max(0, min(255, int(duty)))
        with self.lock:
            self.last_buzzer_state = True
        
        url = f"/buzzer?duty={duty}&beep={max(0, int(duration_ms))}"
        if freq is not None:
            url += f"&freq={int(freq)}"
            
        self.request(url, timeout=(0.12, 0.35))

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


class AdaptiveAlarmManager:
    """
    Gestor de alarma reactivo que conmuta de forma inteligente:
    - Si la fuente activa es XIAO: emite por el buzzer del XIAO ESP32S3.
    - Si está en modo Rollback (Mac): emite por los altavoces de la Mac para no perder alerta.
    """
    def __init__(
        self,
        xiao_controller: "XiaoBuzzerController",
        local_sound: "LocalAlarmSound",
        buzzer_duty_max: int = 255,
        buzzer_duration_ms: int = 150,
    ):
        self.xiao = xiao_controller
        self.local_sound = local_sound
        self.buzzer_duty_max = max(0, min(255, int(buzzer_duty_max)))
        self.buzzer_duration_ms = max(0, int(buzzer_duration_ms))
        
        self.esp_buzzer_enabled = True
        self.active_source = "XIAO_ESP32S3"

        self._lock = threading.Lock()
        self._active = False
        self._intensity = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "AdaptiveAlarmManager":
        self._thread.start()
        return self

    def update(self, active: bool, intensity: float, current_source: str = "XIAO_ESP32S3") -> None:
        with self._lock:
            self._active = bool(active)
            self._intensity = max(0.0, min(1.0, float(intensity)))
            self.active_source = current_source

    def _snapshot(self):
        with self._lock:
            return self._active, self._intensity, self.active_source

    def _run(self) -> None:
        was_active = False

        while not self._stop_event.is_set():
            active, intensity, current_source = self._snapshot()

            if not active:
                if was_active and self.xiao is not None and self.esp_buzzer_enabled:
                    self.xiao.set_buzzer(False)
                was_active = False
                time.sleep(0.04)
                continue

            was_active = True

            # Si la fuente activa es XIAO ESP32S3, usamos su buzzer
            if current_source == "XIAO_ESP32S3" and self.xiao is not None and self.esp_buzzer_enabled:
                duty = int(100 + 155 * intensity)
                beep_dur = int(180 - 120 * intensity)
                freq = int(800 + 2200 * intensity)
                self.xiao.pulse_buzzer(duty, beep_dur, freq=freq)
                interval = 0.48 - 0.38 * intensity
                
                # Si el usuario además tiene activado local sound, también suena en Mac
                if self.local_sound is not None and self.local_sound.enabled:
                    level = self.local_sound.level_for_intensity(intensity)
                    self.local_sound.play(level)
            else:
                # Modo ROLLBACK (Cámara Mac activa): emitir alarma sonora a través de la Mac
                if self.local_sound is not None and self.local_sound.enabled:
                    level = self.local_sound.level_for_intensity(intensity)
                    interval = self.local_sound.interval_for_level(level)
                    self.local_sound.play(level)
                else:
                    interval = 0.50 - 0.38 * intensity

            time.sleep(max(0.04, interval))

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

        if self.xiao is not None:
            self.xiao.set_buzzer(False)

        if self.local_sound is not None:
            self.local_sound.cleanup()


# ---------------------------------------------------------------------------
# Lectores Asíncronos de Video & Dual Camera Source Manager (Cero Latencia)
# ---------------------------------------------------------------------------

class LatestFrameReader:
    """Lector no bloqueante para webcam local con descarte automático de frames viejos."""
    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.consecutive_fails = 0
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            try:
                cap = self.capture
                if cap is None:
                    break
                ok, frame = cap.read()
                if self.stopped:
                    break
                if not ok or frame is None:
                    self.consecutive_fails += 1
                    time.sleep(0.01)
                    continue
                self.consecutive_fails = 0
                with self.lock:
                    self.latest = frame
            except Exception:
                self.consecutive_fails += 1
                time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def is_alive_and_streaming(self) -> bool:
        return self.consecutive_fails < 30 and not self.stopped

    def stop(self):
        """Detención limpia y atómica: primero espera a que termine el hilo lector antes de liberar el hardware."""
        self.stopped = True
        if self.thread.is_alive():
            try:
                self.thread.join(timeout=1.5)
            except Exception:
                pass
        with self.lock:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None


class MjpegStreamReader:
    """
    Lector de stream MJPEG de red con soporte de Transfer-Encoding: chunked
    y validación estricta de integridad con PIL.
    Descarta al 100% fotogramas truncados y corruptos (evitando pantallas en blanco y negro o glitchs).
    """
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.resp = None
        self.latest = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.resp = self.session.get(
            self.url,
            stream=True,
            timeout=(2.5, 6.0),
            headers={"Connection": "keep-alive"},
        )
        try:
            sock = getattr(getattr(getattr(self.resp.raw, "_fp", None), "fp", None), "raw", None)
            if sock and hasattr(sock, "_sock"):
                sock._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.resp.raise_for_status()
        self.thread.start()
        return self

    def _run(self):
        buffer = bytearray()
        try:
            for chunk in self.resp.iter_content(chunk_size=4096):
                if self.stopped:
                    break
                if not chunk:
                    continue

                buffer.extend(chunk)

                # Extraer el fotograma más reciente disponible en el búfer (descarta frames obsoletos por lag de Wi-Fi)
                latest_jpg = None
                while not self.stopped:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 200_000:
                            del buffer[:-2]
                        break

                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break

                    candidate = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    if len(candidate) >= 800:
                        latest_jpg = candidate

                # Decodificación única acelerada C++ SIMD del fotograma más reciente
                if latest_jpg is not None and not self.stopped:
                    try:
                        frame = cv2.imdecode(np.frombuffer(latest_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            with self.lock:
                                self.latest = frame
                    except Exception:
                        pass
        except Exception:
            pass

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return frame is not None, frame

    def release(self):
        self.stopped = True
        try:
            sock = getattr(getattr(getattr(self.resp.raw, "_fp", None), "fp", None), "raw", None)
            if sock and hasattr(sock, "_sock"):
                try:
                    sock._sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                sock._sock.close()
        except Exception:
            pass
        try:
            if self.resp:
                self.resp.close()
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass


class DualCameraSourceManager:
    """
    Gestor con Histéresis y Control Anti-Flapping.
    Conmuta de forma autónoma entre el XIAO ESP32S3 y la cámara local con confirmación de 8 frames
    y tiempo de enfriamiento (cooldown) para evitar parpadeos descontrolados.
    Gestiona el ciclo de vida del hardware: apaga el LED/cámara local cuando XIAO está activo,
    y lo enciende cuando entra en modo Rollback.
    """
    def __init__(
        self,
        xiao_ip = None,
        xiao_stream_port: int = 81,
        xiao_ctrl_port: int = 80,
        local_src: int = 0,
        on_ip_resolved = None,
    ):
        self.discovery = XiaoDiscoveryManager()
        self.on_ip_resolved = on_ip_resolved
        if not xiao_ip:
            xiao_ip = self.discovery.get_known_ips()[0]
        self.xiao_ip = str(xiao_ip).strip()
        self.xiao_stream_port = int(xiao_stream_port)
        self.xiao_ctrl_port = int(xiao_ctrl_port)
        self.xiao_stream_url = f"http://{self.xiao_ip}:{self.xiao_stream_port}/stream"
        self.local_src = local_src

        self.lock = threading.Lock()
        self.local_cam_lock = threading.Lock()
        self.running = False

        self.active_source = "LOCAL_MAC"
        self.xiao_online = False
        self.camera_mode = "auto"  # "auto", "local", "xiao"
        self._seeking_tick = 0
        self.last_valid_frame = None

        self.xiao_reader = None
        self.local_capture = None
        self.local_reader = None

        self.last_xiao_frame_time = 0.0
        self.last_switch_time = 0.0
        self.COOLDOWN_SECONDS = 1.0  # Reducido a 1.0s para reconexión ágil
        self.XIAO_TIMEOUT_SECONDS = 3.5  # Aumentado a 3.5s para tolerar jitter de Wi-Fi sin desconectar
        self.MIN_CONFIRMATION_FRAMES = 1  # 1 frame válido basta para confirmar stream en vivo
        self._last_fast_probe = 0.0
        self._last_broad_scan = time.monotonic()
        self._is_seeking = False
        self.watchdog_thread = None

    def update_xiao_ip(self, new_ip: str):
        new_ip = str(new_ip).strip()
        if not new_ip:
            return
        with self.lock:
            if self.xiao_ip == new_ip:
                return
            safe_print(f"🔄 [XIAO-IP] Dirección actualizada: {self.xiao_ip} -> {new_ip}")
            self.xiao_ip = new_ip
            self.xiao_stream_url = f"http://{self.xiao_ip}:{self.xiao_stream_port}/stream"
            if self.xiao_reader:
                try:
                    self.xiao_reader.release()
                except Exception:
                    pass
                self.xiao_reader = None
            self.xiao_online = False

        if callable(self.on_ip_resolved):
            try:
                self.on_ip_resolved(new_ip)
            except Exception:
                pass

    def start(self):
        self.running = True

        safe_print(f"🔍 Comprobando disponibilidad inicial de XIAO ESP32S3 en {self.xiao_ip}...")
        connected = self._try_connect_xiao()

        if not connected:
            # 1. Intentar con resolución dinámica rápida (Caché conocida + UDP Beacon)
            safe_print("⚡ [XIAO-DISCOVERY] Resolviendo dirección activa del XIAO...")
            resolved_ip = self.discovery.resolve_xiao_ip(preferred_ip=self.xiao_ip, stream_port=self.xiao_stream_port, ctrl_port=self.xiao_ctrl_port, enable_broad_scan=False)
            if resolved_ip:
                self.update_xiao_ip(resolved_ip)
                connected = self._try_connect_xiao(force=True)

        if not connected:
            safe_print("🟡 XIAO no detectado en red. Activando cámara de respaldo...")
            with self.lock:
                self.active_source = "LOCAL_MAC"
                self.xiao_online = False
            self._init_local_camera()

        # Lanzar el escucha del faro UDP y el Watchdog en segundo plano
        self.udp_thread = threading.Thread(target=self._udp_beacon_loop, daemon=True)
        self.udp_thread.start()
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        return self

    def _udp_beacon_loop(self):
        """Escucha continua del faro UDP emitido por el XIAO ESP32S3 en puerto 9999."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.settimeout(2.0)
                s.bind(("", 9999))
                while self.running:
                    try:
                        data, addr = s.recvfrom(512)
                        text = data.decode("utf-8", errors="ignore").strip()
                        if "ORION_XIAO_CAM:" in text:
                            parts = text.split(":")
                            if len(parts) >= 2:
                                beacon_ip = parts[1].strip()
                                if beacon_ip and beacon_ip != self.xiao_ip:
                                    safe_print(f"📻 [UDP-BEACON] Nuevo faro detectado: {beacon_ip}. Enlazando...")
                                    self.update_xiao_ip(beacon_ip)
                                    self._try_connect_xiao(force=True)
                    except socket.timeout:
                        continue
                    except Exception:
                        break
        except Exception:
            pass

    def _init_local_camera(self):
        with self.local_cam_lock:
            if self.local_capture is not None and self.local_capture.isOpened():
                if self.local_reader is not None and self.local_reader.is_alive_and_streaming():
                    return
                self._release_local_camera_locked()

            src = int(self.local_src) if str(self.local_src).isdigit() else self.local_src
            cap = None
            if sys.platform == "win32" and isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            elif sys.platform == "darwin" and isinstance(src, int):
                cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(src)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.local_capture = cap
                self.local_reader = LatestFrameReader(cap).start()
                safe_print("💻 [LOCAL] Cámara web de respaldo activada (LED encendido).")
            else:
                self.local_capture = None
                self.local_reader = None

    def _release_local_camera(self):
        with self.local_cam_lock:
            self._release_local_camera_locked()

    def _release_local_camera_locked(self):
        if self.local_reader is not None:
            try:
                self.local_reader.stop()
            except Exception:
                pass
            self.local_reader = None

        # self.local_reader.stop() ya liberó la captura limpia y atómicamente.
        # Solo garantizamos limpiar la referencia para prevenir double-free en macOS AVFoundation.
        self.local_capture = None
        safe_print("💤 [LOCAL] Cámara web de respaldo liberada (LED apagado).")

    def _check_xiao_socket(self, timeout=0.6) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                # Comprobar exclusivamente el puerto de control (80).
                # NUNCA conectar un socket dummy al puerto 81 de stream porque el
                # microcontrolador solo admite 1 cliente y el probe lo bloquea temporalmente.
                return s.connect_ex((self.xiao_ip, self.xiao_ctrl_port)) == 0
        except OSError:
            return False

    def _try_connect_xiao(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force:
            cooldown = 0.2 if self.camera_mode == "xiao" else self.COOLDOWN_SECONDS
            if now - self.last_switch_time < cooldown:
                return False

        if not self._check_xiao_socket(timeout=0.6):
            return False

        try:
            reader = MjpegStreamReader(self.xiao_stream_url).start()
            
            # Tolerancia de hasta 3.0s para recibir el primer frame por Wi-Fi
            t0 = time.time()
            while time.time() - t0 < 3.0:
                ok, frame = reader.read()
                if ok and frame is not None and frame.size > 0:
                    with self.lock:
                        if self.xiao_reader:
                            self.xiao_reader.release()
                        self.xiao_reader = reader
                        self.last_xiao_frame_time = time.monotonic()
                        self.last_switch_time = time.monotonic()
                        self.xiao_online = True
                        self.active_source = "XIAO_ESP32S3"
                        self.last_valid_frame = frame
                    safe_print("\n🟢 [HOT-SWAP] ¡XIAO ESP32S3 verificado y enlazado! Conmutando a fuente primaria.")
                    # Liberar la cámara local para apagar el LED y ahorrar recursos
                    self._release_local_camera()
                    return True
                time.sleep(0.02)

            reader.release()
        except Exception:
            pass
        return False

    def _generate_seeking_frame(self) -> np.ndarray:
        self._seeking_tick += 1
        h, w = 240, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (26, 20, 16)
        
        dots = "." * ((self._seeking_tick // 5) % 4 + 1)
        cv2.putText(img, f"Buscando XIAO{dots}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"IP: {self.xiao_ip}:{self.xiao_stream_port}", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(img, "Modo: FORZADO XIAO", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1, cv2.LINE_AA)
        cv2.putText(img, "Camara local: APAGADA", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 160), 1, cv2.LINE_AA)
        return img

    def set_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in ("auto", "local", "xiao"):
            return
        with self.lock:
            if self.camera_mode == mode:
                return
            self.camera_mode = mode
            safe_print(f"🔄 [CAM-MODE] Modo de cámara cambiado a: {mode.upper()}")

        if mode == "local":
            with self.lock:
                self.active_source = "LOCAL_MAC"
                if self.xiao_reader:
                    try:
                        self.xiao_reader.release()
                    except Exception:
                        pass
                    self.xiao_reader = None
            self._init_local_camera()
        elif mode == "xiao":
            with self.lock:
                self.active_source = "XIAO_ESP32S3"
            # Modo XIAO forzado: la cámara local debe estar APAGADA (LED apagado)
            self._release_local_camera()
            if not self.xiao_online:
                self._try_connect_xiao(force=True)
        elif mode == "auto":
            if not self.xiao_online:
                self._init_local_camera()
                self._try_connect_xiao(force=True)

    def trigger_rollback(self, reason: str = ""):
        now = time.monotonic()
        should_switch = False
        with self.lock:
            # El Rollback a cámara local SOLO ocurre si el modo es 'auto'
            if self.camera_mode != "auto":
                return
            if self.active_source != "LOCAL_MAC":
                self.active_source = "LOCAL_MAC"
                self.xiao_online = False
                self.last_switch_time = now
                if self.xiao_reader:
                    try:
                        self.xiao_reader.release()
                    except Exception:
                        pass
                    self.xiao_reader = None
                should_switch = True

        if should_switch:
            safe_print(f"\n🟡 [ROLLBACK AUTO] {reason}. Conmutando a CÁMARA LOCAL...")
            self._init_local_camera()

    def _watchdog_loop(self):
        while self.running:
            time.sleep(0.3)  # Intervalo ágil de 0.3s
            now = time.monotonic()

            with self.lock:
                mode = self.camera_mode
                curr_src = self.active_source
                last_frame = self.last_xiao_frame_time
                last_sw = self.last_switch_time
                online = self.xiao_online

            # 1. Modo LOCAL forzado: mantener local activa y no buscar XIAO
            if mode == "local":
                if self.local_reader is None or not self.local_reader.is_alive_and_streaming():
                    self._init_local_camera()
                continue

            # 2. Modo AUTO:
            if mode == "auto":
                if curr_src == "XIAO_ESP32S3":
                    if now - last_frame > self.XIAO_TIMEOUT_SECONDS:
                        if now - last_sw >= self.COOLDOWN_SECONDS:
                            self.trigger_rollback(f"Señal de XIAO ausente por >{self.XIAO_TIMEOUT_SECONDS}s")
                else:
                    if now - last_sw >= self.COOLDOWN_SECONDS:
                        if self._check_xiao_socket(timeout=0.25):
                            self._try_connect_xiao()

            # 3. Modo XIAO forzado: NUNCA pasar a local
            elif mode == "xiao":
                # Si el XIAO se cayó o no está online, reintentar conexión en segundo plano
                if not online or (now - last_frame > self.XIAO_TIMEOUT_SECONDS):
                    with self.lock:
                        self.xiao_online = False
                        if self.xiao_reader and (now - last_frame > self.XIAO_TIMEOUT_SECONDS):
                            try:
                                self.xiao_reader.release()
                            except Exception:
                                pass
                            self.xiao_reader = None
                    # Reintentar conexión activamente sin límite cada ciclo
                    if self._check_xiao_socket(timeout=0.25):
                        self._try_connect_xiao(force=True)

            # 4. Auto-descubrimiento en segundo plano (Tanto en AUTO como en XIAO forzado)
            # Nivel A: Probar periódicamente IPs en caché persistente (Ultrarrápido < 100ms, seguro)
            if not online and not self._is_seeking and (now - self._last_fast_probe > 5.0):
                self._last_fast_probe = now
                cand = self.discovery.fast_probe_known(self.xiao_stream_port, self.xiao_ctrl_port)
                if cand:
                    if cand != self.xiao_ip:
                        self.update_xiao_ip(cand)
                    self._try_connect_xiao(force=True)
                    continue

                # Nivel B: Escaneo amplio de subredes solo tras 60s continuos de desconexión (previene saturación de sockets)
                if now - self._last_broad_scan > 60.0:
                    self._last_broad_scan = now
                    self._is_seeking = True

                    def _bg_seek():
                        try:
                            scanned = self.discovery.scan_network(self.xiao_stream_port, self.xiao_ctrl_port, max_workers=35)
                            if scanned:
                                if scanned != self.xiao_ip:
                                    self.update_xiao_ip(scanned)
                                self._try_connect_xiao(force=True)
                        except Exception:
                            pass
                        finally:
                            self._is_seeking = False

                    threading.Thread(target=_bg_seek, daemon=True).start()

    def read(self):
        now = time.monotonic()
        with self.lock:
            mode = self.camera_mode
            src = self.active_source
            xiao_r = self.xiao_reader
            local_r = self.local_reader

        # 1. Modo XIAO forzado: NUNCA lee de la cámara local
        if mode == "xiao":
            if xiao_r is not None:
                ok, frame = xiao_r.read()
                if ok and frame is not None and frame.size > 0:
                    self.last_xiao_frame_time = now
                    self.last_valid_frame = frame
                    return True, frame, "XIAO_ESP32S3"

            # Si el XIAO tuvo un micro-corte (< 3.0s), mantenemos el último frame congelado
            # sin parpadear ni mostrar la pantalla negra de búsqueda
            if now - self.last_xiao_frame_time <= 3.0 and self.last_valid_frame is not None:
                if now - self.last_xiao_frame_time > 1.2:
                    frozen = self.last_valid_frame.copy()
                    cv2.putText(frozen, "RECONECTANDO XIAO...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
                    time.sleep(0.015)
                    return True, frozen, "XIAO_ESP32S3_CONNECTING"
                time.sleep(0.005)
                return False, None, "XIAO_ESP32S3"

            # Si XIAO lleva más de 3.0s ausente por completo: pantalla de radar
            time.sleep(0.033)
            return True, self._generate_seeking_frame(), "XIAO_ESP32S3_CONNECTING"

        # 2. Modo AUTO con XIAO activo
        if src == "XIAO_ESP32S3":
            if xiao_r is not None:
                ok, frame = xiao_r.read()
                if ok and frame is not None and frame.size > 0:
                    self.last_xiao_frame_time = now
                    self.last_valid_frame = frame
                    return True, frame, "XIAO_ESP32S3"
                elif now - self.last_xiao_frame_time > self.XIAO_TIMEOUT_SECONDS:
                    if now - self.last_switch_time >= self.COOLDOWN_SECONDS:
                        self.trigger_rollback(f"Señal de XIAO ausente por >{self.XIAO_TIMEOUT_SECONDS}s")
            return False, None, "XIAO_ESP32S3"

        # 3. Modo LOCAL (o AUTO en rollback)
        if local_r is not None:
            ok, frame = local_r.read()
            if ok and frame is not None and frame.size > 0:
                return True, frame, "LOCAL_MAC"
            elif not local_r.is_alive_and_streaming():
                self._init_local_camera()

        return False, None, "LOCAL_MAC"

    def stop(self):
        self.running = False
        if self.xiao_reader:
            self.xiao_reader.release()
        self._release_local_camera()


# ---------------------------------------------------------------------------
# ORION Web Engine (Background Processing Core)
# ---------------------------------------------------------------------------

class OrionEngine:
    """
    Orquesta la ingesta de video (conmutada), inferencia MiDaS, alarmas y entrega de
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
        self.ai_fps = 0.0
        self.frame_count = 0
        self.ai_frame_count = 0
        self.current_camera_source = "LOCAL_MAC"

        # Frame and Depth Buffers (Decoupled sync)
        self.depth_lock = threading.Lock()
        self._latest_depth_norm = None
        self._latest_depth_vis = None

        # AI Worker Queue / Signal
        self._ai_input_lock = threading.Lock()
        self._pending_ai_frame = None
        self._pending_source = None
        self._new_frame_event = threading.Event()

        # Visual Frame Store & Lazy JPEG Cache
        self._frame_lock = threading.Lock()
        self.latest_frame_id = 0
        self._latest_raw_vis = None
        self._latest_depth_vis_drawn = None
        self._latest_combined_vis = None
        self._jpeg_cache = {}  # mode -> (frame_id, bytes)

        # ML Core initialization
        configure_torch(args.torch_threads)
        if args.cv_threads > 0:
            cv2.setNumThreads(args.cv_threads)

        self.device = resolve_device(args.device)
        print(f"🧠 Inicializando MiDaS en device: {self.device}...")
        self.model, self.transform, self.device = load_midas_torch(args.model_type, self.device)
        print(f"✅ MiDaS listo ({self.device.type})")

        # Xiao Buzzer Controller
        xiao_base = f"http://{args.xiao_ip}:{args.xiao_ctrl_port}"
        self.xiao_buzzer = XiaoBuzzerController(xiao_base, args.buzzer_duty)

        def _on_xiao_ip_resolved(new_ip: str):
            new_base = f"http://{new_ip}:{args.xiao_ctrl_port}"
            self.xiao_buzzer.update_base_url(new_base)

        # Audio Alarms Adaptativo
        self.local_sound = LocalAlarmSound(enabled=not args.no_local_sound)
        self.alarm = AdaptiveAlarmManager(
            self.xiao_buzzer,
            self.local_sound,
            buzzer_duty_max=args.buzzer_duty,
            buzzer_duration_ms=args.buzzer_duration,
        ).start()

        # Dual Source Camera Manager (XIAO + Mac Rollback) con Auto-Descubrimiento
        self.source_manager = DualCameraSourceManager(
            xiao_ip=args.xiao_ip,
            xiao_stream_port=args.xiao_stream_port,
            xiao_ctrl_port=args.xiao_ctrl_port,
            local_src=args.fallback_src,
            on_ip_resolved=_on_xiao_ip_resolved,
        )

        self.display_thread = threading.Thread(target=self._display_loop, name="DisplayThread", daemon=True)
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, name="AIWorkerThread", daemon=True)

    def start(self):
        self.running = True
        # Iniciar el hilo de IA primero para que inicialice CUDA en paralelo con la comprobación de red
        self.ai_thread.start()
        self.source_manager.start()
        self.display_thread.start()
        return self

    def _display_loop(self):
        """
        Hilo 1: Ingesta continua y visualización a la máxima tasa de la cámara (30+ FPS).
        Desacoplado de la inferencia MiDaS.
        """
        args = self.args
        fps_counter = 0
        fps_time = time.perf_counter()

        while self.running:
            ok, frame_bgr, active_source = self.source_manager.read()

            if not ok or frame_bgr is None:
                time.sleep(0.003)
                continue

            with self.lock:
                self.current_camera_source = active_source

            self.frame_count += 1
            proc_frame = frame_bgr

            # Resize if requested
            if args.proc_width > 0 and frame_bgr.shape[1] > args.proc_width:
                scale = args.proc_width / frame_bgr.shape[1]
                new_size = (args.proc_width, max(1, int(frame_bgr.shape[0] * scale)))
                proc_frame = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

            h, w = proc_frame.shape[:2]

            # Notify AI worker with the latest frame (non-blocking)
            with self._ai_input_lock:
                self._pending_ai_frame = proc_frame
                self._pending_source = active_source
            self._new_frame_event.set()

            # Retrieve latest depth visualization (non-blocking reference)
            with self.depth_lock:
                cur_depth_vis = self._latest_depth_vis

            if cur_depth_vis is None:
                cur_depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
            elif cur_depth_vis.shape[:2] != (h, w):
                cur_depth_vis = cv2.resize(cur_depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

            # Draw lane corridor overlays on raw and depth views
            third = w // 3

            raw_vis = proc_frame.copy()
            cv2.line(raw_vis, (third, 0), (third, h), (0, 200, 200), 1)
            cv2.line(raw_vis, (2 * third, 0), (2 * third, h), (0, 200, 200), 1)

            depth_vis_drawn = cur_depth_vis.copy()
            cv2.line(depth_vis_drawn, (third, 0), (third, h), (0, 200, 200), 1)
            cv2.line(depth_vis_drawn, (2 * third, 0), (2 * third, h), (0, 200, 200), 1)

            # Update latest visual arrays and advance frame ID (lazy cache invalidated)
            with self._frame_lock:
                self.latest_frame_id += 1
                self._latest_raw_vis = raw_vis
                self._latest_depth_vis_drawn = depth_vis_drawn
                self._latest_combined_vis = None  # Built lazily only when requested

            # Visual FPS calculation
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_time >= 1.0:
                with self.lock:
                    self.fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

    def _ai_worker_loop(self):
        """
        Hilo 2: Inferencia MiDaS asíncrona en GPU/MPS/CPU.
        Consume el último fotograma entregado por el hilo de visualización.
        """
        args = self.args
        prev_depth = None
        prev_source = None

        ai_counter = 0
        ai_time = time.perf_counter()

        mps_cache_supported = (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        )

        # Warmup inicial de cuDNN en el contexto específico de este hilo
        try:
            w_size = args.proc_width if args.proc_width > 0 else 320
            infer_depth_torch(self.model, self.transform, self.device, np.zeros((240, w_size, 3), dtype=np.uint8), use_fp16_mps=args.mps_fp16)
        except Exception:
            pass

        while self.running:
            if not self._new_frame_event.wait(timeout=0.05):
                continue
            self._new_frame_event.clear()

            with self._ai_input_lock:
                frame_to_proc = self._pending_ai_frame
                source_to_proc = self._pending_source
                self._pending_ai_frame = None

            if frame_to_proc is None:
                continue

            if source_to_proc != prev_source:
                prev_depth = None
                prev_source = source_to_proc

            try:
                proc_frame = frame_to_proc
                if args.deband and args.deband_ksize >= 3 and args.deband_ksize % 2 == 1:
                    proc_frame = cv2.GaussianBlur(proc_frame, (1, args.deband_ksize), 0)

                # Inferencia MiDaS
                raw_depth = infer_depth_torch(
                    self.model,
                    self.transform,
                    self.device,
                    proc_frame,
                    use_fp16_mps=args.mps_fp16,
                )
                depth_norm = normalize_depth(raw_depth, invert=args.invert_depth).astype(np.float32, copy=False)

                # Suavizado temporal acelerado SIMD con cv2.addWeighted (19x más rápido que NumPy)
                alpha = float(args.temporal_alpha)
                if 0.0 < alpha < 1.0:
                    if prev_depth is None or prev_depth.shape != depth_norm.shape:
                        prev_depth = depth_norm.copy()
                    else:
                        cv2.addWeighted(prev_depth, 1.0 - alpha, depth_norm, alpha, 0.0, dst=prev_depth)
                    depth_used = prev_depth
                else:
                    depth_used = depth_norm

                # Análisis de proximidad y pasillo central
                action, intensity, center_prox, overall_prox = analyze_depth(
                    depth_used,
                    near_threshold=self.near_threshold,
                )

                # Modulación de alarmas
                self.alarm.update(
                    active=(action == "STOP"),
                    intensity=intensity,
                    current_source=source_to_proc,
                )

                # Colorización Magma
                depth_vis = colorize_depth(depth_used)

                # Actualización atómica de resultados
                with self.depth_lock:
                    self._latest_depth_norm = depth_used
                    self._latest_depth_vis = depth_vis
                    self.last_action = action
                    self.last_intensity = intensity
                    self.last_center_prox = center_prox
                    self.last_overall_prox = overall_prox

                self.ai_frame_count += 1
                ai_counter += 1
            except Exception as e:
                safe_print(f"⚠️ [AI-WORKER] Error recuperable en inferencia: {e}")
            now = time.perf_counter()
            if now - ai_time >= 1.0:
                with self.lock:
                    self.ai_fps = ai_counter / (now - ai_time)
                ai_counter = 0
                ai_time = now

            if (
                mps_cache_supported
                and args.mps_empty_cache_every > 0
                and self.ai_frame_count % args.mps_empty_cache_every == 0
            ):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    def get_jpeg(self, mode: str = "combined") -> bytes | None:
        """
        Codificación JPEG perezosa (Lazy Encoding): solo codifica la vista que
        el cliente está solicitando actualmente, cacheando por frame_id.
        Ahorra ~66% de tiempo de CPU en compresión JPEG.
        """
        with self._frame_lock:
            cur_id = self.latest_frame_id
            if cur_id == 0:
                return None

            cached = self._jpeg_cache.get(mode)
            if cached is not None and cached[0] == cur_id:
                return cached[1]

            if mode == "raw":
                img = self._latest_raw_vis
            elif mode == "depth":
                img = self._latest_depth_vis_drawn
            else:  # "combined"
                if self._latest_combined_vis is None and self._latest_raw_vis is not None and self._latest_depth_vis_drawn is not None:
                    self._latest_combined_vis = np.hstack((self._latest_raw_vis, self._latest_depth_vis_drawn))
                img = self._latest_combined_vis

            if img is None:
                return None

            _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            jpg_bytes = encoded.tobytes()
            self._jpeg_cache[mode] = (cur_id, jpg_bytes)
            return jpg_bytes

    def get_telemetry(self) -> dict:
        with self.lock:
            with self.depth_lock:
                act = self.last_action
                inten = self.last_intensity
                c_prox = self.last_center_prox
                o_prox = self.last_overall_prox
            return {
                "action": act,
                "center_prox": round(c_prox, 4),
                "overall_prox": round(o_prox, 4),
                "intensity": round(inten, 4),
                "near_threshold": round(self.near_threshold, 3),
                "fps": round(self.fps, 1),
                "ai_fps": round(self.ai_fps, 1),
                "device": self.device.type,
                "frame_count": self.frame_count,
                "ai_frame_count": self.ai_frame_count,
                "camera_source": self.current_camera_source,
                "camera_mode": self.source_manager.camera_mode,
                "xiao_online": self.source_manager.xiao_online,
                "camera_ip": self.args.xiao_ip,
                "local_sound": self.local_sound.enabled,
                "esp_buzzer": self.alarm.esp_buzzer_enabled,
                "esp_buzzer_enabled": self.alarm.esp_buzzer_enabled,
                "timestamp": time.time(),
            }

    def update_settings(self, settings: dict):
        with self.lock:
            if "camera_ip" in settings:
                new_ip = str(settings["camera_ip"]).strip()
                if new_ip and new_ip != self.args.xiao_ip:
                    self.args.xiao_ip = new_ip
                    self.source_manager.update_xiao_ip(new_ip)
                    self.source_manager._try_connect_xiao(force=True)
            if "near_threshold" in settings:
                self.near_threshold = max(0.05, min(0.95, float(settings["near_threshold"])))
            if "local_sound_enabled" in settings:
                self.local_sound.enabled = bool(settings["local_sound_enabled"])
            if "esp_buzzer_enabled" in settings:
                self.alarm.esp_buzzer_enabled = bool(settings["esp_buzzer_enabled"])
            if "camera_mode" in settings:
                self.source_manager.set_mode(str(settings["camera_mode"]))

    def stop(self):
        self.running = False
        self._new_frame_event.set()
        if hasattr(self, "display_thread") and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        if hasattr(self, "ai_thread") and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        self.source_manager.stop()
        self.alarm.stop()
        self.xiao_buzzer.close()


# ---------------------------------------------------------------------------
# FastAPI Web App
# ---------------------------------------------------------------------------

app = FastAPI(title="ORION Web Dashboard")
engine: OrionEngine | None = None
server_instance: uvicorn.Server | None = None


@app.on_event("startup")
async def startup_event():
    try:
        loop = asyncio.get_running_loop()
        def silence_proactor_error(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)) or (
                isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10054
            ):
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(silence_proactor_error)
    except Exception:
        pass


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
        last_sent_id = -1
        try:
            while engine and engine.running and not getattr(server_instance, "should_exit", False):
                cur_id = engine.latest_frame_id
                if cur_id != last_sent_id:
                    jpg = engine.get_jpeg(mode)
                    if jpg is not None:
                        last_sent_id = cur_id
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                        )
                await asyncio.sleep(0.015)  # Cap a ~60 FPS para máxima fluidez sin saturar CPU
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
        description="ORION Web Service: XIAO ESP32S3 Sense con Hot-Swap y Rollback a Cámara Mac"
    )

    # XIAO ESP32S3 & Fallback Configuration
    parser.add_argument("--xiao-ip", default=None, help="Dirección IP del XIAO ESP32S3 Sense (por defecto usa la última confirmada en caché)")
    parser.add_argument("--xiao-stream-port", type=int, default=81, help="Puerto del stream MJPEG del XIAO (default 81)")
    parser.add_argument("--xiao-ctrl-port", type=int, default=80, help="Puerto HTTP de control del XIAO (default 80)")
    parser.add_argument("--scan-xiao", action="store_true", help="Escanear ampliamente la red local en busca del XIAO ESP32S3")
    parser.add_argument("--fallback-src", default="0", help="Fuente local de respaldo (0 para cámara web integrada)")

    # Server Args
    parser.add_argument("--host", default="0.0.0.0", help="Host IP del servidor web (0.0.0.0 para acceso en red local)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP del servidor web (default 8000)")

    # Video Source & AI Core
    parser.add_argument("--near-threshold", type=float, default=0.6, help="Umbral relativo de proximidad (0.0 a 1.0)")
    parser.add_argument("--invert-depth", action="store_true")
    parser.add_argument("--model-type", default=MODEL_TYPE_DEFAULT, choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"])

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

    return parser.parse_args()


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_available_port(host: str, preferred_port: int) -> int:
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

    # Detección y resolución automática de IP (Caché conocida -> Escaneo dinámico)
    disc = XiaoDiscoveryManager()
    resolved = disc.resolve_xiao_ip(
        preferred_ip=args.xiao_ip,
        stream_port=args.xiao_stream_port,
        ctrl_port=args.xiao_ctrl_port,
        enable_broad_scan=False,
    )
    if resolved:
        args.xiao_ip = resolved

    # Iniciar motor autónomo
    engine = OrionEngine(args).start()

    local_ip = get_local_ip()
    chosen_port = find_available_port(args.host, args.port)

    if chosen_port != args.port:
        print(f"⚠️ El puerto {args.port} está reservado u ocupado. Usando puerto alternativo libre: {chosen_port}")

    # Guardar el puerto elegido en la raíz del proyecto para que el script .sh
    # pueda detectar que ORION arrancó y abrir la URL automáticamente.
    port_file = PROJECT_ROOT / ".orion_port"
    try:
        port_file.write_text(str(chosen_port), encoding="utf-8")
    except OSError as exc:
        print(f"⚠️ No se pudo guardar el puerto de ORION: {exc}")

    print("\n" + "=" * 65)
    print("🌌 ORION Web Service (XIAO ESP32S3 Sense + Rollback Mac)")
    print(f"   👉 Cámara Primaria:     {engine.source_manager.xiao_stream_url}")
    print(f"   👉 IPs en Caché:        {engine.source_manager.discovery.get_known_ips()[:3]}")
    print(f"   👉 Cámara Respaldo:     Cámara integrada (índice {args.fallback_src})")
    print(f"   👉 Acceso Local:        http://localhost:{chosen_port}")
    print(f"   👉 Acceso en Red WiFi:  http://{local_ip}:{chosen_port}")
    print("   Presiona 'Ctrl + C' en la terminal para detener el servidor.")
    print("=" * 65 + "\n")

    # Run Uvicorn Server with instant shutdown
    global server_instance
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=chosen_port,
        log_level="info",
        access_log=False,
        timeout_graceful_shutdown=0.5,
    )
    server_instance = uvicorn.Server(config)

    # Abrir navegador automáticamente (protegido si corre como servicio de fondo)
    def _open_browser():
        time.sleep(2.0)
        try:
            webbrowser.open(f"http://localhost:{chosen_port}")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        server_instance.run()
    except (KeyboardInterrupt, SystemExit):
        safe_print("\n⏹️  Deteniendo servidor...")
    except Exception as e:
        safe_print(f"\n❌ Error crítico en el servidor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.stop()
        try:
            if port_file.exists():
                port_file.unlink()
        except Exception:
            pass
        safe_print("✅ ORION Web Service detenido.")


if __name__ == "__main__":
    main()
