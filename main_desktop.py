import argparse
import ipaddress
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
from urllib.parse import urlparse

# Debe fijarse ANTES de que MiDaS ejecute cualquier operación en MPS.
# Si una operación no está implementada para Apple Silicon, PyTorch
# la ejecuta automáticamente en CPU en vez de lanzar una excepción.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except ImportError:
    _HAS_SIMPLEAUDIO = False


MODEL_TYPE_DEFAULT = "MiDaS_small"
WINDOW_NAME = "ORION v3 | Frame + Depth (Telemetry HUD)"
MIN_WINDOW_WIDTH = 1000  # Tamaño mínimo fijo de la pestaña, en píxeles.


# ---------------------------------------------------------------------------
# Torch / MiDaS
# ---------------------------------------------------------------------------

def configure_torch(threads: int = 0) -> None:
    """Configure PyTorch before loading the model."""
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
    """Chequeo robusto de disponibilidad de MPS (Apple Silicon)."""
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False

    is_built = getattr(backend, "is_built", lambda: True)()
    is_available = getattr(backend, "is_available", lambda: False)()
    return bool(is_built and is_available)


def resolve_device(preferred: str = "auto") -> torch.device:
    """
    Resuelve el backend de cómputo a usar.

    'auto' prioriza CUDA (si existe) y luego MPS (Apple Silicon),
    que es el caso relevante para un Mac M1/M2/M3/M4.
    """
    preferred = (preferred or "auto").lower()

    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("⚠️ CUDA solicitado pero no disponible; usando CPU.")
        return torch.device("cpu")

    if preferred == "mps":
        if _mps_available():
            return torch.device("mps")
        print("⚠️ MPS solicitado pero no disponible; usando CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_midas_torch(model_type: str, device: torch.device):
    """Load MiDaS once and return model and transform ya en el device elegido."""
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
    """MiDaS inference without autograd and with reduced allocations."""
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
    """
    Normaliza la profundidad relativa/disparidad inversa de MiDaS al rango [0.0, 1.0].
    
    - 0.0: Punto más lejano / fondo de la escena observada.
    - 1.0: Objeto más cercano al lente en la escena observada.
    """
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
    """
    Analiza el mapa de profundidad normalizado.
    
    Devuelve:
      - action: 'STOP' o 'FORWARD'
      - intensity: float (0.0 a 1.0) indicando qué tan por encima del umbral está
      - center_mean: proximidad relativa promedio en el tercio central (0.0 a 1.0)
      - overall_mean: proximidad promedio de toda la escena (0.0 a 1.0)
    """
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
# Visualización y Renderizado de Texto / Telemetría 
# ---------------------------------------------------------------------------

def draw_regions_and_action(
    frame: np.ndarray,
    action: str,
    center_prox: float = 0.0,
    near_threshold: float = 0.6,
) -> None:
    """
    Dibuja los delimitadores de carril, el estado de acción con tamaño de letra
    reducido y una barra de telemetría de proximidad medida.
    """
    h, w = frame.shape[:2]
    third = w // 3

    # Líneas divisorias de carril central (amarillas tenues)
    cv2.line(frame, (third, 0), (third, h), (0, 200, 200), 1)
    cv2.line(frame, (2 * third, 0), (2 * third, h), (0, 200, 200), 1)

    action_color = (0, 255, 0) if action == "FORWARD" else (0, 0, 255)
    cv2.putText(
        frame,
        f"Action: {action}",
        (10, 20),                  # Coordenadas (X, Y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Tipografía
        0.39,                       # Tamaño del texto 
        action_color,              # Color (Verde para FORWARD, Rojo para STOP)
        1,                         # Grosor del texto
        cv2.LINE_AA,               # Anti-aliasing suave
    )

    # Telemetría de Proximidad Medida (Porcentaje relativo y umbral)
    prox_percent = center_prox * 100.0
    thresh_percent = near_threshold * 100.0
    cv2.putText(
        frame,
        f"Prox: {prox_percent:4.1f}% | Umbral: {thresh_percent:.0f}%",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.29,                      # Tamaño de fuente pequeño para métricas
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    # Barra gráfica de proximidad
    bar_x = 10
    bar_y = 44
    bar_w = min(120, int(third * 0.8))
    bar_h = 6
    
    # Fondo de la barra (gris oscuro)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    
    # Relleno de la barra según nivel de proximidad
    fill_w = int(bar_w * min(1.0, max(0.0, center_prox)))
    fill_color = (0, 220, 0) if center_prox < near_threshold else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    
    # Borde de la barra y marca del umbral
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    thresh_x = bar_x + int(bar_w * near_threshold)
    cv2.line(frame, (thresh_x, bar_y - 1), (thresh_x, bar_y + bar_h + 1), (0, 255, 255), 1)


# ---------------------------------------------------------------------------
# Alertas de sonido (local en el Mac + buzzer del ESP32)
# ---------------------------------------------------------------------------

def _generate_tone_pcm(
    frequency: float,
    duration: float,
    volume: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Genera un beep senoidal con envolvente suave (evita 'clics')."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)

    fade_len = max(1, int(sample_rate * 0.01))
    envelope = np.ones_like(tone)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    tone = tone * envelope * volume
    return np.int16(np.clip(tone, -1.0, 1.0) * 32767)


class LocalAlarmSound:
    """
    Alerta sonora reproducida en el propio ordenador. Define 4 niveles de
    urgencia: entre más cerca esté el obstáculo, más agudo, más fuerte
    y más frecuente es el beep.
    """

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
        elif enabled:
            print(
                "⚠️ No se encontró un backend de audio en este equipo "
                "(ni 'simpleaudio' ni 'afplay'); la alerta LOCAL queda "
                "desactivada, pero el buzzer del ESP32 sigue funcionando."
            )

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
        if self.backend is None:
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
    """
    Orquesta la alerta iterativa (sonido local + buzzer del ESP32)
    mientras el estado sea STOP en un hilo dedicado.
    """

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
                if was_active and self.esp is not None:
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

            if self.esp is not None:
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
# Capture readers
# ---------------------------------------------------------------------------

class LatestFrameReader:
    """Always retains only the newest frame, dropping stale frames."""

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
    """Background MJPEG reader retaining only the newest decoded frame."""

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
    """Raw HTTP/TCP MJPEG reader with TCP_NODELAY."""

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
    """Reads 4-byte big-endian JPEG length followed by JPEG payload."""

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
        return "192.168.1.1"


def get_network_range():
    local_ip = get_local_ip()
    print(f"🌐 IP local detectada: {local_ip}")
    network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    print(f"📡 Rango de red: {network}")
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
    print(f"📡 Escaneando {len(network_ips)} direcciones IP...")

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
    """Persistent HTTP session para el microcontrolador ESP32-CAM."""

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
            return self.session.get(
                f"{self.base_url}{path}",
                timeout=timeout,
            )
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ORION v3: MiDaS depth + ESP32-CAM (Telemetry HUD & Compact UI)"
    )

    parser.add_argument("--src", default="0")
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.6,
        help="Umbral relativo de proximidad en zona central para STOP (0.0 a 1.0)",
    )
    parser.add_argument("--invert-depth", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save", default="")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument(
        "--model-type",
        default=MODEL_TYPE_DEFAULT,
        choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"],
    )
    parser.add_argument("--esp-base", default="")
    parser.add_argument("--auto-find", action="store_true")
    parser.add_argument("--led-duty", type=int, default=255)
    parser.add_argument("--buzzer-duty", type=int, default=200)
    parser.add_argument("--buzzer-duration", type=int, default=150)
    parser.add_argument("--display-width", type=int, default=1900)
    parser.add_argument("--display-scale", type=float, default=1.0)

    # Lectura asíncrona por defecto para baja latencia
    parser.add_argument("--async-capture", action="store_true", default=True)
    parser.add_argument(
        "--sync-capture",
        action="store_true",
        help="Disable latest-frame capture for debugging.",
    )

    parser.add_argument("--cap-bufsize", type=int, default=1)
    parser.add_argument("--mjpeg-reader", action="store_true")
    parser.add_argument("--socket-mjpeg", action="store_true")
    parser.add_argument("--raw-socket", action="store_true")
    parser.add_argument("--raw-port", type=int, default=3333)
    parser.add_argument("--esp-framesize", type=int, default=6)
    parser.add_argument("--esp-quality", type=int, default=15)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="0=automatic; for CPU/MPS-fallback ops try 2-6.",
    )
    parser.add_argument("--deband", action="store_true")
    parser.add_argument("--deband-ksize", type=int, default=5)
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.65,
        help="EMA weight for current depth. Higher = lower latency.",
    )
    parser.add_argument(
        "--proc-width",
        type=int,
        default=320,
        help="Width used by MiDaS. Lower = faster.",
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=1,
        help="Run MiDaS every N captured frames.",
    )
    parser.add_argument(
        "--no-local-sound",
        action="store_true",
        help="Disable PC alert sound.",
    )

    # --- Hardware Mac M1 / Apple Silicon ---
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Backend de cómputo. 'auto' usa MPS en Apple Silicon si está disponible.",
    )
    parser.add_argument(
        "--mps-fp16",
        action="store_true",
        help="Intenta autocast FP16 en MPS (experimental).",
    )
    parser.add_argument(
        "--cv-threads",
        type=int,
        default=0,
        help="Fuerza el número de hilos internos de OpenCV.",
    )
    parser.add_argument(
        "--mps-empty-cache-every",
        type=int,
        default=300,
        help="Cada cuántos frames liberar caché de MPS (0 = nunca).",
    )

    return parser.parse_args()


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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.auto_find:
        esp_ip = auto_find_esp32()
        if not esp_ip:
            sys.exit(1)

        args.src = f"http://{esp_ip}:81/stream"
        args.esp_base = f"http://{esp_ip}"
        print(f"🎯 Usando ESP32-CAM: {esp_ip}")

    configure_torch(args.torch_threads)

    if args.cv_threads > 0:
        cv2.setNumThreads(args.cv_threads)

    device = resolve_device(args.device)

    print("🧠 Cargando MiDaS...")
    model, transform, device = load_midas_torch(args.model_type, device)
    print(f"✅ Device: {device}")

    if device.type == "cuda":
        print("🚀 CUDA/FP16 habilitado")
    elif device.type == "mps":
        fp16_note = "FP16 (experimental)" if args.mps_fp16 else "FP32 (estable)"
        print(f"🍎 Apple Silicon (MPS) habilitado — {fp16_note}")
    else:
        print("🖥️ CPU")

    esp_base = args.esp_base.strip() or infer_esp_base_from_src(args.src)
    esp = EspController(esp_base, args.buzzer_duty) if esp_base else None

    if esp:
        esp.set_param("quality", args.esp_quality)
        esp.set_param("framesize", args.esp_framesize)

    local_sound = LocalAlarmSound(enabled=not args.no_local_sound)
    alarm = AlarmManager(
        esp,
        local_sound,
        buzzer_duty_max=args.buzzer_duty,
        buzzer_duration_ms=args.buzzer_duration,
    ).start()

    if local_sound.enabled and local_sound.backend:
        print(f"🔊 Alerta local activa (backend: {local_sound.backend})")
    elif args.no_local_sound:
        print("🔇 Alerta local desactivada por --no-local-sound")

    capture = None
    async_reader = None
    stream_reader = None

    if args.raw_socket:
        base = args.esp_base.strip() or infer_esp_base_from_src(args.src)
        host = urlparse(base or args.src).hostname

        if not host:
            raise RuntimeError("No se pudo obtener el host para raw socket.")

        try:
            stream_reader = RawLenSocketReader(host, args.raw_port).start()
        except Exception as exc:
            print(f"⚠️ Raw socket no disponible: {exc}")

    elif args.socket_mjpeg and args.src.startswith(("http://", "https://")):
        try:
            stream_reader = SocketMjpegReader(args.src).start()
        except Exception as exc:
            print(f"⚠️ Socket MJPEG no disponible: {exc}")

    elif args.mjpeg_reader and args.src.startswith(("http://", "https://")):
        try:
            stream_reader = MjpegReader(args.src).start()
        except Exception as exc:
            print(f"⚠️ MJPEG reader no disponible: {exc}")

    if stream_reader is None:
        capture = open_capture(args.src, args.cap_bufsize)

        if not capture.isOpened():
            print(f"❌ No se pudo abrir la fuente: {args.src}")
            return

        if args.async_capture and not args.sync_capture:
            async_reader = LatestFrameReader(capture).start()

    video_writer = None

    if args.save:
        if stream_reader is not None:
            ok, first_frame = stream_reader.read()
        elif async_reader is not None:
            ok, first_frame = async_reader.read()
        else:
            ok, first_frame = capture.read()

        if not ok:
            raise RuntimeError("No se pudo obtener el primer frame para guardar video.")

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))

    frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
    next_frame_time = time.perf_counter()

    prev_depth = None
    last_depth = None
    last_depth_norm = None
    last_action = "FORWARD"
    last_intensity = 0.0
    last_center_prox = 0.0

    frame_counter = 0
    fps_counter = 0
    fps_time = time.perf_counter()
    displayed_fps = 0.0

    mps_cache_supported = (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    )

    window_ready = [False]

    print("▶️ ORION v3 iniciado. Presiona 'q' para salir.")

    try:
        while True:
            if frame_interval:
                now = time.perf_counter()
                delay = next_frame_time - now

                if delay > 0:
                    time.sleep(delay)

                next_frame_time = max(
                    next_frame_time + frame_interval,
                    time.perf_counter(),
                )

            if stream_reader is not None:
                ok, frame_bgr = stream_reader.read()
            elif async_reader is not None:
                ok, frame_bgr = async_reader.read()
            else:
                ok, frame_bgr = capture.read()

            if not ok or frame_bgr is None:
                time.sleep(0.002)
                continue

            frame_counter += 1

            proc_frame = frame_bgr

            if args.proc_width > 0 and frame_bgr.shape[1] > args.proc_width:
                scale = args.proc_width / frame_bgr.shape[1]

                new_size = (
                    args.proc_width,
                    max(1, int(frame_bgr.shape[0] * scale)),
                )

                proc_frame = cv2.resize(
                    frame_bgr,
                    new_size,
                    interpolation=cv2.INTER_AREA,
                )

            if (
                args.deband
                and args.deband_ksize >= 3
                and args.deband_ksize % 2 == 1
            ):
                proc_frame = cv2.GaussianBlur(
                    proc_frame,
                    (1, args.deband_ksize),
                    0,
                )

            run_inference = (
                last_depth is None
                or args.inference_every <= 1
                or frame_counter % args.inference_every == 0
            )

            if run_inference:
                last_depth = infer_depth_torch(
                    model,
                    transform,
                    device,
                    proc_frame,
                    use_fp16_mps=args.mps_fp16,
                )

                depth_norm = normalize_depth(
                    last_depth,
                    invert=args.invert_depth,
                )

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

                last_action, last_intensity, last_center_prox, _overall = analyze_depth(
                    depth_used,
                    near_threshold=args.near_threshold,
                )

                alarm.update(active=(last_action == "STOP"), intensity=last_intensity)

            else:
                depth_used = last_depth_norm

            if (
                mps_cache_supported
                and args.mps_empty_cache_every > 0
                and frame_counter % args.mps_empty_cache_every == 0
            ):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            if not args.no_gui or video_writer is not None:
                depth_vis = colorize_depth(depth_used)
                
                # Renderizado de carril, acción con tamaño compacto y telemetría de proximidad
                draw_regions_and_action(
                    depth_vis,
                    last_action,
                    center_prox=last_center_prox,
                    near_threshold=args.near_threshold,
                )

                stacked = np.hstack((proc_frame, depth_vis))

                if not args.no_gui:
                    vis = stacked

                    target_width = max(args.display_width, MIN_WINDOW_WIDTH)

                    if target_width > 0 and vis.shape[1] != target_width:
                        scale = target_width / vis.shape[1]
                        interpolation = (
                            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                        )

                        vis = cv2.resize(
                            vis,
                            (
                                max(1, int(vis.shape[1] * scale)),
                                max(1, int(vis.shape[0] * scale)),
                            ),
                            interpolation=interpolation,
                        )

                    if not window_ready[0]:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME, vis.shape[1], vis.shape[0])
                        window_ready[0] = True

                    cv2.imshow(WINDOW_NAME, vis)

                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                if video_writer is not None:
                    video_writer.write(stacked)

            fps_counter += 1
            now = time.perf_counter()

            if now - fps_time >= 1.0:
                displayed_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            sys.stdout.write(
                f"\rDecision: {last_action:<7} | Prox: {last_center_prox*100:4.1f}% | "
                f"FPS: {displayed_fps:5.1f} | Device: {device.type} "
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nInterrumpido.")

    finally:
        if async_reader is not None:
            async_reader.stop()

        if capture is not None:
            capture.release()

        if stream_reader is not None:
            stream_reader.release()

        if video_writer is not None:
            video_writer.release()

        alarm.stop()

        if esp is not None:
            esp.set_buzzer(False)
            esp.close()

        if not args.no_gui:
            cv2.destroyAllWindows()

        print("\n✅ ORION v3 finalizado.")


if __name__ == "__main__":
    main()
