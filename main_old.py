import os
import sys
import time
import argparse
import threading
import re
from pathlib import Path
from typing import Tuple, List, Optional
import socket
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import requests
from urllib.parse import urlparse
import os

MODEL_TYPE_DEFAULT = "MiDaS_small"  # fast, CPU-friendly


def load_midas_torch(model_type: str = MODEL_TYPE_DEFAULT):
    """Load MiDaS via Torch Hub and return (model, transform, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return model, transform, device


def infer_depth_torch(model, transform, device, frame_bgr: np.ndarray) -> np.ndarray:
    """Run MiDaS inference using Torch and return depth resized to original frame size."""
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(image_rgb.shape[0], image_rgb.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)
    depth = prediction.detach().cpu().numpy().astype(np.float32)
    return depth


    


def normalize_depth(depth_raw: np.ndarray, invert: bool) -> np.ndarray:
    """Normalize depth to 0..1, where 1.0 means closer if invert=False (default MiDaS is inverse depth).

    If your device produces the opposite mapping, pass invert=True.
    """
    min_value = float(np.min(depth_raw))
    max_value = float(np.max(depth_raw))
    if max_value - min_value < 1e-6:
        normalized = np.zeros_like(depth_raw, dtype=np.float32)
    else:
        normalized = (depth_raw - min_value) / (max_value - min_value)
    if invert:
        normalized = 1.0 - normalized
    return normalized


def decide_navigation_action(depth_norm: np.ndarray, near_threshold: float = 0.6) -> str:
    """Decide a simple navigation action based on mean proximity in center region.

    Returns one of: 'FORWARD', 'STOP'.
    """
    height, width = depth_norm.shape[:2]
    third = width // 3
    center_region = depth_norm[:, third : 2 * third]

    center_mean = float(np.mean(center_region))
    overall_mean = float(np.mean(depth_norm))
    
    if overall_mean > 0.9 or center_mean > near_threshold:
        return "STOP"

    return "FORWARD"


def colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Convert 0..1 depth to a colorful heatmap for visualization."""
    depth_uint8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    return colored


def draw_regions_and_action(frame: np.ndarray, action: str) -> None:
    """Draw center region separator and current action label on the frame in-place."""
    height, width = frame.shape[:2]
    third = width // 3
    color_line = (0, 255, 255)
    cv2.line(frame, (third, 0), (third, height), color_line, 1)
    cv2.line(frame, (2 * third, 0), (2 * third, height), color_line, 1)
    cv2.putText(
        frame,
        f"Action: {action}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if action == "FORWARD" else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-based obstacle detection with MiDaS (Torch Hub)")
    parser.add_argument("--src", type=str, default="0", help="Video source: camera index (e.g., 0) or URL (e.g., http://IP:81/stream)")
    parser.add_argument("--near-threshold", type=float, default=0.6, help="Threshold on 0..1 proximity for obstacle in center")
    parser.add_argument("--invert-depth", action="store_true", help="Invert normalized depth if mapping is reversed")
    parser.add_argument("--no-gui", action="store_true", help="Disable visualization windows")
    parser.add_argument("--save", type=str, default="", help="Optional path to save annotated video (e.g., output.mp4)")
    parser.add_argument("--fps", type=float, default=0.0, help="Target processing FPS (0 = no limit)")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE_DEFAULT, choices=["MiDaS_small", "DPT_Large", "DPT_Hybrid"], help="MiDaS model variant")
    parser.add_argument("--esp-base", type=str, default="", help="Base URL of ESP32-CAM control server (e.g., http://192.168.1.42). If empty and --src is a URL, it will be inferred.")
    parser.add_argument("--auto-find", action="store_true", help="Automatically find ESP32-CAM on local network")
    parser.add_argument("--led-duty", type=int, default=255, help="LED intensity 0..255 when obstacle detected")
    parser.add_argument("--buzzer-duty", type=int, default=200, help="Buzzer intensity 0..255 when obstacle detected (default 200 for louder sound)")
    parser.add_argument("--buzzer-duration", type=int, default=150, help="Buzzer beep duration in milliseconds when obstacle detected")
    parser.add_argument("--display-width", type=int, default=0, help="Resize visualization to this width in pixels (0 = original width)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Resize visualization by this scale factor (used only if --display-width is 0)")
    parser.add_argument("--async-capture", action="store_true", help="Use a background reader that always keeps the latest frame (drops old frames) to reduce latency")
    parser.add_argument("--cap-bufsize", type=int, default=1, help="Try to set OpenCV capture buffer size (if backend supports it)")
    parser.add_argument("--proc-width", type=int, default=480, help="Resize incoming frames to this width for processing/visualization (keeps aspect). Set 0 to use source size.")
    parser.add_argument("--mjpeg-reader", action="store_true", help="Use a custom HTTP MJPEG reader for lower latency on ESP32 streams (requests)")
    parser.add_argument("--socket-mjpeg", action="store_true", help="Use a raw TCP socket MJPEG reader (lowest overhead)")
    parser.add_argument("--raw-socket", action="store_true", help="Use raw length-prefixed JPEG socket reader (matches ESP raw socket server)")
    parser.add_argument("--raw-port", type=int, default=3333, help="Port for raw socket stream (default 3333)")
    parser.add_argument("--esp-framesize", type=int, default=6, help="If ESP base is known, set framesize via /control (e.g., 6=VGA 640x480, 5=CIF, 4=QVGA)")
    parser.add_argument("--esp-quality", type=int, default=12, help="If ESP base is known, set JPEG quality via /control (lower is better quality; typical 10-20)")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set torch.set_num_threads to limit CPU contention (0 = leave default)")
    parser.add_argument("--deband", action="store_true", help="Apply a light vertical blur to suppress horizontal stripe noise before depth")
    parser.add_argument("--deband-ksize", type=int, default=5, help="Odd kernel size for vertical blur (e.g., 3,5,7)")
    parser.add_argument("--temporal-alpha", type=float, default=0.4, help="Temporal EMA smoothing on depth (0=no smoothing, 1=only current)")
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    # Interpret numeric strings as camera indices
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)
    # Reduce internal buffering where supported
    try:
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, float(args.cap_bufsize) if 'args' in globals() else 1)
    except Exception:
        pass
    return capture


class LatestFrameReader:
    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.latest = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None  # optional: allow skipping to newest
        return frame is not None, frame

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=0.2)
        except Exception:
            pass


def get_local_ip() -> str:
    """Obtiene la IP local de la máquina."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "192.168.1.1"  # fallback

def get_network_range() -> List[str]:
    """Obtiene el rango de red local."""
    local_ip = get_local_ip()
    print(f"🌐 IP local detectada: {local_ip}")
    
    # Detectar el rango de red apropiado
    if local_ip.startswith("10."):
        # Red clase A (10.x.x.x)
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    elif local_ip.startswith("172."):
        # Red clase B (172.16.x.x - 172.31.x.x)
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    elif local_ip.startswith("192.168"):
        # Red clase C (192.168.x.x)
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    else:
        # Fallback
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    
    print(f"📡 Rango de red: {network}")
    return [str(ip) for ip in network.hosts()]

def check_esp32_port(ip: str, port: int = 81, timeout: float = 2) -> bool:
    """Verifica si un puerto está abierto en una IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((ip, port))
            return result == 0
    except:
        return False

def check_esp32_stream(ip: str, timeout: float = 5) -> bool:
    """Verifica si hay un stream ESP32 en la IP."""
    try:
        url = f"http://{ip}:81/stream"
        # Usar timeout más largo para ESP32-CAM
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code == 200:
            # ESP32-CAM streams pueden tener diferentes content-types
            content_type = response.headers.get('content-type', '').lower()
            print(f"📄 Content-Type para {ip}: {content_type}")
            
            # Verificar si es un stream de video válido
            if any(x in content_type for x in ['multipart', 'mjpeg', 'video', 'image']):
                return True
            # También verificar si el stream responde (incluso sin content-type específico)
            if len(content_type) == 0 or 'text/html' not in content_type:
                return True
            # Si responde con 200, probablemente es ESP32-CAM
            return True
    except Exception as e:
        print(f"Error checking stream {ip}: {e}")
    return False

def check_esp32_control(ip: str, timeout: float = 2) -> bool:
    """Verifica si hay un servidor de control ESP32 en la IP."""
    try:
        url = f"http://{ip}/control"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def auto_find_esp32() -> Optional[str]:
    """Encuentra automáticamente ESP32-CAM en la red local."""
    print("🔍 Escaneando red local en busca de ESP32-CAM...")
    
    # Primero probar IPs conocidas comunes
    known_ips = ["10.35.132.231", "172.16.121.9", "192.168.1.100", "192.168.0.100", "192.168.1.1"]
    print("🎯 Probando IPs conocidas...")
    
    for ip in known_ips:
        print(f"🔍 Probando {ip}...")
        if check_esp32_port(ip, 81, 1):
            print(f"✅ Puerto 81 abierto en {ip}")
            print(f"🔍 Verificando stream en {ip}...")
            if check_esp32_stream(ip, 5) or check_esp32_control(ip, 3):
                print(f"🎉 ESP32-CAM encontrado en {ip}")
                return ip
            else:
                print(f"⚠️  Puerto abierto pero no es ESP32-CAM: {ip}")
    
    # Si no se encuentra en IPs conocidas, escanear red completa
    network_ips = get_network_range()
    print(f"📡 Escaneando {len(network_ips)} direcciones IP...")
    
    found_ips = []
    
    # Escanear puertos con menos workers para ser más estable
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ip = {
            executor.submit(check_esp32_port, ip, 81, 1): ip 
            for ip in network_ips
        }
        
        for future in as_completed(future_to_ip):
            ip = future_to_ip[future]
            try:
                if future.result():
                    found_ips.append(ip)
                    print(f"✅ Puerto 81 abierto en {ip}")
            except Exception as e:
                print(f"Error checking {ip}: {e}")
    
    if not found_ips:
        print("❌ No se encontraron dispositivos con puerto 81 abierto")
        return None
    
    # Verificar cuáles son realmente ESP32-CAM
    print("🔍 Verificando si son ESP32-CAM...")
    esp32_ips = []
    
    for ip in found_ips:
        print(f"🔍 Verificando {ip}...")
        if check_esp32_stream(ip, 5) or check_esp32_control(ip, 3):
            esp32_ips.append(ip)
            print(f"🎯 ESP32-CAM encontrado en {ip}")
        else:
            print(f"❌ No es ESP32-CAM: {ip}")
    
    if not esp32_ips:
        print("❌ No se encontró ningún ESP32-CAM en la red")
        return None
    
    if len(esp32_ips) == 1:
        ip = esp32_ips[0]
        print(f"🎉 ESP32-CAM encontrado: {ip}")
        return ip
    else:
        print(f"🔍 Se encontraron {len(esp32_ips)} ESP32-CAM:")
        for i, ip in enumerate(esp32_ips, 1):
            print(f"  {i}. {ip}")
        
        # Usar el primero por defecto
        ip = esp32_ips[0]
        print(f"🎯 Usando el primero: {ip}")
        return ip

def infer_esp_base_from_src(src: str) -> str:
    if not (src.startswith("http://") or src.startswith("https://")):
        return ""
    parsed = urlparse(src)
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    port = parsed.port
    if port == 81 or port is None:
        # camera_httpd runs on 80 by default
        return f"{scheme}://{host}"
    else:
        return f"{scheme}://{host}:{port}"


def update_led(base_url: str, on: bool, duty: int) -> None:
    if not base_url:
        return
    try:
        url = f"{base_url}/led?on={'1' if on else '0'}&duty={max(0, min(255, int(duty)))}"
        requests.get(url, timeout=0.5)
    except Exception:
        pass


def update_buzzer(base_url: str, on: bool, duty: int = 128) -> None:
    """Control the buzzer on ESP32-CAM."""
    if not base_url:
        return
    try:
        url = f"{base_url}/buzzer?on={'1' if on else '0'}&duty={max(0, min(255, int(duty)))}"
        requests.get(url, timeout=0.5)
    except Exception:
        pass


def buzzer_beep(base_url: str, duration_ms: int = 200) -> None:
    """Make a short beep sound on the buzzer."""
    if not base_url:
        return
    try:
        url = f"{base_url}/buzzer?beep={max(50, min(5000, duration_ms))}"
        requests.get(url, timeout=0.5)
    except Exception:
        pass


def esp_set_param(base_url: str, var: str, val: int) -> None:
    if not base_url:
        return
    try:
        url = f"{base_url}/control?var={var}&val={val}"
        requests.get(url, timeout=1.0)
    except Exception:
        pass


class MjpegReader:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.resp = None
        self.buffer = bytearray()
        self.stopped = False
        self.lock = threading.Lock()
        self.latest = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.resp = self.session.get(self.url, stream=True, timeout=5)
        self.resp.raise_for_status()
        self.thread.start()
        return self

    def _run(self):
        for chunk in self.resp.iter_content(chunk_size=4096):
            if self.stopped:
                break
            if not chunk:
                continue
            self.buffer.extend(chunk)
            while True:
                start = self.buffer.find(b"\xff\xd8")
                end = self.buffer.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = self.buffer[start : end + 2]
                    del self.buffer[: end + 2]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.latest = frame
                else:
                    break

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return (frame is not None), frame

    def release(self):
        self.stopped = True
        try:
            if self.resp is not None:
                self.resp.close()
        except Exception:
            pass


class SocketMjpegReader:
    def __init__(self, url: str):
        self.url = url
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
        import socket
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        req = f"GET {self.path} HTTP/1.1\r\nHost: {self.host}\r\nConnection: keep-alive\r\n\r\n".encode()
        self.sock.sendall(req)
        # Read headers
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            header += chunk
        self.buffer.extend(header.split(b"\r\n\r\n", 1)[-1])
        self.thread.start()
        return self

    def _run(self):
        while not self.stopped:
            try:
                data = self.sock.recv(4096)
                if not data:
                    time.sleep(0.005)
                    continue
                self.buffer.extend(data)
                while True:
                    start = self.buffer.find(b"\xff\xd8")
                    end = self.buffer.find(b"\xff\xd9")
                    if start != -1 and end != -1 and end > start:
                        jpg = self.buffer[start : end + 2]
                        del self.buffer[: end + 2]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.latest = frame
                    else:
                        break
            except Exception:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return (frame is not None), frame

    def release(self):
        self.stopped = True
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
        import socket
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.thread.start()
        return self

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n and not self.stopped:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _run(self):
        import struct
        while not self.stopped:
            try:
                hdr = self._recv_exact(4)
                if len(hdr) != 4:
                    time.sleep(0.005)
                    continue
                (length,) = struct.unpack('>I', hdr)
                if length <= 0 or length > 5_000_000:
                    continue
                jpg = self._recv_exact(length)
                if len(jpg) != length:
                    continue
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest = frame
            except Exception:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.latest
            self.latest = None
        return (frame is not None), frame

    def release(self):
        self.stopped = True
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

def main() -> None:
    args = parse_args()
    
    # Auto-find ESP32 if requested
    if args.auto_find:
        esp_ip = auto_find_esp32()
        if esp_ip:
            args.src = f"http://{esp_ip}:81/stream"
            args.esp_base = f"http://{esp_ip}"
            print(f"🎯 Usando ESP32-CAM encontrado: {esp_ip}")
        else:
            print("❌ No se pudo encontrar ESP32-CAM automáticamente")
            sys.exit(1)
    
    model, transform, device = load_midas_torch(args.model_type)
    print(f"Using device: {device}")

    # Torch threads (optional)
    if args.torch_threads > 0:
        try:
            torch.set_num_threads(int(args.torch_threads))
        except Exception:
            pass

    # Determine ESP32-CAM control base URL
    esp_base = args.esp_base.strip()
    if not esp_base:
        esp_base = infer_esp_base_from_src(args.src)

    # Optionally tune ESP stream for latency/bandwidth
    if esp_base:
        esp_set_param(esp_base, "quality", args.esp_quality)
        esp_set_param(esp_base, "framesize", args.esp_framesize)
        # Set buzzer duty cycle - commented out since we only use beep
        # update_buzzer(esp_base, False, args.buzzer_duty)

    # Select capture backend
    async_reader = None
    mjpeg_reader = None
    capture = None
    if args.raw_socket:
        # derive host from --esp-base or --src
        base = args.esp_base.strip() or infer_esp_base_from_src(args.src)
        host = urlparse(base if base else args.src).hostname
        if not host:
            print("Error: could not infer host for raw socket from --src/--esp-base")
            sys.exit(1)
        try:
            mjpeg_reader = RawLenSocketReader(host, args.raw_port).start()
        except Exception:
            mjpeg_reader = None
    elif args.socket_mjpeg and (args.src.startswith("http://") or args.src.startswith("https://")):
        try:
            mjpeg_reader = SocketMjpegReader(args.src).start()
        except Exception:
            mjpeg_reader = None
    elif args.mjpeg_reader and (args.src.startswith("http://") or args.src.startswith("https://")):
        try:
            mjpeg_reader = MjpegReader(args.src).start()
        except Exception:
            mjpeg_reader = None
    if mjpeg_reader is None:
        # Make args visible in open_capture for CAP_PROP_BUFFERSIZE try
        globals()['args'] = args
        capture = open_capture(args.src)
        if not capture.isOpened():
            print(f"Error: Unable to open video source '{args.src}'.")
            sys.exit(1)
        if args.async_capture:
            async_reader = LatestFrameReader(capture).start()

    # LED variables commented out since LED is disabled
    # last_led_on = None
    # last_led_ts = 0.0

    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Probe one frame to get size
        grabbed, frame = capture.read()
        if not grabbed:
            print("Error: Could not read from source to initialize writer.")
            sys.exit(1)
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(args.save, fourcc, 20.0, (width, height))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_time = time.time()
    frame_interval = 0.0 if args.fps <= 0 else 1.0 / float(args.fps)

    print("Starting stream. Press 'q' to quit (if GUI enabled).")
    prev_depth = None
    while True:
        if frame_interval > 0.0:
            now = time.time()
            sleep_duration = last_time + frame_interval - now
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            last_time = time.time()

        if mjpeg_reader is not None:
            ok, frame_bgr = mjpeg_reader.read()
            if not ok:
                time.sleep(0.005)
                continue
        elif async_reader is not None:
            ok, frame_bgr = async_reader.read()
            if not ok:
                time.sleep(0.005)
                continue
        else:
            ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            # Some ESP32 streams need extra retries
            time.sleep(0.02)
            continue

        # Optional downscale for processing
        proc_frame = frame_bgr
        if args.proc_width and args.proc_width > 0:
            scale = float(args.proc_width) / float(frame_bgr.shape[1])
            if scale < 0.99:
                proc_frame = cv2.resize(frame_bgr, (int(frame_bgr.shape[1]*scale), int(frame_bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)

        # Optional vertical debanding filter to reduce horizontal stripes
        if args.deband and args.deband_ksize >= 3 and (args.deband_ksize % 2 == 1):
            try:
                proc_frame = cv2.GaussianBlur(proc_frame, (1, int(args.deband_ksize)), 0)
            except Exception:
                pass

        depth_raw = infer_depth_torch(model, transform, device, proc_frame)
        if proc_frame is not frame_bgr:
            depth_raw = cv2.resize(depth_raw, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        depth_norm = normalize_depth(depth_raw, invert=args.invert_depth)

        # Temporal smoothing (EMA) to suppress flicker/stripes without much lag
        if 0.0 < args.temporal_alpha < 1.0:
            if prev_depth is None or prev_depth.shape != depth_norm.shape:
                prev_depth = depth_norm.copy()
            else:
                prev_depth = (1.0 - args.temporal_alpha) * prev_depth + args.temporal_alpha * depth_norm
            depth_used = prev_depth
        else:
            depth_used = depth_norm

        action = decide_navigation_action(depth_used, near_threshold=args.near_threshold)

        # LED policy: DISABLED - only buzzer will be used
        # led_on = action != "FORWARD"
        # now_ts = time.time()
        # if esp_base and (led_on != last_led_on or (now_ts - last_led_ts) > 0.5):
        #     update_led(esp_base, led_on, args.led_duty)
        #     last_led_on = led_on
        #     last_led_ts = now_ts

        # BUZZER policy: continuous sound when obstacle detected (STOP), off for FORWARD
# sonido de alerta en la Mac
        if action == "STOP":
            os.system("afplay -v 1 /System/Library/Sounds/Ping.aiff &")
            sound_playing = True

                # buzzer de la ESP32
        if esp_base:
            if action == "STOP":
                update_buzzer(esp_base, True, args.buzzer_duty)
            else:
                os.system("killall afplay")
                sound_playing = False


        if not args.no_gui or video_writer is not None:
            depth_vis = colorize_depth(depth_used)
            # Draw L/C/R lines and action only on the depth panel
            draw_regions_and_action(depth_vis, action)
            stacked = np.hstack((frame_bgr, depth_vis))
            if not args.no_gui:
                vis = stacked
                scale = 1.0
                if args.display_width and args.display_width > 0:
                    scale = max(0.05, float(args.display_width) / float(vis.shape[1]))
                else:
                    scale = max(0.05, float(args.display_scale))
                if abs(scale - 1.0) > 1e-3:
                    vis = cv2.resize(
                        vis,
                        (int(vis.shape[1] * scale), int(vis.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("Frame | Depth", vis)
            if video_writer is not None:
                video_writer.write(stacked)

        # Example control output (replace with motor/actuator commands as needed)
        # For now, just print the decision to stdout.
        sys.stdout.write(f"\rDecision: {action}   ")
        sys.stdout.flush()

        if not args.no_gui:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if async_reader is not None:
        async_reader.stop()
    if capture is not None:
        capture.release()
    if mjpeg_reader is not None:
        mjpeg_reader.release()
    if video_writer is not None:
        video_writer.release()
    if not args.no_gui:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()


