#!/usr/bin/env python3
"""
=============================================================================
ORION - Optical Recognition & Intelligent Obstacle Navigation
Lanzador Principal Multi-Versión
=============================================================================
Por defecto ejecuta la versión más reciente (v5: Web XIAO ESP32S3 Sense).
Permite seleccionar y ejecutar cualquier versión anterior.

Uso:
    python main.py                          # Inicia la versión actual (v5)
    python main.py --version 3              # Inicia la versión v3 (Web Cámara Local)
    python main.py -V 2                     # Inicia la versión v2 (Desktop HUD)
    python main.py --list                   # Lista todas las versiones disponibles
    python main.py --device mps             # Pasa argumentos a la versión activa
=============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

# Configurar salida UTF-8 en consolas Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent

VERSIONS = {
    "1": {
        "id": "v1_desktop_initial",
        "name": "v1: Desktop Inicial (OpenCV + MiDaS)",
        "path": PROJECT_ROOT / "versions" / "v1_desktop_initial" / "main.py",
        "desc": "Primera implementación de estimación de profundidad con ventana OpenCV directa.",
    },
    "2": {
        "id": "v2_desktop_optimized",
        "name": "v2: Desktop Optimizado (HUD Telemetría + Audio)",
        "path": PROJECT_ROOT / "versions" / "v2_desktop_optimized" / "main.py",
        "desc": "Versión de escritorio multi-hilo con HUD de telemetría y alarmas sonoras.",
    },
    "3": {
        "id": "v3_web_local",
        "name": "v3: Web Dashboard (Cámara Local)",
        "path": PROJECT_ROOT / "versions" / "v3_web_local" / "main.py",
        "desc": "Servidor web FastAPI + WebSockets con transmisión desde la cámara integrada/USB.",
    },
    "4": {
        "id": "v4_web_esp32",
        "name": "v4: Web ESP32-CAM (AI-Thinker)",
        "path": PROJECT_ROOT / "versions" / "v4_web_esp32" / "main.py",
        "desc": "Servidor web conectado a módulo inalámbrico ESP32-CAM estándar.",
    },
    "5": {
        "id": "v5_web_xiao_esp32s3",
        "name": "v5: Web XIAO ESP32S3 Sense (Hot-Swap + macOS Service) [ACTUAL]",
        "path": PROJECT_ROOT / "versions" / "v5_web_xiao_esp32s3" / "main.py",
        "desc": "Versión insignia: 30 FPS MJPEG, auto-discovery de IP, hot-swap a cámara Mac y servicio en segundo plano.",
    },
}

DEFAULT_VERSION = "5"


def print_banner():
    print("=" * 68)
    print("      🌌 ORION - Optical Recognition & Intelligent Obstacle Navigation")
    print("=" * 68)


def list_versions():
    print_banner()
    print("Versiones disponibles en el proyecto:\n")
    for key, info in sorted(VERSIONS.items()):
        is_default = " ⭐ (Por defecto)" if key == DEFAULT_VERSION else ""
        print(f"  [{key}] {info['name']}{is_default}")
        print(f"      📁 Ruta: {info['path'].relative_to(PROJECT_ROOT)}")
        print(f"      ℹ️  {info['desc']}\n")
    print("Para ejecutar una versión específica:")
    print("  python main.py --version <número> [argumentos opcionales]\n")


def main():
    args = sys.argv[1:]

    # Ayuda o listado de versiones
    if "--list" in args or "-l" in args:
        list_versions()
        sys.exit(0)

    # Detectar selector de versión
    chosen_ver = DEFAULT_VERSION
    remaining_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--version", "-V", "-v"):
            if i + 1 < len(args):
                chosen_ver = args[i + 1].strip()
                # Aceptar tanto '5' como 'v5'
                if chosen_ver.lower().startswith("v"):
                    chosen_ver = chosen_ver[1:]
                i += 2
                continue
            else:
                print("❌ Falta el número de versión después de --version / -V", file=sys.stderr)
                sys.exit(1)
        elif arg.startswith("--version="):
            chosen_ver = arg.split("=", 1)[1].strip()
            if chosen_ver.lower().startswith("v"):
                chosen_ver = chosen_ver[1:]
            i += 1
            continue
        else:
            remaining_args.append(arg)
            i += 1

    if chosen_ver not in VERSIONS:
        print(f"❌ Versión desconocida: '{chosen_ver}'.", file=sys.stderr)
        print(f"   Versiones válidas: {', '.join(VERSIONS.keys())}", file=sys.stderr)
        print("   Ejecuta 'python main.py --list' para ver todas las opciones.", file=sys.stderr)
        sys.exit(1)

    target_info = VERSIONS[chosen_ver]
    target_script = target_info["path"]

    if not target_script.exists():
        print(f"❌ No se encontró el archivo ejecutable en: {target_script}", file=sys.stderr)
        sys.exit(1)

    print_banner()
    print(f"🚀 Iniciando {target_info['name']}")
    print(f"📁 Script: {target_script.relative_to(PROJECT_ROOT)}")
    if remaining_args:
        print(f"⚙️  Argumentos: {' '.join(remaining_args)}")
    print("=" * 68 + "\n")

    cmd = [sys.executable, str(target_script)] + remaining_args
    env = os.environ.copy()

    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  ORION detenido por el usuario.")
        sys.exit(0)


if __name__ == "__main__":
    main()
