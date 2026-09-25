#!/usr/bin/env python3
"""
Script de inicio rápido para ORION con XIAO ESP32S3 Sense y Hot-Swap/Rollback a Cámara Mac.
"""

import os
import subprocess
import sys

# Asegurar compatibilidad UTF-8 en terminales Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    print("🚀 Iniciando ORION (XIAO ESP32S3 + Rollback Autónomo a Cámara Local)...")
    print("=" * 65)
    print("⚠️  AVISO IMPORTANTE:")
    print("   NO abras la URL directa del stream en otra pestaña del navegador mientras ORION corre,")
    print("   ya que saturarías el socket del microcontrolador y forzarías el Rollback.")
    print("   Abre únicamente el Dashboard web de ORION (http://localhost:8000).")
    print("=" * 65)
    
    device = "mps" if sys.platform == "darwin" else "auto"
    cmd = [
        sys.executable, "main_web_xiao.py",
        "--device", device,
    ] + sys.argv[1:]
    
    print("📝 Comando:", " ".join(cmd))
    print("=" * 65 + "\n")
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n⏹️  Sistema detenido por el usuario.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar el sistema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

