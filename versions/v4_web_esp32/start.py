#!/usr/bin/env python3
"""
Script de inicio rápido que encuentra automáticamente el ESP32-CAM y ejecuta el sistema.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Iniciando sistema de detección de obstáculos...")
    print("=" * 50)
    
    # Ejecutar el script principal con detección automática
    script_path = str(Path(__file__).resolve().parent / "main.py")
    cmd = [
        sys.executable, script_path,
        "--auto-find",
        "--no-local-sound"
    ]
    
    print("🔍 Buscando ESP32-CAM automáticamente...")
    print("📝 Comando:", " ".join(cmd))
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Sistema detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar el sistema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
