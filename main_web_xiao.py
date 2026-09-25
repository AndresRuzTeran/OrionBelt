#!/usr/bin/env python3
"""
=============================================================================
ORION - Wrapper de Compatibilidad para Servicio macOS
=============================================================================
Este archivo mantiene compatibilidad con scripts de servicio y automatizaciones
existentes (como el script .sh de arranque automático en macOS) que invocan:
    python main_web_xiao.py

Redirige automáticamente la ejecución a:
    versions/v5_web_xiao_esp32s3/main.py
=============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "versions" / "v5_web_xiao_esp32s3" / "main.py"
    if not target.exists():
        print(f"❌ Error: No se encontró el script de la versión 5 en {target}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(target)] + sys.argv[1:]
    env = os.environ.copy()

    try:
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)
