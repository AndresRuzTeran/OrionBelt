# ⚡ ORION v2: Desktop Optimizado (HUD Telemetría + Audio)

Versión de escritorio optimizada con arquitectura multi-hilo para desacoplar la captura de video y la inferencia de la red neuronal. Incluye un HUD interactivo superpuesto en pantalla y sistema de alarmas sonoras asíncronas para macOS y Windows.

---

## 🚀 Ejecución

Desde la raíz del proyecto:
```bash
python main.py --version 2
```

O directamente desde esta carpeta:
```bash
python main.py
```

### Argumentos Opcionales
* `--device {auto,cpu,cuda,mps}`: Selección del backend de cómputo.
* `--near-threshold 0.65`: Umbral de proximidad para activar la alarma.
* `--no-local-sound`: Desactivar alertas sonoras.
