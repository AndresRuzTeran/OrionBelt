# 📚 Documentación Oficial de ORION

Bienvenido al centro de documentación de **ORION (Optical Recognition & Intelligent Obstacle Navigation)**. Este repositorio contiene todas las especificaciones de arquitectura, guías de hardware, configuración de servicios del sistema operativo y referencia de versiones.

---

## 🗺️ Índice de Guías

| Documento | Descripción |
| :--- | :--- |
| **[Arquitectura del Sistema](ARCHITECTURE.md)** | Análisis profundo del pipeline de visión por computadora (MiDaS), arquitectura multihilo, máquina de estados de Hot-Swap y sistema de audio. |
| **[Historial y Detalle de Versiones](VERSIONS.md)** | Comparativa detallada de las 5 versiones (v1 a v5), árbol de evolución y dependencias asociadas. |
| **[Guía Hardware: XIAO ESP32S3 Sense](HARDWARE_XIAO_ESP32S3.md)** | Configuración de la placa Seeed Studio XIAO ESP32S3 Sense, instalación de cámara OV2640, flasheo en Arduino IDE, configuración de PSRAM y optimizaciones térmicas (16 MHz XCLK). |
| **[Guía Hardware: ESP32-CAM (AI-Thinker)](HARDWARE_ESP32_CAM.md)** | Cableado con programador FTDI USB a UART, puente de GPIO0 a GND para flasheo y configuración en Arduino IDE. |
| **[Servicio Automático en macOS](MACOS_SERVICE.md)** | Configuración para arrancar ORION automáticamente al encender el Mac, protocolo de sincronización `.orion_port` y permisos de cámara AVFoundation. |
| **[API REST y Telemetría WebSocket](API_AND_TELEMETRY.md)** | Documentación completa de los endpoints HTTP, streaming MJPEG y el protocolo de telemetría a 25 Hz en tiempo real. |

---

## 🚀 Resumen Rápido de Ejecución

Para iniciar la versión más avanzada y estable del proyecto (**v5: Web XIAO ESP32S3 Sense con Hot-Swap a Mac**):

```bash
# Iniciar con parámetros automáticos
python main.py

# Iniciar forzando aceleración por Metal en Apple Silicon
python main.py --device mps

# Iniciar especificando una IP estática de la cámara
python main.py --xiao-ip 172.16.121.4
```

Para listar o ejecutar versiones anteriores:
```bash
# Listar todas las versiones
python main.py --list

# Ejecutar versión anterior (ej. v3 Web Cámara Local)
python main.py --version 3
```
