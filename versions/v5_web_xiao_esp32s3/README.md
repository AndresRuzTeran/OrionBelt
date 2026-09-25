# 🚀 ORION v5: Web XIAO ESP32S3 Sense (Versión Insignia Actual)

Versión principal de producción de ORION. Integra la placa **Seeed Studio XIAO ESP32S3 Sense** con un flujo de video ultra-fluido a 30 FPS, arquitectura de tolerancia a fallos con conmutación en caliente (*Hot-Swap*) a la cámara web integrada de la computadora y soporte para arranque en segundo plano en macOS.

---

## 🌟 Características Clave
* **Streaming Estable a 30 FPS:** Firmware libre del bloqueo TCP `SO_SNDTIMEO` y con reloj de sensor a 16 MHz para máxima estabilidad térmica.
* **Hot-Swap & Rollback Bidireccional:** Si la cámara XIAO se apaga, la cámara de la Mac toma el relevo en menos de 1 segundo sin reiniciar el servidor; al encenderse nuevamente el XIAO, el sistema regresa a la transmisión inalámbrica.
* **Integración con macOS Service:** Escribe el puerto dinámico en `.orion_port` para sincronización con scripts automáticos `.sh`.
* **Cero Caídas:** Limpieza de procesos `afplay` y cierre ordenado de buffers OpenCV AVFoundation / DirectShow.

---

## 📁 Estructura de la Versión
* `main.py`: Servidor web FastAPI principal de la versión v5.
* `start.py`: Lanzador rápido con detección de arquitectura y flags recomendados.
* `firmware/XIAO_Camera_Orion/`: Sketch para Arduino IDE optimizado para el XIAO ESP32S3 Sense.
* `firmware/XIAO_Camera_Orion.zip`: Paquete comprimido del sketch listo para importar.

---

## 🚀 Ejecución

Desde la raíz del proyecto (es la versión por defecto):
```bash
python main.py
```

Con parámetros específicos:
```bash
python main.py --xiao-ip 172.16.121.118 --device mps --port 8000
```

O usando el lanzador rápido:
```bash
python versions/v5_web_xiao_esp32s3/start.py
```
