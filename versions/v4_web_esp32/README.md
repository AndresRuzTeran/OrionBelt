# 📷 ORION v4: Web ESP32-CAM (AI-Thinker)

Versión web conectada a la placa inalámbrica clásica **AI-Thinker ESP32-CAM**. Incluye escaneo automático de subred para localizar la dirección IP de la placa y firmware en Arduino IDE.

---

## 📁 Estructura de la Versión
* `main.py`: Servidor web FastAPI con captura remota MJPEG desde el ESP32.
* `start.py`: Lanzador rápido con bandera `--auto-find`.
* `firmware/CameraWebServer/`: Sketch para Arduino IDE listo para flashear en el módulo ESP32-CAM.

---

## 🚀 Ejecución

Desde la raíz del proyecto:
```bash
python main.py --version 4
```

O usando el lanzador rápido:
```bash
python versions/v4_web_esp32/start.py
```
