# 📜 Historial y Evolución de Versiones de ORION

El proyecto **ORION** ha evolucionado a través de 5 versiones principales, transitando desde un prototipo monolítico de escritorio hasta una plataforma web distribuida con conmutación en caliente de hardware IoT.

---

## 📊 Matriz Comparativa de Versiones

| Característica | v1: Desktop Inicial | v2: Desktop Optimizado | v3: Web Local | v4: Web ESP32 | v5: Web XIAO ESP32S3 (Actual) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Entorno de Visualización** | Ventana OpenCV | Ventana OpenCV + HUD | Navegador Web | Navegador Web | Navegador Web Reactivo |
| **Fuente de Video** | Cámara Local / IP | Cámara Local / IP | Cámara Local / USB | ESP32-CAM (AI-Thinker) | **XIAO ESP32S3 + Hot-Swap Mac** |
| **Tasa de Cuadros (FPS)** | 10 - 15 FPS | 20 - 25 FPS | 20 - 25 FPS | 8 - 15 FPS | **30 FPS (Constantes)** |
| **Conmutación en Caliente** | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Sí (Bidireccional < 1s)** |
| **Descubrimiento de IP** | ❌ Manual | ❌ Manual | N/A | ⚠️ Búsqueda lenta | **✅ Instantáneo + Multicast** |
| **Arranque en Segundo Plano**| ❌ No | ❌ No | ❌ No | ❌ No | **✅ Sí (.orion_port + .sh daemon)**|
| **Soporte Apple Silicon (MPS)**| ⚠️ Parcial | ⚠️ Parcial | ✅ Sí | ✅ Sí | **✅ Optimizado + Fallback** |
| **Prevención Crash macOS** | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Thread-safe join AVFoundation**|

---

## 🗂️ Detalle por Versión

### 🔹 Versión 1: Desktop Inicial (`versions/v1_desktop_initial/`)
* **Archivo principal:** [`versions/v1_desktop_initial/main.py`](../versions/v1_desktop_initial/main.py)
* **Objetivo:** Primer prototipo de validación de concepto de estimación de profundidad monocular usando MiDaS en Python.
* **Características:**
  - Pipeline síncrono: la captura, inferencia y renderizado se ejecutaban en el mismo hilo, reduciendo la fluidez a la velocidad de la red neuronal.
  - Renderizado mediante la función tradicional `cv2.imshow()`.
* **Cómo ejecutar:**
  ```bash
  python main.py --version 1
  ```

---

### 🔹 Versión 2: Desktop Optimizado (`versions/v2_desktop_optimized/`)
* **Archivo principal:** [`versions/v2_desktop_optimized/main.py`](../versions/v2_desktop_optimized/main.py)
* **Objetivo:** Maximizar el rendimiento en escritorio, desacoplando la inferencia de la tasa de refresco y agregando métricas visuales.
* **Características:**
  - Introducción del lector de último fotograma para eliminar la latencia acumulada.
  - HUD interactivo superpuesto en pantalla con histogramas, porcentaje de proximidad y FPS.
  - Implementación de audio de advertencia mediante `simpleaudio` / `winsound` / `afplay`.
* **Cómo ejecutar:**
  ```bash
  python main.py --version 2
  ```

---

### 🔹 Versión 3: Web Dashboard con Cámara Local (`versions/v3_web_local/`)
* **Archivo principal:** [`versions/v3_web_local/main.py`](../versions/v3_web_local/main.py)
* **Objetivo:** Migrar la interfaz gráfica desde ventanas nativas de escritorio hacia una arquitectura cliente-servidor accesible desde cualquier navegador.
* **Características:**
  - Servidor asíncrono con **FastAPI** y **Uvicorn**.
  - Transmisión de video comprimido mediante multipart MJPEG (`/video_feed`).
  - Telemetría en tiempo real a 25 Hz mediante **WebSockets** (`/ws/telemetry`).
  - Dashboard responsivo basado en HTML5, CSS moderno y JavaScript modular.
* **Cómo ejecutar:**
  ```bash
  python main.py --version 3
  ```

---

### 🔹 Versión 4: Web con ESP32-CAM (`versions/v4_web_esp32/`)
* **Archivo principal:** [`versions/v4_web_esp32/main.py`](../versions/v4_web_esp32/main.py)
* **Lanzador rápido:** [`versions/v4_web_esp32/start.py`](../versions/v4_web_esp32/start.py)
* **Firmware:** [`versions/v4_web_esp32/firmware/CameraWebServer/`](../versions/v4_web_esp32/firmware/CameraWebServer/)
* **Objetivo:** Integrar la primera cámara inalámbrica basada en el SoC ESP32 estándar (placa AI-Thinker).
* **Características:**
  - Firmware Arduino basado en el ejemplo clásico de Espressif con resolución SVGA/VGA.
  - Escaneo de subred para localizar el ESP32 en la red WiFi.
  - Limitaciones: Tasa de cuadros limitada (8 a 15 FPS) y sin mecanismo de respaldo si la cámara perdía conexión WiFi.
* **Cómo ejecutar:**
  ```bash
  python main.py --version 4
  # o bien:
  python versions/v4_web_esp32/start.py
  ```

---

### 🔹 Versión 5: Web XIAO ESP32S3 Sense (Versión Insignia Actual) (`versions/v5_web_xiao_esp32s3/`)
* **Archivo principal:** [`versions/v5_web_xiao_esp32s3/main.py`](../versions/v5_web_xiao_esp32s3/main.py)
* **Lanzador rápido:** [`versions/v5_web_xiao_esp32s3/start.py`](../versions/v5_web_xiao_esp32s3/start.py)
* **Firmware:** [`versions/v5_web_xiao_esp32s3/firmware/XIAO_Camera_Orion/`](../versions/v5_web_xiao_esp32s3/firmware/XIAO_Camera_Orion/)
* **Objetivo:** Solución definitiva de grado de producción: hardware de última generación, máxima fluidez (30 FPS), tolerancia total a fallos de red y arranque autónomo como servicio en macOS.
* **Características Destacadas:**
  - **Firmware dedicado para XIAO ESP32S3:** Servidor HTTP en Núcleo 1, puerto dedicado 81, triple buffer DMA en PSRAM y eliminación del bloqueo TCP `SO_SNDTIMEO`.
  - **Hot-Swap y Rollback Autónomo:** Si el XIAO se apaga, la cámara de la Mac entra en acción inmediatamente; cuando el XIAO vuelve a encenderse, conmuta de regreso sin tocar nada.
  - **Soporte Servicio macOS:** Generación del archivo `.orion_port` en arranque para sincronización con scripts automáticos `.sh`.
  - **Seguridad contra caídas en macOS:** Cierre ordenado de subprocesos `afplay` y de buffers de captura para prevenir `SIGSEGV: 11`.
* **Cómo ejecutar:**
  ```bash
  python main.py
  # o con opciones:
  python main.py --xiao-ip 172.16.121.4 --device mps
  ```
