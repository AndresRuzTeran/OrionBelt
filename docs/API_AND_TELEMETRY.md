# 🌐 Especificación de API y Telemetría WebSocket

El backend de ORION (basado en FastAPI y Starlette) expone endpoints REST para control y monitoreo, junto con una conexión WebSocket de alta frecuencia para telemetría visual.

---

## 📡 Endpoints REST

### 1. `GET /`
* **Descripción:** Sirve la interfaz web principal (`templates/index.html`).
* **Respuesta:** HTML/Text (`200 OK`).

### 2. `GET /video_feed`
* **Descripción:** Flujo continuo de video en formato multipart/x-mixed-replace (MJPEG), consumible directamente por etiquetas HTML `<img>`.
* **Parámetros Query:**
  - `mode` *(string, opcional)*:
    - `combined` *(por defecto)*: Muestra el fotograma original y el mapa de profundidad lado a lado.
    - `depth`: Muestra únicamente el mapa de calor de profundidad (inferencia MiDaS).
    - `raw`: Muestra únicamente la imagen RGB de la cámara.
* **Respuesta:** Stream de imágenes JPEG (`Content-Type: multipart/x-mixed-replace; boundary=frame`).

### 3. `GET /api/status`
* **Descripción:** Endpoint de comprobación de salud (*health check*) y estado de conectividad de los dispositivos de captura.
* **Respuesta (JSON):**
```json
{
  "status": "online",
  "camera_mode": "xiao",
  "xiao_ip": "172.16.121.118",
  "xiao_alive": true,
  "fallback_active": false,
  "device": "mps",
  "model_type": "MiDaS_small",
  "uptime_seconds": 142.5
}
```

### 4. `POST /api/camera_switch`
* **Descripción:** Fuerza manualmente la conmutación entre la cámara inalámbrica XIAO y la cámara de respaldo local.
* **Cuerpo (JSON):**
```json
{
  "target": "local"  // o "xiao"
}
```

---

## ⚡ Telemetría en Tiempo Real por WebSocket

* **Ruta:** `ws://<host>:<port>/ws/telemetry`
* **Frecuencia:** ~25 Hz (actualizaciones cada 40 ms).

### Esquema del Payload JSON
```json
{
  "fps_camera": 29.8,
  "fps_inference": 24.2,
  "latency_ms": 32.4,
  "active_source": "XIAO ESP32S3 (172.16.121.118)",
  "hot_swap_status": "LOCKED",
  "obstacle_detected": true,
  "proximity_score": 0.82,
  "near_threshold": 0.65,
  "zones": {
    "left": 0.35,
    "center": 0.82,
    "right": 0.28
  },
  "alarm_active": true,
  "system_memory_mb": 420.5
}
```

### Campos Principales
* `fps_camera`: Cuadros por segundo capturados de la fuente de video.
* `fps_inference`: Cuadros por segundo procesados por el modelo de profundidad.
* `latency_ms`: Tiempo transcurrido desde la recepción del fotograma hasta la finalización del postprocesamiento.
* `zones`: Estimación de proximidad normalizada (0.0 a 1.0) dividida en 3 regiones horizontales del campo visual.
* `alarm_active`: Bandera booleana que indica si el sistema está emitiendo una alerta sonora en el equipo anfitrión.
