# 🌐 ORION v3: Web Dashboard (Cámara Local)

Primera versión basada en servidor web (FastAPI + Uvicorn). Permite acceder a la transmisión de video en tiempo real (`/video_feed`) y a la telemetría continua por WebSockets desde cualquier navegador en la red local o equipo anfitrión.

---

## 🚀 Ejecución

Desde la raíz del proyecto:
```bash
python main.py --version 3
```

O directamente desde esta carpeta:
```bash
python main.py --port 8000
```

* **Dashboard Web:** [http://localhost:8000](http://localhost:8000)
* **Documentación OpenAPI:** [http://localhost:8000/docs](http://localhost:8000/docs)
