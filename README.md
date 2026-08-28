# 🌌 ORION | Sistema de Navegación y Detección de Obstáculos

<div align="center">
  <img src="images/portada.jpeg" alt="ORION - Sistema de Navegación y Detección de Obstáculos" width="100%">
</div>

> **ORION** es un sistema de visión artificial y navegación asistida en tiempo real basado en estimación de profundidad monocular (**MiDaS**). Procesa video desde cámaras web o módulos **ESP32-CAM** para detectar obstáculos en el corredor de avance, calcular su nivel de proximidad y emitir alertas visuales y sonoras progresivas.

---

## 📋 Tabla de Contenidos

- [Visión General](#-visión-general)
- [Comparativa: `main.py` (Original) vs `main-prueba.py` (Optimizado)](#-comparativa-mainpy-original-vs-main-pruebapy-optimizado)
- [Nueva Funcionalidad: `main_web.py` (Dashboard Web)](#-nueva-funcionalidad-main_webpy-dashboard-web)
- [Métrica de Medición y Distancia de Activación](#-métrica-de-medición-y-distancia-de-activación)
- [Guía de Usuario e Instalación](#-guía-de-usuario-e-instalación)
  - [1. Instalación](#1-instalación)
  - [2. Modo Escritorio (Ventana OpenCV)](#2-modo-escritorio-ventana-opencv)
  - [3. Modo Servicio Web (Dashboard HTML5)](#3-modo-servicio-web-dashboard-html5)
- [Parámetros de Línea de Comandos (CLI)](#-parámetros-de-línea-de-comandos-cli)
- [Herramientas Auxiliares](#-herramientas-auxiliares)

---

## 🔭 Visión General

ORION analiza el flujo de video en tiempo real, ejecuta la red neuronal **MiDaS** para generar un mapa de disparidad relativa y evalúa el tercio central de la escena:
- **`FORWARD`**: Corredor de avance despejado.
- **`STOP`**: Obstáculo detectado por encima del umbral de cercanía (por defecto $\ge 60\%$).
- **Alertas Progresivas**: Emisión de beeps sonoros que aumentan en tono, volumen y cadencia a medida que el obstáculo se acerca.

---

## ⚡ Comparativa: `main.py` (Original) vs `main-prueba.py` (Optimizado)

El archivo [`main.py`](main.py) fue la versión inicial del proyecto. El archivo [`main-prueba.py`](main-prueba.py) introduce optimizaciones críticas de rendimiento, audio y visualización:

### 1. Aceleración de Hardware Multiplataforma
- **Original (`main.py`):** Solo soportaba CUDA (Nvidia). En Mac corría forzosamente en CPU a ~3-5 FPS.
- **Optimizado (`main-prueba.py`):**
  - **Apple Silicon (MPS):** Usa la GPU integrada de Mac (M1/M2/M3/M4) alcanzando **20-30+ FPS**, con fallback automático ante operaciones no soportadas (`PYTORCH_ENABLE_MPS_FALLBACK`).
  - **Windows / Linux:** Soporte completo para **CUDA (Nvidia)** y CPU.
  - **Cámara:** Backend nativo `AVFoundation` en macOS y `DirectShow/MSMF` en Windows.

### 2. Pipeline de Inferencia Rápido
- **`@torch.inference_mode()`**: Elimina la sobrecarga de seguimiento de tensores.
- **Interpolación Bilineal**: Reemplaza `bicubic` por `bilinear`, mucho más rápida en GPU/MPS.
- **Resolución Balanceada (`--proc-width 320`)**: Triplica la velocidad de inferencia manteniendo precisión en la detección de obstáculos.
- **Suavizado Temporal In-Place**: Reduce consumo de memoria y micro-tirones.

### 3. Sistema de Alarma Multinivel Multihilo
- **Hilo Independiente (`AlarmManager`)**: La alarma acústica no se traba ni depende de los FPS del video.
- **4 Niveles de Urgencia (`LocalAlarmSound`)**: Genera tonos senoidales puros (700 Hz a 1500 Hz) que escalan según la cercanía.
- **Soporte Nativo en Mac**: Utiliza `/usr/bin/afplay` sin necesidad de instalar librerías externas de audio.

### 4. Captura Asíncrona y UI Compacta
- **`LatestFrameReader` activo por defecto**: Descarta cuadros viejos en buffer para eliminar el retraso acumulado del video.
- **HUD Compacto**: Letras de acción con tamaño optimizado y legible, indicador numérico de proximidad en porcentaje (`Prox: XX.X%`) y barra gráfica de seguridad.

| Característica | `main.py` (Original) | `main-prueba.py` (Optimizado) |
| :--- | :--- | :--- |
| **Aceleración Hardware** | Solo CUDA o CPU lenta | **Apple Silicon (MPS)**, CUDA y CPU |
| **FPS Típicos (Mac M1/M2)** | ~3 - 5 FPS | **20 - 30+ FPS** |
| **Modo PyTorch** | `torch.no_grad()` | `@torch.inference_mode()` |
| **Alarma Sonora en PC/Mac** | ❌ No | ✅ **Sí, 4 niveles progresivos (700Hz - 1500Hz)** |
| **Arquitectura de Audio** | Síncrona (se corta si hay lag) | **Multihilo (`AlarmManager`), sonido continuo** |
| **Latencia de Video** | Acumulativa en buffer | **Mínima (descarte de frames obsoletos)** |
| **Visualización en Pantalla** | Texto grande sin telemetría | **Texto compacto + Medidor de Proximidad %** |

---

## 🌐 Nueva Funcionalidad: `main_web.py` (Dashboard Web)

[`main_web.py`](main_web.py) sustituye la ventana de escritorio de OpenCV por un **Servicio Web interactivo en tiempo real** desarrollado con **FastAPI + Uvicorn + WebSockets** y un panel en **HTML5/CSS3/JS**:

### Ventajas y Capacidades:
1. **Separación de Cómputo y Renderizado:** Python solo procesa video y telemetría ligera; el navegador web renderiza la interfaz fluida a 60 FPS mediante la GPU del cliente.
2. **Selector de Vistas en Vivo:** Alterna al instante entre **Vista Combinada**, **Solo Cámara** o **Solo Mapa de Profundidad**.
3. **Telemetría en Tiempo Real:**
   - Badge gigante de estado (`FORWARD` en verde / `STOP` en rojo brillante con animación de pulso).
   - Manómetro y barra de proximidad en porcentaje con marca del umbral.
   - Gráfica temporal (Canvas 2D) con los últimos 15 segundos de historial de proximidad.
   - Indicadores de FPS, backend de cómputo (`MPS`, `CUDA`, `CPU`) e intensidad.
4. **Controles Interactivos en Vivo (sin reiniciar el script):**
   - Slider para ajustar la distancia de activación del umbral `STOP` en tiempo real.
   - Interruptor para silenciar o activar la alarma sonora local de la computadora.
5. **Acceso Multi-Dispositivo:** Abre el dashboard desde tu Mac, PC, teléfono móvil (iPhone/Android) o tablet en la misma red WiFi.
6. **Apagado Instantáneo:** Cierre limpio y seguro con `Ctrl + C` desde la terminal.

---

## 📐 Métrica de Medición y Distancia de Activación

- **¿Qué mide MiDaS?**
  Calcula **Disparidad Inversa Relativa** ($d \propto \frac{1}{\text{distancia}}$). En [`normalize_depth()`](main-prueba.py) se normaliza de $0.0$ ($0\%$, punto más lejano) a $1.0$ ($100\%$, punto más cercano al lente).
- **¿A qué distancia se activa `STOP`?**
  Con el umbral estándar del $60\%$ (`--near-threshold 0.6`):
  - **$> 2.5\text{ m}$ ($0\% - 30\%$):** `FORWARD` (Silencio).
  - **$1.5\text{ m} - 2.5\text{ m}$ ($30\% - 59\%$):** `FORWARD` (Advertencia visual en telemetría).
  - **$\approx 0.8\text{ m} - 1.5\text{ m}$ ($60\% - 84\%$):** **`STOP`** (Alarma activa de 700 Hz a 1200 Hz).
  - **$< 0.5\text{ m}$ ($85\% - 100\%$):** **`STOP` Crítico** (Alarma máxima a 1500 Hz rápida).

---

## 🚀 Guía de Usuario e Instalación

### 1. Instalación
```bash
# Clonar repositorio
git clone <URL_DEL_REPOSITORIO>
cd ORION

# Crear y activar entorno virtual
python -m venv .venv
# En Windows:
.\.venv\Scripts\activate
# En macOS/Linux:
source .venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Modo Escritorio (Ventana OpenCV)

Ejecuta [`main-prueba.py`](main-prueba.py) para abrir la ventana clásica de OpenCV:

* **Con cámara web local:**
  ```bash
  python main-prueba.py --src 0
  ```
* **Con ESP32-CAM (Búsqueda automática en WiFi):**
  ```bash
  python main-prueba.py --auto-find
  ```
* **Con ESP32-CAM (Por IP directa):**
  ```bash
  python main-prueba.py --src http://192.168.1.50:81/stream --esp-base http://192.168.1.50
  ```

---

### 3. Modo Servicio Web (Dashboard HTML5)

Ejecuta [`main_web.py`](main_web.py) para iniciar el servidor web:

* **Iniciar con cámara web:**
  ```bash
  python main_web.py --src 0
  ```
* **Iniciar con ESP32-CAM:**
  ```bash
  python main_web.py --auto-find
  ```
* **Forzar aceleración Apple Silicon (MPS) o puerto específico:**
  ```bash
  python main_web.py --src 0 --device mps --port 8080
  ```

**Acceso al Dashboard:**
- Desde tu ordenador: Abre en el navegador **`http://localhost:8080`** (o el puerto indicado en terminal).
- Desde tu celular o tablet: Abre **`http://<IP_DE_TU_PC>:8080`** conectado a la misma red WiFi.

---

## ⚙️ Parámetros de Línea de Comandos (CLI)

| Parámetro | Opciones / Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--src` | `0`, `1` o URL HTTP | `0` | Índice de cámara local o URL del stream del ESP32. |
| `--auto-find` | Flag | `False` | Escanea la red local y se conecta automáticamente al ESP32-CAM. |
| `--near-threshold` | Float (`0.0` a `1.0`) | `0.6` | Umbral de proximidad para activar el estado `STOP`. |
| `--device` | `auto`, `mps`, `cuda`, `cpu` | `auto` | Backend de cómputo para la red neuronal. |
| `--port` | Entero | `8000` / auto | Puerto HTTP para `main_web.py` (autodetecta puertos libres). |
| `--no-local-sound` | Flag | `False` | Desactiva las alertas sonoras en la computadora. |
| `--proc-width` | Entero | `320` | Ancho de imagen procesado por MiDaS (menor = más rápido). |
| `--save` | String | `""` | Guarda el video procesado en archivo (ej. `output.mp4`). |

---

## 🛠️ Herramientas Auxiliares

- **[`find_esp32.py`](find_esp32.py):** Escanea la red local para identificar la dirección IP del ESP32-CAM.
- **[`test_esp32.py`](test_esp32.py):** Comprueba la conectividad TCP y la respuesta del stream de video.
- **[`diagnose_network.py`](diagnose_network.py):** Diagnóstico de velocidad y estabilidad del enlace de red.
- **[`CameraWebServer/`](CameraWebServer/):** Código fuente / sketch Arduino para la placa ESP32-CAM.

---

<div align="center">
  <sub>Proyecto <b>ORION</b> • Visión Artificial & Navegación Autónoma / Asistida.</sub>
</div>
