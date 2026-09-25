# 🛠️ Guía de Hardware y Firmware: Seeed Studio XIAO ESP32S3 Sense

Esta guía detalla el ensamblaje físico, configuración del entorno de desarrollo y flasheo del firmware optimizado para la placa **Seeed Studio XIAO ESP32S3 Sense** utilizada en ORION v5.

---

## 📋 Especificaciones de la Placa

| Parámetro | Valor |
| :--- | :--- |
| **Microcontrolador** | ESP32-S3 (Xtensa Dual-Core 32-bit LX7 @ hasta 240 MHz) |
| **Memoria PSRAM** | 8 MB Octal PSRAM (OPI) de alta velocidad |
| **Memoria Flash** | 8 MB SPI Flash |
| **Módulo de Cámara** | Sensor OV2640 de 2 Megapíxeles (FPC desmontable) |
| **Micrófono Integrado** | PDM digital MSM261D3526H1CPM |
| **Almacenamiento** | Ranura MicroSD en placa de expansión Sense |
| **Conectividad Inalámbrica**| Wi-Fi 802.11 b/g/n (2.4 GHz) + Bluetooth 5.0 (LE) |
| **Conector de Antena** | U.FL / IPEX para antena externa |

---

## 🔌 Conexión Física y Ensamblaje

1. **Placa de Expansión Sense:**
   - La placa XIAO ESP32S3 se monta encima de la placa inferior de expansión (Sense).
   - Asegúrate de que los pines de cabecera hagan contacto firme y queden orientados en la misma dirección (el puerto USB-C queda visible en el extremo).
2. **Instalación de la Cámara OV2640:**
   - Levanta suavemente la pestaña de bloqueo negra del conector FPC en la parte superior de la placa Sense.
   - Inserta la cinta flexible de la cámara con los **contactos dorados mirando hacia la placa**.
   - Presiona la pestaña negra hacia abajo hasta que quede asegurada.
3. **Antena Externa:**
   - Conecta la antena de varilla al conector micro-coaxial U.FL (IPEX). **Nunca enciendas la transmisión Wi-Fi sin la antena conectada**, ya que puede reducir drásticamente el alcance y sobrecalentar la etapa RF.

---

## ⚙️ Configuración en Arduino IDE

### 1. Instalación del Core ESP32
En Arduino IDE, ve a **Archivo $\rightarrow$ Preferencias** y añade la siguiente URL al gestor de URLs de placas adicionales:
```text
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
Luego ve a **Herramientas $\rightarrow$ Placa $\rightarrow$ Gestor de tarjetas**, busca `esp32` e instala la versión **2.0.14** o superior (recomendado 2.0.17).

### 2. Opciones de Placa (Board Settings)
Selecciona el puerto COM correspondiente y ajusta los parámetros en el menú **Herramientas**:

| Opción de Menú | Parámetro a Seleccionar | Razón Técnica |
| :--- | :--- | :--- |
| **Placa (Board)** | `XIAO_ESP32S3` | Define los pines nativos del factor de forma XIAO. |
| **PSRAM** | `OPI PSRAM` | **Crítico:** Habilita los 8 MB de memoria para triple buffering. |
| **Flash Size** | `8MB (64Mb)` | Capacidad física de la memoria flash del XIAO. |
| **Partition Scheme** | `Huge APP (3MB No OTA/1MB SPIFFS)` | Otorga suficiente espacio para el binario y librerías de cámara. |
| **Core Debug Level** | `None` o `Error` | Minimiza la latencia de serial para mantener 30 FPS. |
| **Upload Speed** | `921600` | Velocidad de carga rápida. |

---

## 🚀 Flasheo del Firmware

1. Abre el sketch ubicado en:
   [`versions/v5_web_xiao_esp32s3/firmware/XIAO_Camera_Orion/XIAO_Camera_Orion.ino`](../versions/v5_web_xiao_esp32s3/firmware/XIAO_Camera_Orion/XIAO_Camera_Orion.ino)
2. En las líneas iniciales de `XIAO_Camera_Orion.ino`, ingresa las credenciales de tu red WiFi:
   ```cpp
   const char* ssid = "TU_RED_WIFI";
   const char* password = "TU_CONTRASENA";
   ```
3. Presiona el botón **Subir (Upload)** en Arduino IDE.
4. Abre el **Monitor Serie** a `115200 baudios`.
5. Una vez conectado al WiFi, verás en la consola:
   ```text
   WiFi conectado!
   IP Asignada: 172.16.121.118
   Servidor de Control: http://172.16.121.118:80
   Servidor de Stream:  http://172.16.121.118:81/stream
   ```

---

## 🧠 Optimizaciones de Firmware Implementadas

### 1. Eliminación del Timeout Socket Prematuro (`SO_SNDTIMEO`)
En versiones estándar de los ejemplos de cámara de Espressif, se configuraba `SO_SNDTIMEO` a 150 ms en el socket de stream. Debido a las variaciones naturales de latencia en redes Wi-Fi (retransmisiones y retrasos de ACK TCP de 100 a 200 ms), el servidor de la cámara abortaba la conexión tras enviar el primer fotograma, congelando el video. En este firmware, se configuraron timeouts globales de 5 segundos con desconexión ordenada.

### 2. Estabilidad Térmica del Sensor (XCLK a 16 MHz)
El reloj de píxeles (`xclk_freq_hz`) fue ajustado a **16 MHz** (`16000000 Hz`) en lugar de 20 MHz. En el factor de forma ultracompacto del XIAO, 20 MHz provoca fluctuación de reloj (*clock jitter*) en el cable flex cuando el procesador toma temperatura, provocando que la función `esp_camera_fb_get()` se congele indefinidamente esperando una señal VSYNC que nunca llega. A 16 MHz el stream es completamente estable a 30 FPS sin calentamiento excesivo.

### 3. Asignación del Servidor al Núcleo 1 (Core 1)
El servidor HTTP del stream MJPEG corre anclado al **Núcleo 1** del procesador ESP32-S3 (`config.ctrl_port = 81`, Core ID 1), dejando el **Núcleo 0** libre para la pila TCP/IP de Wi-Fi y tareas de control.
