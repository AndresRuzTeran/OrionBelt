# 🛠️ Guía de Hardware: ESP32-CAM (AI-Thinker)

Esta guía explica la conexión y flasheo de la placa clásica **ESP32-CAM de AI-Thinker** utilizada en ORION v4.

---

## 🔌 Cableado con Programador FTDI USB a Serie

A diferencia de la placa XIAO ESP32S3 (que cuenta con puerto USB-C nativo con soporte JTAG/CDC), la placa ESP32-CAM tradicional no incluye chip conversor USB a UART. Debe programarse mediante un módulo FTDI o CH340 externo:

```
+--------------------+           +----------------------+
|  Programador FTDI  |           |      ESP32-CAM       |
+--------------------+           +----------------------+
|  VCC (5V / 3.3V)*  | --------> |  5V (Recomendado)    |
|  GND               | --------> |  GND                 |
|  TX                | --------> |  U0R (GPIO 3 - RX)   |
|  RX                | --------> |  U0T (GPIO 1 - TX)   |
+--------------------+           +----------------------+

* Modo Programación (Flasheo):
  Unir con un cable puente (jumper):
  [ GPIO 0 ] <------------> [ GND ]
```

> [!IMPORTANT]
> **Alimentación:** Aunque el ESP32 trabaja a 3.3V, el módulo de cámara requiere picos de corriente de hasta 500 mA durante la transmisión Wi-Fi. Se recomienda alimentar por el pin de **5V** utilizando una fuente USB estable para evitar reinicios continuos por caída de tensión (*brownout reset*).

---

## ⚙️ Configuración en Arduino IDE

1. **Gestor de Placas:**
   - Selecciona **AI Thinker ESP32-CAM** en el menú de placas.
2. **Configuración de Parámetros:**
   - **CPU Frequency:** `240MHz (WiFi/BT)`
   - **Flash Frequency:** `80MHz`
   - **Flash Mode:** `QIO`
   - **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)`
3. **Firmware:**
   - Abre el sketch ubicado en:
     [`versions/v4_web_esp32/firmware/CameraWebServer/CameraWebServer.ino`](../versions/v4_web_esp32/firmware/CameraWebServer/CameraWebServer.ino)
   - Configura el SSID y contraseña en el archivo.
   - En `CameraWebServer.ino`, verifica que esté descomentada la línea:
     ```cpp
     #define CAMERA_MODEL_AI_THINKER // Has PSRAM
     ```
4. **Pasos para Flashear:**
   1. Conecta el puente entre **GPIO 0** y **GND**.
   2. Presiona el botón de **RESET** en la parte inferior de la placa ESP32-CAM.
   3. Haz clic en **Subir** en Arduino IDE.
   4. Una vez finalizada la subida ("Leaving... Hard resetting via RTS pin..."), **desconecta el puente entre GPIO 0 y GND**.
   5. Presiona el botón **RESET** una vez más para iniciar el firmware en modo normal.
