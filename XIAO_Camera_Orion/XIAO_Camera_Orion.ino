// ============================================================================
// XIAO_Camera_Orion.ino
// Firmware de Alta Velocidad (30+ FPS Sostenidos) para Seeed Studio XIAO ESP32S3
// Optimizado: Resolución QVGA (320x240), XCLK 20MHz y Doble Buffer en PSRAM
// ============================================================================

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include "camera_pins.h"
#include "xiao_stream_httpd.h"

// ----------------------------------------------------------------------------
// Configuración de Red Wi-Fi (Directo a Audacia)
// ----------------------------------------------------------------------------
const char* ssid     = "Audacia";
const char* password = "R3d4ud4c14?2026*&wl?!";

// ----------------------------------------------------------------------------
// Configuración de Faro UDP (Para enlace inicial con ORION en PC)
// ----------------------------------------------------------------------------
WiFiUDP udp;
const int UDP_BEACON_PORT = 9999;
unsigned long lastBeaconTime = 0;
const unsigned long BEACON_INTERVAL_MS = 3000;

void sendBeacon() {
    if (WiFi.status() != WL_CONNECTED) return;
    String beaconMsg = "ORION_XIAO_CAM:" + WiFi.localIP().toString() + ":81";
    udp.beginPacket(IPAddress(255, 255, 255, 255), UDP_BEACON_PORT);
    udp.write((const uint8_t*)beaconMsg.c_str(), beaconMsg.length());
    udp.endPacket();
}

void setup() {
    Serial.begin(115200);
    delay(600);

    // Monitoreo activo de hardware: se descomenta para auditar posibles FB-OVF en Serial
    // esp_log_level_set("cam_hal", ESP_LOG_ERROR);

    Serial.println("\n==================================================");
    Serial.println("🚀  Iniciando XIAO ESP32S3 Sense - ORION Vision");
    Serial.println("==================================================");

    // ------------------------------------------------------------------------
    // 1. INICIALIZACIÓN INMEDIATA DE PSRAM (8 MB OPI)
    // ------------------------------------------------------------------------
    if (!psramFound()) {
        psramInit();
    }
    size_t psram_size = ESP.getPsramSize();
    bool has_psram = psramFound() || (psram_size > 0);

    if (has_psram) {
        Serial.printf("✅ [PSRAM] PSRAM activa (%d MB). Doble buffer habilitado a 30+ FPS.\n",
                      (int)(psram_size > 0 ? (psram_size / (1024 * 1024)) : 8));
    } else {
        Serial.println("⚠️ [PSRAM] PSRAM no detectada. Verifica en Arduino IDE: Tools > PSRAM: 'OPI PSRAM'.");
    }

    // ------------------------------------------------------------------------
    // 2. CONEXIÓN WI-FI DIRECTA A AUDACIA
    // Enlace nativo directo sin escaneo previo para permitir roaming fluido
    // ------------------------------------------------------------------------
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);

    Serial.printf("📡 [WIFI] Conectando directamente a '%s'...", ssid);
    WiFi.begin(ssid, password);

    unsigned long startWait = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(400);
        Serial.print(".");
        if (millis() - startWait > 25000) {
            Serial.println("\n🔄 [WIFI] Reintentando conexión...");
            WiFi.disconnect();
            delay(200);
            WiFi.begin(ssid, password);
            startWait = millis();
        }
    }

    Serial.println("\n✅ [WIFI] ¡Conectado con éxito!");
    // Desactivar permanentemente el modo de ahorro de energía (Modem-Sleep) de la radio Wi-Fi.
    // En ESP32 viene activo por defecto tras WiFi.begin y genera pausas de 150-200ms entre balizas DTIM.
    esp_wifi_set_ps(WIFI_PS_NONE);
    // Configurar potencia equilibrada (17 dBm): excelente alcance sin sobrecalentar el sensor OV2640
    WiFi.setTxPower(WIFI_POWER_17dBm);

    Serial.printf("   📍 Dirección IP: http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("   📹 URL Stream:   http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("   ⚙️ URL Control:  http://%s/status\n", WiFi.localIP().toString().c_str());

    // ------------------------------------------------------------------------
    // 3. INICIALIZACIÓN DE LA CÁMARA (OV2640 - ESTABILIDAD 16 MHz Y TRIPLE BÚFER)
    // ------------------------------------------------------------------------
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    // 16 MHz: Frecuencia recomendada por Seeed Studio para máxima estabilidad de reloj sin desincronización VSYNC
    config.xclk_freq_hz = 16000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Resolución QVGA (320x240): Ultra ligera (~2.5 KB por frame con quality=20),
    // reduce el peso de transmisión al 50% y maximiza la tasa de FPS sostenida.
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 20;

    if (has_psram) {
        config.fb_count     = 3;                 // Triple buffer DMA en PSRAM: previene DMA starvation
        config.grab_mode    = CAMERA_GRAB_LATEST; // Siempre entrega el fotograma más fresco sin acumular latencia
        config.fb_location  = CAMERA_FB_IN_PSRAM;
    } else {
        config.fb_count     = 2;
        config.grab_mode    = CAMERA_GRAB_LATEST;
        config.fb_location  = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("❌ [CAM] Error al inicializar cámara: 0x%x\n", err);
        return;
    }
    Serial.println("✅ [CAM] Sensor OV2640 activo en QVGA (320x240) a 20 MHz.");

    // Ajustes finos de imagen
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
    }

    // ------------------------------------------------------------------------
    // 4. INICIAR SERVIDOR WEB Y FARO UDP
    // ------------------------------------------------------------------------
    startCameraServer();

    udp.begin(UDP_BEACON_PORT);
    for (int i = 0; i < 3; i++) {
        sendBeacon();
        delay(40);
    }
    Serial.println("📻 [BEACON] Faro de anuncio UDP activo en puerto 9999.");

    Serial.println("\n==================================================");
    Serial.println("🎉  XIAO ESP32S3 transmitiendo a máxima fluidez (30+ FPS)");
    Serial.println("==================================================\n");
}

void loop() {
    unsigned long now = millis();

    // Silenciar el faro UDP mientras haya un cliente transmitiendo video para no pausar el flujo
    if (!isClientStreaming() && WiFi.status() == WL_CONNECTED && (now - lastBeaconTime >= BEACON_INTERVAL_MS)) {
        lastBeaconTime = now;
        sendBeacon();
    }

    delay(20);
}
