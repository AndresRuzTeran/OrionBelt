// ============================================================================
// xiao_stream_httpd.cpp - Servidor de Streaming de Alta Velocidad y Control
// Optimizado para Seeed Studio XIAO ESP32S3 Sense (30 FPS Sostenidos)
// ============================================================================

#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "Arduino.h"
#include <lwip/sockets.h>
#include "camera_pins.h"
#include "xiao_stream_httpd.h"

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

httpd_handle_t stream_httpd = NULL;
httpd_handle_t control_httpd = NULL;

#if __has_include("esp_arduino_version.h")
#include "esp_arduino_version.h"
#endif

// ----------------------------------------------------------------------------
// Buzzer PWM (LEDC) - Compatible con ESP32 Arduino Core 2.x y 3.x
// ----------------------------------------------------------------------------
#define BUZZER_CHANNEL 1
#define BUZZER_FREQ 2000
#define BUZZER_RES 8

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
// API para ESP32 Arduino Core 3.0+
void setupBuzzer() {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    ledcAttach(BUZZER_PIN, BUZZER_FREQ, BUZZER_RES);
    ledcWrite(BUZZER_PIN, 0);
#endif
}

void setBuzzerState(int on, int duty) {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    if (on) {
        ledcWrite(BUZZER_PIN, duty > 0 ? duty : 200);
    } else {
        ledcWrite(BUZZER_PIN, 0);
    }
#endif
}

void beepBuzzer(int duty, int duration_ms, int freq) {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    if (freq > 0) {
        ledcAttach(BUZZER_PIN, freq, BUZZER_RES);
    }
    ledcWrite(BUZZER_PIN, duty > 0 ? duty : 200);
    delay(duration_ms > 0 ? duration_ms : 100);
    ledcWrite(BUZZER_PIN, 0);
#endif
}

#else
// API para ESP32 Arduino Core 2.x (Legacy)
void setupBuzzer() {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    ledcSetup(BUZZER_CHANNEL, BUZZER_FREQ, BUZZER_RES);
    ledcAttachPin(BUZZER_PIN, BUZZER_CHANNEL);
    ledcWrite(BUZZER_CHANNEL, 0);
#endif
}

void setBuzzerState(int on, int duty) {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    if (on) {
        ledcWrite(BUZZER_CHANNEL, duty > 0 ? duty : 200);
    } else {
        ledcWrite(BUZZER_CHANNEL, 0);
    }
#endif
}

void beepBuzzer(int duty, int duration_ms, int freq) {
#if defined(BUZZER_PIN) && (BUZZER_PIN >= 0)
    if (freq > 0) {
        ledcSetup(BUZZER_CHANNEL, freq, BUZZER_RES);
    }
    ledcWrite(BUZZER_CHANNEL, duty > 0 ? duty : 200);
    delay(duration_ms > 0 ? duration_ms : 100);
    ledcWrite(BUZZER_CHANNEL, 0);
#endif
}
#endif

// ----------------------------------------------------------------------------
// Handler: /stream (Puerto 81 - MJPEG de Alta Velocidad)
// ----------------------------------------------------------------------------
static volatile bool s_client_streaming = false;

bool isClientStreaming() {
    return s_client_streaming;
}

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];

    // Optimización de baja latencia: desactivar algoritmo de Nagle (TCP_NODELAY)
    int sockfd = httpd_req_to_sockfd(req);
    if (sockfd >= 0) {
        int nodelay = 1;
        setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&nodelay, sizeof(nodelay));
    }

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "30");

    s_client_streaming = true;

    while (true) {
        // Obtener fotograma del buffer DMA (sincronizado por hardware VSYNC del OV2640 a ~30 FPS)
        fb = esp_camera_fb_get();
        if (!fb) {
            res = ESP_FAIL;
            break;
        }

        if (fb->format == PIXFORMAT_JPEG) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf), 
                                   "\r\n--" PART_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", 
                                   fb->len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
            if (res == ESP_OK) {
                res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
            }
        }

        esp_camera_fb_return(fb);
        fb = NULL;

        if (res != ESP_OK) {
            // El cliente cerró la pestaña del navegador
            break;
        }

        // Ceder 1 tick real de FreeRTOS (10ms) a la pila Wi-Fi LwIP para vaciar paquetes TCP sin colapso
        vTaskDelay(1);
    }

    s_client_streaming = false;
    return res;
}

// ----------------------------------------------------------------------------
// Handler: /status (Puerto 80 - JSON de Telemetría)
// ----------------------------------------------------------------------------
static esp_err_t status_handler(httpd_req_t *req) {
    static char json_response[256];
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    char *p = json_response;
    *p++ = '{';
    p += sprintf(p, "\"device\":\"XIAO_ESP32S3\",");
    p += sprintf(p, "\"framesize\":%u,", s->status.framesize);
    p += sprintf(p, "\"quality\":%u,", s->status.quality);
    p += sprintf(p, "\"brightness\":%d,", s->status.brightness);
    p += sprintf(p, "\"contrast\":%d,", s->status.contrast);
    p += sprintf(p, "\"saturation\":%d", s->status.saturation);
    *p++ = '}';
    *p = 0;

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

// ----------------------------------------------------------------------------
// Handler: /control (Puerto 80 - Ajuste Dinámico de Parámetros)
// ----------------------------------------------------------------------------
static esp_err_t control_handler(httpd_req_t *req) {
    char *buf = NULL;
    size_t buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char *)malloc(buf_len);
        if (!buf) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            char var[32] = {0};
            char val[32] = {0};
            if (httpd_query_key_value(buf, "var", var, sizeof(var)) == ESP_OK &&
                httpd_query_key_value(buf, "val", val, sizeof(val)) == ESP_OK) {
                sensor_t *s = esp_camera_sensor_get();
                int int_val = atoi(val);
                if (s) {
                    if (!strcmp(var, "framesize")) s->set_framesize(s, (framesize_t)int_val);
                    else if (!strcmp(var, "quality")) s->set_quality(s, int_val);
                    else if (!strcmp(var, "contrast")) s->set_contrast(s, int_val);
                    else if (!strcmp(var, "brightness")) s->set_brightness(s, int_val);
                    else if (!strcmp(var, "saturation")) s->set_saturation(s, int_val);
                }
            }
        }
        free(buf);
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", 2);
}

// ----------------------------------------------------------------------------
// Handler: /buzzer (Puerto 80 - Control de Proximidad)
// ----------------------------------------------------------------------------
static esp_err_t buzzer_handler(httpd_req_t *req) {
    char *buf = NULL;
    size_t buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char *)malloc(buf_len);
        if (buf && httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            char on_val[8] = {0};
            char duty_val[8] = {0};
            char beep_val[8] = {0};
            char freq_val[8] = {0};

            int duty = 200;
            if (httpd_query_key_value(buf, "duty", duty_val, sizeof(duty_val)) == ESP_OK) {
                duty = atoi(duty_val);
            }

            if (httpd_query_key_value(buf, "beep", beep_val, sizeof(beep_val)) == ESP_OK) {
                int dur = atoi(beep_val);
                int freq = 2000;
                if (httpd_query_key_value(buf, "freq", freq_val, sizeof(freq_val)) == ESP_OK) {
                    freq = atoi(freq_val);
                }
                beepBuzzer(duty, dur, freq);
            } else if (httpd_query_key_value(buf, "on", on_val, sizeof(on_val)) == ESP_OK) {
                setBuzzerState(atoi(on_val), duty);
            }
        }
        if (buf) free(buf);
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", 2);
}

// ----------------------------------------------------------------------------
// Handler: /whoami y / (Identificación Inmediata)
// ----------------------------------------------------------------------------
static esp_err_t whoami_handler(httpd_req_t *req) {
    const char *resp = "{\"device\":\"XIAO_ESP32S3_ORION\",\"status\":\"online\",\"stream_port\":81}";
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

// ----------------------------------------------------------------------------
// Inicialización de Servidores HTTP (Puertos 80 y 81)
// ----------------------------------------------------------------------------
void startCameraServer() {
    setupBuzzer();

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 8;
    config.stack_size = 8192;

    // --- Protección contra desconexiones y conexiones colgadas ---
    config.lru_purge_enable = true;   // Purga automáticamente sockets inactivos ante cierres abruptos
    config.recv_wait_timeout = 5;     // 5 segundos de margen para tolerar fluctuaciones normales de Wi-Fi
    config.send_wait_timeout = 5;     // Evita desconexiones prematuras que maten el stream
    config.backlog_conn = 4;          // Aceptación fluida sin rechazo de conexiones

    // 1. Servidor de Control en Puerto 80
    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };
    httpd_uri_t control_uri = {
        .uri       = "/control",
        .method    = HTTP_GET,
        .handler   = control_handler,
        .user_ctx  = NULL
    };
    httpd_uri_t buzzer_uri = {
        .uri       = "/buzzer",
        .method    = HTTP_GET,
        .handler   = buzzer_handler,
        .user_ctx  = NULL
    };
    httpd_uri_t whoami_uri = {
        .uri       = "/whoami",
        .method    = HTTP_GET,
        .handler   = whoami_handler,
        .user_ctx  = NULL
    };
    httpd_uri_t root_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = whoami_handler,
        .user_ctx  = NULL
    };

    config.server_port = 80;
    config.ctrl_port = 32768;
    if (httpd_start(&control_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(control_httpd, &status_uri);
        httpd_register_uri_handler(control_httpd, &control_uri);
        httpd_register_uri_handler(control_httpd, &buzzer_uri);
        httpd_register_uri_handler(control_httpd, &whoami_uri);
        httpd_register_uri_handler(control_httpd, &root_uri);
        Serial.println("  [HTTP] Servidor de control iniciado en puerto 80.");
    }

    // 2. Servidor de Streaming en Puerto 81
    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    config.server_port = 81;
    config.ctrl_port = 32769;
    config.core_id = 1;               // CRÍTICO: Núcleo 1 exclusivo para streaming, liberando Núcleo 0 para Wi-Fi
    config.max_open_sockets = 2;      // Purga inmediata de socket viejo al recargar la página con F5
    config.lru_purge_enable = true;   // Desconecta sockets inactivos sin dar error 'accept (128)'
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        Serial.println("  [HTTP] Servidor de video stream iniciado en puerto 81 (Core 1).");
    }
}
