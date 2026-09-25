// ============================================================================
// camera_pins.h - Definición de pines para Seeed Studio XIAO ESP32S3 Sense
// ============================================================================
#ifndef CAMERA_PINS_H_
#define CAMERA_PINS_H_

// Pinout oficial Seeed Studio XIAO ESP32S3 Sense (Cámara OV2640 / OV5640)
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39

#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// Pin para Buzzer pasivo/activo en placa de expansión XIAO (opcional)
#define BUZZER_PIN     2

#endif // CAMERA_PINS_H_
