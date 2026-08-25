#ifndef APP_HTTPD_H
#define APP_HTTPD_H

// Public API for the camera web server module
void startCameraServer();
void setupLedFlash();
void startRawSocketStreamServer(uint16_t port);

// Optional: expose LED control when available
#if defined(LED_GPIO_NUM)
void enable_led(bool en);
#endif

#endif // APP_HTTPD_H
