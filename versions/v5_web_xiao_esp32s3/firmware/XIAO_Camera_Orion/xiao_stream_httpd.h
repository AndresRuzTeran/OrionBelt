// ============================================================================
// xiao_stream_httpd.h - Servidor HTTP optimizado para XIAO ESP32S3
// ============================================================================
#ifndef XIAO_STREAM_HTTPD_H_
#define XIAO_STREAM_HTTPD_H_

#ifdef __cplusplus
extern "C" {
#endif

void startCameraServer();
void setupBuzzer();
void setBuzzerState(int on, int duty);
void beepBuzzer(int duty, int duration_ms, int freq);
bool isClientStreaming();

#ifdef __cplusplus
}
#endif

#endif // XIAO_STREAM_HTTPD_H_
