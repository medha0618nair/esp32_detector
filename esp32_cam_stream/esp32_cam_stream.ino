/*
 * esp32_cam_stream.ino - Turns an AI-Thinker ESP32-CAM into a Wi-Fi camera.
 *
 * This is the camera half of the project. The ESP32-CAM only captures JPEG
 * frames and streams them over Wi-Fi as MJPEG. The laptop runs main.py,
 * which reads the stream, recognises objects with YOLO and speaks their
 * names through the laptop speaker.
 *
 *   ESP32-CAM (this sketch) --Wi-Fi MJPEG--> laptop (main.py: YOLO + say)
 *
 * Stream URL (set it as ESP32_STREAM_URL in config.py):
 *   http://esp32cam.local:81/stream   (no IP needed, works on macOS)
 *   http://<IP printed on Serial Monitor>:81/stream
 *
 * Arduino IDE settings:
 *   Board: "AI Thinker ESP32-CAM"   (esp32 by Espressif board package)
 *   Serial Monitor: 115200 baud
 *   Uploading with an FTDI adapter (not needed with the ESP32-CAM-MB USB board):
 *   connect GPIO0 to GND, press RESET, upload, then remove the jumper and
 *   press RESET again.
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <ESPmDNS.h>

// ---------------------------------------------------------------------------
// Settings - put your Wi-Fi details here (2.4 GHz networks only)
// ---------------------------------------------------------------------------
const char *WIFI_SSID     = "YOUR_WIFI_NAME";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Reachable as http://esp32cam.local:81/stream
const char *HOSTNAME = "esp32cam";

// VGA (640x480) is plenty for YOLO and keeps the stream smooth.
// Options: FRAMESIZE_QVGA (320x240), FRAMESIZE_VGA, FRAMESIZE_SVGA (800x600)
const framesize_t FRAME_SIZE = FRAMESIZE_VGA;

// JPEG quality: 10-63, lower = better quality but bigger frames.
const int JPEG_QUALITY = 12;

// ---------------------------------------------------------------------------
// AI-Thinker ESP32-CAM camera pins
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ---------------------------------------------------------------------------
// MJPEG streaming: an endless HTTP response where each part is one JPEG.
// OpenCV's cv2.VideoCapture(url) understands this format directly.
// ---------------------------------------------------------------------------
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  Serial.println("Client connected to stream");

  char part_header[64];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    size_t header_len = snprintf(part_header, sizeof(part_header), STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_header, header_len);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);

    if (res != ESP_OK) break;  // client disconnected
  }
  Serial.println("Client disconnected from stream");
  return res;
}

void start_stream_server() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  } else {
    Serial.println("Failed to start stream server");
  }
}

bool init_camera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  // Always hand out the newest frame, so the laptop sees what's in front
  // of the camera now rather than a backlog of old frames.
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 2;
  } else {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
    config.frame_size = FRAMESIZE_QVGA;  // without PSRAM, VGA may not fit
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  // If the picture is upside down or mirrored, flip these to 1:
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 0);
  s->set_hmirror(s, 0);
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.println();

  if (!init_camera()) {
    Serial.println("Check the camera ribbon cable and power supply, then press RESET.");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // Wi-Fi power saving makes the stream stutter
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to Wi-Fi \"%s\"", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" connected");

  if (MDNS.begin(HOSTNAME)) {
    MDNS.addService("http", "tcp", 81);
  }

  start_stream_server();

  Serial.println("Ready! Put one of these in config.py as ESP32_STREAM_URL:");
  Serial.printf("  http://%s.local:81/stream\n", HOSTNAME);
  Serial.printf("  http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() {
  // The stream server runs in its own task; nothing to do here.
  delay(10000);
}
