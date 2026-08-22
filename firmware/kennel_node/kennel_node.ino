/*
 * QMLKit Kennel Node - ESP32-WROOM-32 firmware
 *
 * Canine olfactory examination kennel telemetry node.
 *   4x FSR pressure sensors      -> ADC1 pins 34/35/36/39 (bottom corners)
 *   6x IR proximity sensors      -> 4 lower corners (19/18/17/16) +
 *                                   2 top front L/R (5/23)
 *   2x HC-SR04 ultrasonic        -> bottom (TRIG 26 / ECHO 27),
 *                                   top    (TRIG 14 / ECHO 13)
 *   1x MPU6050 (collar-mounted)  -> I2C SDA 21 / SCL 22, ~100 Hz
 *
 * Streams newline-delimited JSON frames over a TCP server on port 3333
 * (GAIT-monorepo compatible contract). The Python server connects IN as a
 * client. Serial @115200 mirrors frames for debugging.
 *
 * Commands (send a single char over any TCP client or Serial):
 *   z -> re-zero FSR baselines     s -> one-line status JSON
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <ESPmDNS.h>
#include <Wire.h>
#include <Preferences.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ============================ PIN MAP ============================
static const int PIN_FSR[4]        = {34, 35, 36, 39}; // FL, FR, RL, RR (ADC1)
static const int PIN_IR_BOTTOM[4]  = {19, 18, 17, 16};  // FL, FR, RL, RR (active-LOW modules)
static const int PIN_IR_TOP[2]     = {5, 23};           // front-left, front-right
static const int PIN_US_BOT_TRIG   = 26;
static const int PIN_US_BOT_ECHO   = 27;
static const int PIN_US_TOP_TRIG   = 14;
static const int PIN_US_TOP_ECHO   = 13;
static const int PIN_I2C_SDA       = 21;
static const int PIN_I2C_SCL       = 22;
static const int PIN_LED           = 2;

// ============================ TUNABLES ============================
static const uint16_t TCP_PORT          = 3333;
static const uint32_t IMU_INTERVAL_MS   = 10;    // ~100 Hz collar sampling
static const uint32_t SLOW_INTERVAL_MS  = 50;    // FSR/IR/ultrasonic rate
static const uint32_t TX_INTERVAL_MS    = 100;   // frame batching period
static const uint32_t CALIBRATION_MS    = 2000;  // FSR baseline window (info)
static const uint32_t SNIFF_WINDOW_MS   = 8000;  // tagged capture length
static const uint32_t COOLDOWN_MS       = 10000;
static const uint32_t HEAD_HOLD_MS      = 500;   // sustained head presence to sniff

static const float    US_BODY_DIST_CM   = 60.0;  // bottom sensor: dog present
static const float    US_HEAD_DIST_CM   = 90.0;  // top sensor: head near sample
static const uint32_t OCC_LOAD_DELTA     = 400;   // summed FSR delta vs baseline
static const uint32_t LOAD_LOST_MS       = 3000;  // load gone this long -> IDLE

static const char*    AP_FALLBACK_SSID  = "QMLKit-Kennel";
static const char*    AP_FALLBACK_PASS  = "sniff1234";
static const char*    HOSTNAME          = "kennel";

// ============================ STATE ============================
enum KennelState : uint8_t { ST_BOOT, ST_CALIBRATE, ST_IDLE, ST_OCCUPIED, ST_SNIFF, ST_COOLDOWN };
static const char* STATE_NAMES[] = {"BOOT","CALIBRATE","IDLE","OCCUPIED","SNIFF","COOLDOWN"};

static KennelState g_state = ST_BOOT;

static Adafruit_MPU6050 mpu;
static bool g_mpu_ok = false;

static WiFiServer g_server(TCP_PORT);
static WiFiClient g_clients[4];

static uint16_t g_fsr_raw[4]      = {0, 0, 0, 0};
static uint16_t g_fsr_baseline[4] = {0, 0, 0, 0};
static uint32_t g_load_delta      = 0;
static bool     g_ir_bottom[4]    = {false, false, false, false};
static bool     g_ir_top[2]       = {false, false};
static float    g_us_bottom_cm    = NAN;
static float    g_us_top_cm       = NAN;
static float    g_acc[3]          = {0, 0, 0};
static float    g_gyr[3]          = {0, 0, 0};
static float    g_imu_temp_c      = NAN;

static uint64_t g_seq             = 0;
static uint32_t g_sniff_started   = 0;
static uint32_t g_head_since      = 0;
static uint32_t g_last_seen_ms    = 0;
static uint32_t g_cooldown_until  = 0;
static uint32_t g_next_imu_ms     = 0;
static uint32_t g_next_slow_ms    = 0;
static uint32_t g_next_tx_ms      = 0;
static uint32_t g_next_serial_ms  = 0;

static Preferences g_prefs;

// ============================ HELPERS ============================
static float readUltrasonicCm(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(3);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  const unsigned long timeout = 25000UL;  // ~4.3 m range
  unsigned long dur = pulseIn(echoPin, HIGH, timeout);
  if (dur == 0) return NAN;
  return dur / 58.0f;
}

static void readSlowChannels() {
  uint32_t delta_sum = 0;
  for (int i = 0; i < 4; i++) {
    g_fsr_raw[i] = analogRead(PIN_FSR[i]);
    delta_sum += g_fsr_raw[i] > g_fsr_baseline[i] ? (g_fsr_raw[i] - g_fsr_baseline[i]) : 0;
  }
  g_load_delta = delta_sum;

  for (int i = 0; i < 4; i++) g_ir_bottom[i] = digitalRead(PIN_IR_BOTTOM[i]) == LOW;
  for (int i = 0; i < 2; i++) g_ir_top[i]    = digitalRead(PIN_IR_TOP[i])    == LOW;

  g_us_bottom_cm = readUltrasonicCm(PIN_US_BOT_TRIG, PIN_US_BOT_ECHO);
  g_us_top_cm    = readUtrasonicCm(PIN_US_TOP_TRIG, PIN_US_TOP_ECHO);
}

static void zeroBaselines() {
  uint32_t acc[4] = {0, 0, 0, 0};
  const int N = 32;
  for (int k = 0; k < N; k++) {
    for (int i = 0; i < 4; i++) acc[i] += analogRead(PIN_FSR[i]);
    delay(5);
  }
  for (int i = 0; i < 4; i++) g_fsr_baseline[i] = acc[i] / N;
}

static bool headDetected() {
  if (g_ir_top[0] || g_ir_top[1]) return true;
  return (!isnan(g_us_top_cm) && g_us_top_cm > 0 && g_us_top_cm < US_HEAD_DIST_CM);
}

static bool bodyDetected() {
  bool us_ok = isnan(g_us_bottom_cm) || g_us_bottom_cm < US_BODY_DIST_CM;
  return g_load_delta > OCC_LOAD_DELTA && us_ok;
}

static void setState(KennelState s) {
  if (g_state == s) return;
  g_state = s;
  if (s == ST_SNIFF) g_sniff_started = millis();
}

static void updateStateMachine(uint32_t now) {
  switch (g_state) {
    case ST_CALIBRATE:
      break;  // handled during setup()
    case ST_IDLE:
      if (bodyDetected()) {
        g_last_seen_ms = now;
        setState(ST_OCCUPIED);
      }
      break;
    case ST_OCCUPIED:
      if (!bodyDetected()) {
        if (now - g_last_seen_ms > LOAD_LOST_MS) setState(ST_IDLE);
      } else {
        g_last_seen_ms = now;
        if (headDetected()) {
          if (g_head_since == 0) g_head_since = now;
          if (now - g_head_since >= HEAD_HOLD_MS) {
            g_head_since = 0;
            setState(ST_SNIFF);
          }
        } else {
          g_head_since = 0;
        }
      }
      break;
    case ST_SNIFF:
      if (now - g_sniff_started >= SNIFF_WINDOW_MS) {
        g_cooldown_until = now + COOLDOWN_MS;
        setState(ST_COOLDOWN);
      } else if (!bodyDetected() && now - g_last_seen_ms > LOAD_LOST_MS) {
        setState(ST_IDLE);
      }
      break;
    case ST_COOLDOWN:
      if (now >= g_cooldown_until) setState(ST_IDLE);
      break;
    default:
      setState(ST_IDLE);
      break;
  }
}

static void ledPattern(uint32_t now) {
  uint32_t on = 0, off = 0;
  switch (g_state) {
    case ST_BOOT:      on = 60;  off = 60;  break;  // fast blink
    case ST_CALIBRATE: on = 500; off = 0;   break;  // solid
    case ST_IDLE:      on = 40;  off = 960; break;  // heartbeat
    case ST_OCCUPIED:  on = 120; off = 120; break;  // double-blink-ish
    case ST_SNIFF:     on = 30;  off = 30;  break;  // rapid
    case ST_COOLDOWN:  on = 500; off = 500; break;
  }
  uint32_t cycle = now % (on + off + 1);
  digitalWrite(PIN_LED, cycle < on ? HIGH : LOW);
}

// ---- JSON framing -------------------------------------------------------
static size_t buildFrame(char* buf, size_t cap) {
  snprintf(buf, cap,
    "{\"ts_ms\":%lu,\"seq\":%llu,\"state\":\"%s\","
    "\"fsr\":[%u,%u,%u,%u],"
    "\"ir\":[%d,%d,%d,%d,%d,%d],"
    "\"us\":{\"bottom\":%.1f,\"top\":%.1f},"
    "\"acc\":[%.3f,%.3f,%.3f],\"gyr\":[%.3f,%.3f,%.3f],"
    "\"imu_temp_c\":%.1f}",
    (unsigned long)millis(), (unsigned long long)g_seq, STATE_NAMES[g_state],
    g_fsr_raw[0], g_fsr_raw[1], g_fsr_raw[2], g_fsr_raw[3],
    (int)g_ir_bottom[0], (int)g_ir_bottom[1], (int)g_ir_bottom[2], (int)g_ir_bottom[3],
    (int)g_ir_top[0], (int)g_ir_top[1],
    (double)(isnan(g_us_bottom_cm) ? -1.0 : g_us_bottom_cm),
    (double)(isnan(g_us_top_cm) ? -1.0 : g_us_top_cm),
    (double)g_acc[0], (double)g_acc[1], (double)g_acc[2],
    (double)g_gyr[0], (double)g_gyr[1], (double)g_gyr[2],
    (double)(isnan(g_imu_temp_c) ? -1.0 : g_imu_temp_c));
  return strlen(buf);
}

static void broadcastFrame(const char* buf) {
  for (auto& c : g_clients) {
    if (c && c.connected()) c.println(buf);
  }
}

static void acceptClients() {
  while (true) {
    WiFiClient nc = g_server.available();
    if (!nc) break;
    bool placed = false;
    for (auto& c : g_clients) {
      if (!c || !c.connected()) { c = nc; placed = true; break; }
    }
    if (!placed) nc.stop();  // slot table full
  }

  // Handle single-char commands + prune dead clients.
  for (auto& c : g_clients) {
    if (!c || !c.connected()) continue;
    while (c.available() > 0) {
      char cmd = (char)c.read();
      if (cmd == 'z') {
        zeroBaselines();
      } else if (cmd == 's') {
        char buf[320];
        size_t n = buildFrame(buf, sizeof(buf));
        c.write((const uint8_t*)buf, n);
        c.println();
      }
    }
  }
}

// ============================ SETUP / LOOP ============================
void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED, OUTPUT);
  for (int p : PIN_FSR) pinMode(p, INPUT);
  for (int p : PIN_IR_BOTTOM) pinMode(p, INPUT_PULLUP);
  for (int p : PIN_IR_TOP) pinMode(p, INPUT_PULLUP);
  pinMode(PIN_US_BOT_TRIG, OUTPUT); pinMode(PIN_US_BOT_ECHO, INPUT);
  pinMode(PIN_US_TOP_TRIG, OUTPUT); pinMode(PIN_US_TOP_ECHO, INPUT);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);
  g_mpu_ok = mpu.begin();
  if (g_mpu_ok) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setBandwidth(MPU6050_BAND_44_HZ);  // anti-alias for ~100 Hz sampling
  }

  // --- WiFi: stored credentials first, SoftAP fallback ---
  g_prefs.begin("kennel", true);
  String ssid = g_prefs.getString("ssid", "");
  String pass = g_prefs.getString("pass", "");
  g_prefs.end();

  bool sta_ok = false;
  if (ssid.length()) {
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(HOSTNAME);
    WiFi.begin(ssid.c_str(), pass.c_str());
    const uint32_t deadline = millis() + 20000;
    while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
      ledPattern(millis());
      delay(50);
    }
    sta_ok = WiFi.status() == WL_CONNECTED;
  }
  if (!sta_ok) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_FALLBACK_SSID, AP_FALLBACK_PASS);
  }

  if (MDNS.begin(HOSTNAME)) MDNS.addService("qmlkit-kennel", "tcp", TCP_PORT);
  g_server.begin();
  g_server.setNoDelay(true);

  setState(ST_CALIBRATE);
  zeroBaselines();
  setState(ST_IDLE);

  enableLoopWDT();  // recover from hangs automatically
}

void loop() {
  const uint32_t now = millis();

  if (g_mpu_ok && now >= g_next_imu_ms) {
    sensors_event_t a, gv, temp;
    mpu.getEvent(&a, &gv, &temp);
    g_acc[0] = a.acceleration.x; g_acc[1] = a.acceleration.y; g_acc[2] = a.acceleration.z;
    g_gyr[0] = gv.gyro.x;        g_gyr[1] = gv.gyro.y;        g_gyr[2] = gv.gyro.z;
    g_imu_temp_c = temp.temperature;
    g_next_imu_ms = now + IMU_INTERVAL_MS;
  }

  if (now >= g_next_slow_ms) {
    readSlowChannels();
    updateStateMachine(now);
    g_next_slow_ms = now + SLOW_INTERVAL_MS;
  }

  acceptClients();
  ledPattern(now);

  if (now >= g_next_tx_ms) {
    char buf[320];
    buildFrame(buf, sizeof(buf));
    broadcastFrame(buf);
    g_seq++;
    if (now >= g_next_serial_ms) {
      Serial.println(buf);
      g_next_serial_ms = now + 2000;  // mirror at 0.5 Hz to keep serial light
    }
    g_next_tx_ms = now + TX_INTERVAL_MS;
  }

  feedLoopWDT();
}
