/*
 * QMLKit Kennel Node - ESP32-WROOM-32 firmware
 *
 * Canine olfactory examination kennel telemetry node.
 *
 * Sensors:
 *   4x FSR pressure sensors
 *   6x IR proximity sensors
 *   2x HC-SR04 ultrasonic sensors
 *   1x MPU6050
 *
 * WiFi:
 *   Credentials loaded from secrets.h
 *
 * TCP:
 *   Server on port 3333
 *
 * Serial:
 *   115200 baud
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <ESPmDNS.h>
#include <Wire.h>
#include <Preferences.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <SparkFun_MAX3010x_Satellite_Library.h>  // heart-rate + SpO2 (MAX30102, paper §IV-A)

#include "secrets.h"


// ================================================================
// PIN MAP
// ================================================================

static const int PIN_FSR[4] = {
  34, 35, 36, 39
};

static const int PIN_IR_BOTTOM[4] = {
  19, 18, 17, 16
};

static const int PIN_IR_TOP[2] = {
  5, 23
};

static const int PIN_US_BOT_TRIG = 26;
static const int PIN_US_BOT_ECHO = 27;

static const int PIN_US_TOP_TRIG = 14;
static const int PIN_US_TOP_ECHO = 13;

static const int PIN_I2C_SDA = 21;
static const int PIN_I2C_SCL = 22;

static const int PIN_LED = 2;


// ================================================================
// CONFIGURATION
// ================================================================

static const uint16_t TCP_PORT = 3333;

static const uint32_t IMU_INTERVAL_MS = 10;
static const uint32_t SLOW_INTERVAL_MS = 50;
static const uint32_t TX_INTERVAL_MS = 100;

static const uint32_t CALIBRATION_MS = 2000;

static const uint32_t SNIFF_WINDOW_MS = 8000;
static const uint32_t COOLDOWN_MS = 10000;

static const uint32_t HEAD_HOLD_MS = 500;

static const float US_BODY_DIST_CM = 60.0;
static const float US_HEAD_DIST_CM = 90.0;

static const uint32_t OCC_LOAD_DELTA = 400;
static const uint32_t LOAD_LOST_MS = 3000;

static const char* HOSTNAME = "kennel";


// ================================================================
// STATE
// ================================================================

enum KennelState : uint8_t {
  ST_BOOT,
  ST_CALIBRATE,
  ST_IDLE,
  ST_OCCUPIED,
  ST_SNIFF,
  ST_COOLDOWN
};

static const char* STATE_NAMES[] = {
  "BOOT",
  "CALIBRATE",
  "IDLE",
  "OCCUPIED",
  "SNIFF",
  "COOLDOWN"
};

static KennelState g_state = ST_BOOT;


// ================================================================
// SENSOR OBJECTS
// ================================================================

static Adafruit_MPU6050 mpu;

static bool g_mpu_ok = false;


// ================================================================
// NETWORK OBJECTS
// ================================================================

static WiFiServer g_server(TCP_PORT);

static WiFiClient g_clients[4];


// ================================================================
// SENSOR DATA
// ================================================================

static uint16_t g_fsr_raw[4] = {
  0, 0, 0, 0
};

static uint16_t g_fsr_baseline[4] = {
  0, 0, 0, 0
};

static uint32_t g_load_delta = 0;

static bool g_ir_bottom[4] = {
  false, false, false, false
};

static bool g_ir_top[2] = {
  false, false
};

static float g_us_bottom_cm = NAN;
static float g_us_top_cm = NAN;

static float g_acc[3] = {
  0, 0, 0
};

static float g_gyr[3] = {
  0, 0, 0
};

static float g_imu_temp_c = NAN;

// ================================================================
// PHYSIOLOGY (MAX30102) - paper §IV-A
// Values stay -1.0 when the sensor is absent or no probe attached.
// ================================================================

static MAX30105 g_max30102;
static bool g_hr_ok = false;
static float g_hr_bpm = -1.0f;
static float g_spo2_pct = -1.0f;
static uint32_t g_hr_last_ms = 0;
static const uint32_t HR_INTERVAL_MS = 500;  // physiological channel ~2 Hz
static uint32_t g_next_hr_ms = 0;


// ================================================================
// RUNTIME STATE
// ================================================================

static uint64_t g_seq = 0;

static uint32_t g_sniff_started = 0;
static uint32_t g_head_since = 0;
static uint32_t g_last_seen_ms = 0;
static uint32_t g_cooldown_until = 0;

static uint32_t g_next_imu_ms = 0;
static uint32_t g_next_slow_ms = 0;
static uint32_t g_next_tx_ms = 0;
static uint32_t g_next_serial_ms = 0;


// ================================================================
// HELPER: ULTRASONIC
// ================================================================

static float readUltrasonicCm(int trigPin, int echoPin) {

  digitalWrite(trigPin, LOW);

  delayMicroseconds(3);

  digitalWrite(trigPin, HIGH);

  delayMicroseconds(10);

  digitalWrite(trigPin, LOW);

  const unsigned long timeout = 25000UL;

  unsigned long dur = pulseIn(
    echoPin,
    HIGH,
    timeout
  );

  if (dur == 0) {
    return NAN;
  }

  return dur / 58.0f;
}


// ================================================================
// READ SLOW CHANNELS
// ================================================================

static void readSlowChannels() {

  uint32_t delta_sum = 0;

  // -----------------------------
  // FSR
  // -----------------------------

  for (int i = 0; i < 4; i++) {

    g_fsr_raw[i] = analogRead(
      PIN_FSR[i]
    );

    if (g_fsr_raw[i] > g_fsr_baseline[i]) {

      delta_sum +=
        g_fsr_raw[i] -
        g_fsr_baseline[i];
    }
  }

  g_load_delta = delta_sum;


  // -----------------------------
  // IR sensors
  // -----------------------------

  for (int i = 0; i < 4; i++) {

    g_ir_bottom[i] =
      digitalRead(PIN_IR_BOTTOM[i]) == LOW;
  }

  for (int i = 0; i < 2; i++) {

    g_ir_top[i] =
      digitalRead(PIN_IR_TOP[i]) == LOW;
  }


  // -----------------------------
  // Ultrasonic
  // -----------------------------

  g_us_bottom_cm =
    readUltrasonicCm(
      PIN_US_BOT_TRIG,
      PIN_US_BOT_ECHO
    );

  g_us_top_cm =
    readUltrasonicCm(
      PIN_US_TOP_TRIG,
      PIN_US_TOP_ECHO
    );
}


// ================================================================
// PHYSIOLOGY SAMPLING (MAX30102)
// Beat-interval heart rate + empirical R-ratio SpO2 estimate.
// ================================================================

static uint32_t g_last_beat_ms = 0;
static float g_beat_interval_ms = 0.0f;
static float g_red_min = 1e9f, g_red_max = -1e9f;
static float g_ir_min = 1e9f, g_ir_max = -1e9f;

static void readPhysiology(uint32_t now) {

  if (!g_hr_ok) {
    return;
  }

  uint32_t ir_value = g_max30102.getIR();
  uint32_t red_value = g_max30102.getRed();

  if (ir_value < 50000) {
    // No probe / no tissue contact.
    g_hr_bpm = -1.0f;
    g_spo2_pct = -1.0f;
    g_red_min = 1e9f; g_red_max = -1e9f;
    g_ir_min = 1e9f; g_ir_max = -1e9f;
    return;
  }

  // Track AC envelope over the sliding window for SpO2 estimation.
  if (red_value < g_red_min) g_red_min = (float)red_value;
  if (red_value > g_red_max) g_red_max = (float)red_value;
  if (ir_value < g_ir_min) g_ir_min = (float)ir_value;
  if (ir_value > g_ir_max) g_ir_max = (float)ir_value;

  if (g_max30102.checkForBeat((uint32_t)(g_max30102.getIR()))) {
    if (g_last_beat_ms > 0) {
      uint32_t interval = now - g_last_beat_ms;
      if (interval >= 300 && interval <= 2000) {  // plausibility gate: 30-200 bpm
        g_beat_interval_ms =
          0.7f * g_beat_interval_ms + 0.3f * (float)interval;  // EMA smoothing
        g_hr_bpm = 60000.0f / g_beat_interval_ms;

        float red_ac = g_red_max - g_red_min;
        float ir_ac = g_ir_max - g_ir_min;
        if (red_ac > 0 && ir_ac > 0 && ir_value > 0 && red_value > 0) {
          float r =
            ((float)red_value / red_ac) /
            ((float)ir_value / ir_ac);
          g_spo2_pct =
            constrain(110.0f - 25.0f * r, 70.0f, 100.0f);  // empirical calibration
        }
        g_red_min = 1e9f; g_red_max = -1e9f;
        g_ir_min = 1e9f; g_ir_max = -1e9f;
      }
    }
    g_last_beat_ms = now;
  }
}


// ================================================================
// ZERO FSR BASELINES
// ================================================================

static void zeroBaselines() {

  uint32_t acc[4] = {
    0, 0, 0, 0
  };

  const int N = 32;

  for (int k = 0; k < N; k++) {

    for (int i = 0; i < 4; i++) {

      acc[i] += analogRead(
        PIN_FSR[i]
      );
    }

    delay(5);
  }

  for (int i = 0; i < 4; i++) {

    g_fsr_baseline[i] =
      acc[i] / N;
  }

  Serial.println("FSR baselines calibrated.");
}


// ================================================================
// HEAD DETECTION
// ================================================================

static bool headDetected() {

  if (
    g_ir_top[0] ||
    g_ir_top[1]
  ) {

    return true;
  }

  return (
    !isnan(g_us_top_cm) &&
    g_us_top_cm > 0 &&
    g_us_top_cm < US_HEAD_DIST_CM
  );
}


// ================================================================
// BODY DETECTION
// ================================================================

static bool bodyDetected() {

  bool us_ok =
    isnan(g_us_bottom_cm) ||
    g_us_bottom_cm < US_BODY_DIST_CM;

  return (
    g_load_delta > OCC_LOAD_DELTA &&
    us_ok
  );
}


// ================================================================
// STATE CHANGE
// ================================================================

static void setState(KennelState s) {

  if (g_state == s) {
    return;
  }

  g_state = s;

  Serial.print("STATE -> ");
  Serial.println(STATE_NAMES[g_state]);

  if (s == ST_SNIFF) {

    g_sniff_started = millis();
  }
}


// ================================================================
// STATE MACHINE
// ================================================================

static void updateStateMachine(uint32_t now) {

  switch (g_state) {

    case ST_CALIBRATE:

      break;


    case ST_IDLE:

      if (bodyDetected()) {

        g_last_seen_ms = now;

        setState(ST_OCCUPIED);
      }

      break;


    case ST_OCCUPIED:

      if (!bodyDetected()) {

        if (
          now - g_last_seen_ms >
          LOAD_LOST_MS
        ) {

          setState(ST_IDLE);
        }

      } else {

        g_last_seen_ms = now;

        if (headDetected()) {

          if (g_head_since == 0) {

            g_head_since = now;
          }

          if (
            now - g_head_since >=
            HEAD_HOLD_MS
          ) {

            g_head_since = 0;

            setState(ST_SNIFF);
          }

        } else {

          g_head_since = 0;
        }
      }

      break;


    case ST_SNIFF:

      if (
        now - g_sniff_started >=
        SNIFF_WINDOW_MS
      ) {

        g_cooldown_until =
          now + COOLDOWN_MS;

        setState(ST_COOLDOWN);

      } else if (
        !bodyDetected() &&
        now - g_last_seen_ms >
        LOAD_LOST_MS
      ) {

        setState(ST_IDLE);
      }

      break;


    case ST_COOLDOWN:

      if (now >= g_cooldown_until) {

        setState(ST_IDLE);
      }

      break;


    default:

      setState(ST_IDLE);

      break;
  }
}


// ================================================================
// LED
// ================================================================

static void ledPattern(uint32_t now) {

  uint32_t on = 0;
  uint32_t off = 0;

  switch (g_state) {

    case ST_BOOT:
      on = 60;
      off = 60;
      break;

    case ST_CALIBRATE:
      on = 500;
      off = 0;
      break;

    case ST_IDLE:
      on = 40;
      off = 960;
      break;

    case ST_OCCUPIED:
      on = 120;
      off = 120;
      break;

    case ST_SNIFF:
      on = 30;
      off = 30;
      break;

    case ST_COOLDOWN:
      on = 500;
      off = 500;
      break;
  }

  uint32_t cycle =
    now % (on + off + 1);

  digitalWrite(
    PIN_LED,
    cycle < on ? HIGH : LOW
  );
}


// ================================================================
// JSON FRAME
// ================================================================

static size_t buildFrame(
  char* buf,
  size_t cap
) {

  snprintf(
    buf,
    cap,

    "{\"ts_ms\":%lu,\"seq\":%llu,\"state\":\"%s\","
    "\"fsr\":[%u,%u,%u,%u],"
    "\"ir\":[%d,%d,%d,%d,%d,%d],"
    "\"us\":{\"bottom\":%.1f,\"top\":%.1f},"
    "\"acc\":[%.3f,%.3f,%.3f],"
    "\"gyr\":[%.3f,%.3f,%.3f],"
    "\"imu_temp_c\":%.1f,"
    "\"hr_bpm\":%.1f,\"spo2_pct\":%.1f}",

    (unsigned long)millis(),

    (unsigned long long)g_seq,

    STATE_NAMES[g_state],

    g_fsr_raw[0],
    g_fsr_raw[1],
    g_fsr_raw[2],
    g_fsr_raw[3],

    (int)g_ir_bottom[0],
    (int)g_ir_bottom[1],
    (int)g_ir_bottom[2],
    (int)g_ir_bottom[3],

    (int)g_ir_top[0],
    (int)g_ir_top[1],

    (double)(
      isnan(g_us_bottom_cm)
        ? -1.0
        : g_us_bottom_cm
    ),

    (double)(
      isnan(g_us_top_cm)
        ? -1.0
        : g_us_top_cm
    ),

    (double)g_acc[0],
    (double)g_acc[1],
    (double)g_acc[2],

    (double)g_gyr[0],
    (double)g_gyr[1],
    (double)g_gyr[2],

    (double)(
      isnan(g_imu_temp_c)
        ? -1.0
        : g_imu_temp_c
    ),

    (double)g_hr_bpm,

    (double)g_spo2_pct
  );

  return strlen(buf);
}


// ================================================================
// BROADCAST FRAME
// ================================================================

static void broadcastFrame(
  const char* buf
) {

  for (auto& c : g_clients) {

    if (
      c &&
      c.connected()
    ) {

      c.println(buf);
    }
  }
}


// ================================================================
// ACCEPT TCP CLIENTS
// ================================================================

static void acceptClients() {

  while (true) {

    WiFiClient nc =
      g_server.available();

    if (!nc) {
      break;
    }

    bool placed = false;

    for (auto& c : g_clients) {

      if (
        !c ||
        !c.connected()
      ) {

        c = nc;

        placed = true;

        Serial.println(
          "TCP client connected."
        );

        break;
      }
    }

    if (!placed) {

      Serial.println(
        "TCP client rejected: slots full."
      );

      nc.stop();
    }
  }


  // ---------------------------------
  // Handle commands
  // ---------------------------------

  for (auto& c : g_clients) {

    if (
      !c ||
      !c.connected()
    ) {

      continue;
    }

    while (c.available() > 0) {

      char cmd =
        (char)c.read();

      // Re-zero FSR
      if (cmd == 'z') {

        Serial.println(
          "FSR re-zero requested."
        );

        zeroBaselines();
      }

      // Status
      else if (cmd == 's') {

        char buf[512];

        size_t n =
          buildFrame(
            buf,
            sizeof(buf)
          );

        c.write(
          (const uint8_t*)buf,
          n
        );

        c.println();
      }
    }
  }
}


// ================================================================
// WIFI
// ================================================================

static bool connectWiFi() {

  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "Connecting to WiFi..."
  );

  Serial.print(
    "SSID: "
  );

  Serial.println(
    WIFI_SSID
  );

  WiFi.mode(WIFI_STA);

  WiFi.setHostname(
    HOSTNAME
  );

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  uint32_t start =
    millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - start < 20000
  ) {

    delay(500);

    Serial.print(".");
  }

  Serial.println();

  if (
    WiFi.status() ==
    WL_CONNECTED
  ) {

    Serial.println(
      "WiFi CONNECTED!"
    );

    Serial.print(
      "IP address: "
    );

    Serial.println(
      WiFi.localIP()
    );

    Serial.print(
      "Gateway: "
    );

    Serial.println(
      WiFi.gatewayIP()
    );

    Serial.print(
      "RSSI: "
    );

    Serial.print(
      WiFi.RSSI()
    );

    Serial.println(
      " dBm"
    );

    Serial.print(
      "TCP server: "
    );

    Serial.print(
      WiFi.localIP()
    );

    Serial.print(
      ":"
    );

    Serial.println(
      TCP_PORT
    );

    Serial.println(
      "================================"
    );

    return true;
  }

  Serial.println(
    "WiFi CONNECTION FAILED!"
  );

  Serial.print(
    "WiFi status code: "
  );

  Serial.println(
    WiFi.status()
  );

  Serial.println(
    "================================"
  );

  return false;
}


// ================================================================
// SETUP
// ================================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println();
  Serial.println(
    "================================"
  );
  Serial.println(
    "QMLKit Kennel Node"
  );
  Serial.println(
    "ESP32-WROOM-32"
  );
  Serial.println(
    "================================"
  );


  // ---------------------------------
  // GPIO
  // ---------------------------------

  pinMode(
    PIN_LED,
    OUTPUT
  );

  for (int p : PIN_FSR) {

    pinMode(
      p,
      INPUT
    );
  }

  for (int p : PIN_IR_BOTTOM) {

    pinMode(
      p,
      INPUT_PULLUP
    );
  }

  for (int p : PIN_IR_TOP) {

    pinMode(
      p,
      INPUT_PULLUP
    );
  }

  pinMode(
    PIN_US_BOT_TRIG,
    OUTPUT
  );

  pinMode(
    PIN_US_BOT_ECHO,
    INPUT
  );

  pinMode(
    PIN_US_TOP_TRIG,
    OUTPUT
  );

  pinMode(
    PIN_US_TOP_ECHO,
    INPUT
  );

  Serial.println(
    "GPIO initialized."
  );


  // ---------------------------------
  // I2C
  // ---------------------------------

  Wire.begin(
    PIN_I2C_SDA,
    PIN_I2C_SCL
  );

  Wire.setClock(400000);

  Serial.println(
    "I2C initialized."
  );


  // ---------------------------------
  // MPU6050
  // ---------------------------------

  Serial.println(
    "Initializing MPU6050..."
  );

  g_mpu_ok =
    mpu.begin();

  if (g_mpu_ok) {

    Serial.println(
      "MPU6050 detected."
    );

    mpu.setAccelerometerRange(
      MPU6050_RANGE_8_G
    );

    mpu.setGyroRange(
      MPU6050_RANGE_500_DEG
    );

    mpu.setFilterBandwidth(
      MPU6050_BAND_44_HZ
    );

  } else {

    Serial.println(
      "WARNING: MPU6050 NOT detected."
    );
  }


  // ---------------------------------
  // MAX30102 heart-rate / SpO2
  // ---------------------------------

  Serial.println(
    "Initializing MAX30102..."
  );

  g_hr_ok =
    g_max30102.begin(Wire, I2C_SPEED_FAST);

  if (g_hr_ok) {

    Serial.println(
      "MAX30102 detected."
    );

    g_max30102.setup(
      60,   // LED brightness
      4,    // sample average
      2,    // led mode (SpO2)
      100,  // sample rate
      411,  // pulse width
      4096  // adc range
    );
  } else {

    Serial.println(
      "WARNING: MAX30102 NOT detected - hr_bpm/spo2_pct will be -1."
    );
  }


  // ---------------------------------
  // WiFi
  // ---------------------------------

  bool wifi_ok =
    connectWiFi();

  if (!wifi_ok) {

    Serial.println(
      "Continuing without WiFi."
    );
  }


  // ---------------------------------
  // mDNS
  // ---------------------------------

  if (wifi_ok) {

    if (MDNS.begin(HOSTNAME)) {

      MDNS.addService(
        "qmlkit-kennel",
        "tcp",
        TCP_PORT
      );

      Serial.println(
        "mDNS started."
      );

      Serial.print(
        "Hostname: "
      );

      Serial.print(
        HOSTNAME
      );

      Serial.println(
        ".local"
      );

    } else {

      Serial.println(
        "mDNS failed."
      );
    }
  }


  // ---------------------------------
  // TCP server
  // ---------------------------------

  g_server.begin();

  g_server.setNoDelay(true);

  Serial.print(
    "TCP server listening on port "
  );

  Serial.println(
    TCP_PORT
  );


  // ---------------------------------
  // Calibration
  // ---------------------------------

  setState(
    ST_CALIBRATE
  );

  Serial.println(
    "Calibrating FSR baselines..."
  );

  zeroBaselines();

  setState(
    ST_IDLE
  );


  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "SETUP COMPLETE"
  );

  Serial.println(
    "================================"
  );
}


// ================================================================
// LOOP
// ================================================================

void loop() {

  const uint32_t now =
    millis();


  // ---------------------------------
  // MPU6050 - ~100 Hz
  // ---------------------------------

  if (
    g_mpu_ok &&
    now >= g_next_imu_ms
  ) {

    sensors_event_t a;
    sensors_event_t gv;
    sensors_event_t temp;

    mpu.getEvent(
      &a,
      &gv,
      &temp
    );

    g_acc[0] =
      a.acceleration.x;

    g_acc[1] =
      a.acceleration.y;

    g_acc[2] =
      a.acceleration.z;


    g_gyr[0] =
      gv.gyro.x;

    g_gyr[1] =
      gv.gyro.y;

    g_gyr[2] =
      gv.gyro.z;


    g_imu_temp_c =
      temp.temperature;


    g_next_imu_ms =
      now + IMU_INTERVAL_MS;
  }


  // ---------------------------------
  // Slow sensors
  // ---------------------------------

  if (
    now >= g_next_slow_ms
  ) {

    readSlowChannels();

    updateStateMachine(
      now
    );

    g_next_slow_ms =
      now + SLOW_INTERVAL_MS;
  }


  // ---------------------------------
  // Physiology (heart-rate / SpO2)
  // ---------------------------------

  if (
    now >= g_next_hr_ms
  ) {

    readPhysiology(now);

    g_next_hr_ms =
      now + HR_INTERVAL_MS;
  }


  // ---------------------------------
  // TCP clients
  // ---------------------------------

  acceptClients();


  // ---------------------------------
  // LED
  // ---------------------------------

  ledPattern(now);


  // ---------------------------------
  // Transmit
  // ---------------------------------

  if (
    now >= g_next_tx_ms
  ) {

    char buf[512];

    buildFrame(
      buf,
      sizeof(buf)
    );

    broadcastFrame(
      buf
    );

    g_seq++;


    // Serial output every 2 sec
    if (
      now >= g_next_serial_ms
    ) {

      Serial.println(buf);

      g_next_serial_ms =
        now + 2000;
    }


    g_next_tx_ms =
      now + TX_INTERVAL_MS;
  }
}