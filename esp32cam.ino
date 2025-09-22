// ESP8266 Wemos D1 mini - Logic Analyzer like stream on D4 (GPIO2)
// Samples D4 at a fixed rate using hardware timer1 and streams packed bits over Serial.
// Each byte contains 8 consecutive samples (MSB first).

#include <Arduino.h>

#define SAMPLE_PIN        D5          // ثابت: D3
#define SAMPLE_GPIO       2           // GPIO number for D4
#define SAMPLE_RATE_HZ    100000      // نرخ نمونه‌برداری (تغییرپذیر). 100 kHz پیش‌فرض
#define SERIAL_BAUD       921600      // برای خروجی سریع
#define RB_SIZE           8192        // سایز بافر حلقوی (به بایت، باید توانِ ۲ باشد)

static volatile uint8_t  ringbuf[RB_SIZE];
static volatile uint16_t rb_head = 0, rb_tail = 0;

static volatile uint8_t  pack_byte = 0;
static volatile uint8_t  pack_bits = 0;
static volatile bool     overflowed = false;

extern "C" {
  #include "gpio.h" // برای خواندن خیلی سریع رجیستر ورودی
}

static inline uint32_t IRAM_ATTR fastReadD4() {
  // D4 = GPIO2 -> بیت 2 از رجیستر ورودی
  return (GPIO_REG_READ(GPIO_IN_ADDRESS) >> SAMPLE_GPIO) & 0x1;
}

void IRAM_ATTR timerISR() {
  // خواندن سریع سطح پایه
  uint32_t bit = fastReadD4();

  // فشرده‌سازی: 8 نمونه -> 1 بایت (MSB first)
  pack_byte = (uint8_t)((pack_byte << 1) | (bit & 0x1));
  pack_bits++;

  if (pack_bits == 8) {
    uint16_t next_head = (rb_head + 1) & (RB_SIZE - 1);
    if (next_head == rb_tail) {
      overflowed = true; // بافر پر شد؛ این بایت را دور می‌ریزیم
    } else {
      ringbuf[rb_head] = pack_byte;
      rb_head = next_head;
    }
    pack_bits = 0;
    pack_byte = 0;
  }
}

void setupTimer(uint32_t rate_hz) {
  // Timer1 tick = F_CPU / div. با TIM_DIV16 روی 80MHz => 5MHz
  uint32_t ticks = (F_CPU / 16) / rate_hz; // 80e6/16=5e6 => ticks=5e6/rate
  if (ticks < 5) ticks = 5;                // محدودیت امن حداقل تیک

  noInterrupts();
  timer1_isr_init();
  timer1_attachInterrupt(timerISR);
  // EDGE + LOOP: تریگر لبه و تکرارشونده
  timer1_enable(TIM_DIV16, TIM_EDGE, TIM_LOOP);
  timer1_write(ticks);
  interrupts();
}

void setup() {
  pinMode(SAMPLE_PIN, INPUT); // اگر خروجی کلکتور-باز دارید: INPUT_PULLUP
  Serial.begin(SERIAL_BAUD);
  delay(50);

  // هِدِر اولیه برای هماهنگی سمت PC
  Serial.println(F("=== ESP8266 D4 Sampler ==="));
  Serial.printf("Rate: %u Hz, Pin: D4(GPIO2), Packed: 8 samples/byte, Baud: %u\n",
                (unsigned)SAMPLE_RATE_HZ, (unsigned)SERIAL_BAUD);
  Serial.println(F("Stream starts... (binary bytes follow)"));

  setupTimer(SAMPLE_RATE_HZ);
}

void loop() {
  // تخلیه بافر به سریال بدون مسدود کردن وقفه‌ها طولانی
  // 1) گرفتن snapshot از head
  uint16_t h;
  noInterrupts();
  h = rb_head;
  interrupts();

  while (rb_tail != h) {
    // ارسال یک بخش پیوسته
    uint16_t contiguous = (h >= rb_tail) ? (h - rb_tail) : (RB_SIZE - rb_tail);
    // حداکثر chunk منطقی برای write (جلوگیری از بلوک شدن طولانی)
    if (contiguous > 256) contiguous = 256;
    size_t wrote = Serial.write((const uint8_t*)&ringbuf[rb_tail], contiguous);
    rb_tail = (rb_tail + wrote) & (RB_SIZE - 1);

    // اگر سریال هم‌قدم نبود، از حلقه خارج شو تا وقفه‌ها فرصت داشته باشند
    if (wrote < contiguous) break;

    // snapshot جدید
    noInterrupts();
    h = rb_head;
    interrupts();
  }

  // هر چند وقت یک‌بار پرچم overflow را گزارش کن (متنی کوتاه، آسیب به استریم نمی‌زند)
  static uint32_t lastMsgMs = 0;
  uint32_t now = millis();
  if (overflowed && now - lastMsgMs > 1000) {
    lastMsgMs = now;
    overflowed = false;
    Serial.println(F("\n[WARN] Buffer overflow: increase baud or reduce sample rate.\n"));
  }
}
