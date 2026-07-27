#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/adc.h"
#include "driver/gpio.h"

#include "esp_cpu.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "sdkconfig.h"

/* ---------------- Experiment settings ---------------- */

#define SETTLING_IDLE_MS        3000
#define SYNC_PULSE_MS           1000
#define RECOVERY_IDLE_MS        5000

#define INITIAL_IDLE_MS         10000
#define TEST_WINDOW_MS          20000
#define FINAL_IDLE_MS           10000

#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ \
    ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

#define WORKLOAD_CORE           1

/* Burst condition:
   Change this for each condition:
   10   = every 100 ms read 10 samples
   100  = every 100 ms read 100 samples
   1000 = every 100 ms read 1000 samples
*/
#define ADC_BURST_PERIOD_MS     100
#define ADC_BURST_SAMPLES       1000

/* ADC configuration */
#define ADC_CHANNEL             ADC1_CHANNEL_6   /* GPIO34 */
#define ADC_WIDTH               ADC_WIDTH_BIT_12
#define ADC_ATTENUATION         ADC_ATTEN_DB_11

#define ENABLE_PHASE_LOGS       0
#define USE_MARKER              0
#define MARKER_GPIO             25

static const char *TAG = "adc_periodic_burst";

static volatile uint32_t g_sync_result = 1;
static volatile int g_adc_value = 0;

/* ---------------- GPIO marker ---------------- */

static void marker_init(void)
{
#if USE_MARKER
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << MARKER_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    ESP_ERROR_CHECK(gpio_config(&io_conf));
    ESP_ERROR_CHECK(gpio_set_level(MARKER_GPIO, 0));
#endif
}

static inline void marker_set(int level)
{
#if USE_MARKER
    gpio_set_level(MARKER_GPIO, level);
#else
    (void)level;
#endif
}

/* ---------------- CPU sync pulse workload ---------------- */

static inline void do_sync_workload_operation(void)
{
    g_sync_result =
        (g_sync_result * 1664525U) + 1013904223U;

    g_sync_result ^= g_sync_result >> 13;
    g_sync_result *= 2654435761U;
}

static void run_busy_ccount_us(uint32_t busy_us)
{
    const uint32_t target_cycles =
        (uint32_t)(((uint64_t)CPU_FREQ_HZ * busy_us) /
                   1000000ULL);

    const uint32_t start_cycles =
        esp_cpu_get_cycle_count();

    while ((uint32_t)(
               esp_cpu_get_cycle_count() - start_cycles
           ) < target_cycles) {
        do_sync_workload_operation();
    }
}

static void run_sync_cpu_pulse_ms(uint32_t duration_ms)
{
    const int64_t start_us = esp_timer_get_time();
    const int64_t duration_us =
        (int64_t)duration_ms * 1000;

    while ((esp_timer_get_time() - start_us) < duration_us) {
        run_busy_ccount_us(1000);
        taskYIELD();
    }
}

/* ---------------- System setup ---------------- */

static void disable_wifi_for_measurement(void)
{
    esp_err_t err;

    err = esp_wifi_stop();
    if (err != ESP_OK &&
        err != ESP_ERR_WIFI_NOT_INIT) {
        ESP_LOGW(TAG, "esp_wifi_stop: %s", esp_err_to_name(err));
    }

    err = esp_wifi_deinit();
    if (err != ESP_OK &&
        err != ESP_ERR_WIFI_NOT_INIT) {
        ESP_LOGW(TAG, "esp_wifi_deinit: %s", esp_err_to_name(err));
    }
}

static void adc_init(void)
{
    ESP_ERROR_CHECK(adc1_config_width(ADC_WIDTH));

    ESP_ERROR_CHECK(
        adc1_config_channel_atten(
            ADC_CHANNEL,
            ADC_ATTENUATION
        )
    );
}

/* ---------------- ADC periodic burst workload ---------------- */

static void run_adc_burst_once(int n_samples)
{
    int last_value = 0;

    for (int i = 0; i < n_samples; i++) {
        last_value = adc1_get_raw(ADC_CHANNEL);
    }

    g_adc_value = last_value;
}

static void run_periodic_burst_adc_sampling_ms(
    uint32_t period_ms,
    int burst_samples,
    uint32_t duration_ms
)
{
    TickType_t last_wake_time = xTaskGetTickCount();
    const TickType_t period_ticks = pdMS_TO_TICKS(period_ms);

    const int64_t start_us = esp_timer_get_time();
    const int64_t duration_us = (int64_t)duration_ms * 1000;

    while ((esp_timer_get_time() - start_us) < duration_us) {
        run_adc_burst_once(burst_samples);

        vTaskDelayUntil(&last_wake_time, period_ticks);
    }
}

/* ---------------- Core1 experiment task ---------------- */

static void adc_experiment_task(void *arg)
{
    (void)arg;

#if ENABLE_PHASE_LOGS
    printf("# phase=settling_idle_adc_initialized\n");
#endif
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(SETTLING_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=sync_cpu_pulse\n");
#endif
    marker_set(1);
    run_sync_cpu_pulse_ms(SYNC_PULSE_MS);
    marker_set(0);

#if ENABLE_PHASE_LOGS
    printf("# phase=recovery_idle_adc_initialized\n");
#endif
    vTaskDelay(pdMS_TO_TICKS(RECOVERY_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=initial_idle_adc_initialized_no_sampling\n");
#endif
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=test_window_periodic_burst_adc_sampling\n");
#endif
    marker_set(1);
    run_periodic_burst_adc_sampling_ms(
        ADC_BURST_PERIOD_MS,
        ADC_BURST_SAMPLES,
        TEST_WINDOW_MS
    );
    marker_set(0);

#if ENABLE_PHASE_LOGS
    printf("# phase=final_idle_adc_initialized_no_sampling\n");
#endif
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    vTaskDelay(pdMS_TO_TICKS(1000));

#if ENABLE_PHASE_LOGS
    printf("# experiment_complete\n");
    printf("# sync_result=%lu\n", (unsigned long)g_sync_result);
    printf("# adc_value=%d\n", g_adc_value);
#endif

    vTaskDelete(NULL);
}

/* ---------------- Main ---------------- */

void app_main(void)
{
    const uint32_t cpu_freq_hz = CPU_FREQ_HZ;

#if ENABLE_PHASE_LOGS
    printf("# experiment=ppk2_adc_periodic_burst_sampling\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n",
           (unsigned long)EXPECTED_CPU_FREQ_HZ);
    printf("# workload_core=%d\n", WORKLOAD_CORE);

    printf("# adc_unit=ADC1\n");
    printf("# adc_gpio=34\n");
    printf("# adc_channel=ADC1_CH6\n");
    printf("# adc_resolution_bits=12\n");
    printf("# adc_attenuation=11dB\n");
    printf("# adc_access_method=cpu_driven_no_dma\n");
    printf("# adc_workload=periodic_burst_sampling\n");
    printf("# adc_burst_period_ms=%d\n", ADC_BURST_PERIOD_MS);
    printf("# adc_burst_samples=%d\n", ADC_BURST_SAMPLES);

    printf("# settling_idle_ms=%d\n", SETTLING_IDLE_MS);
    printf("# sync_pulse_ms=%d\n", SYNC_PULSE_MS);
    printf("# recovery_idle_ms=%d\n", RECOVERY_IDLE_MS);
    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# test_window_ms=%d\n", TEST_WINDOW_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);
#endif

    if (cpu_freq_hz != EXPECTED_CPU_FREQ_HZ) {
#if ENABLE_PHASE_LOGS
        printf("# WARNING: CPU frequency is not 240MHz\n");
#endif
    }

    marker_init();
    disable_wifi_for_measurement();

    ESP_ERROR_CHECK(
        esp_sleep_disable_wakeup_source(
            ESP_SLEEP_WAKEUP_ALL
        )
    );

    adc_init();

    xTaskCreatePinnedToCore(
        adc_experiment_task,
        "adc_experiment_task",
        4096,
        NULL,
        5,
        NULL,
        WORKLOAD_CORE
    );

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}