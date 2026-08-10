#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/adc.h"
#include "driver/gpio.h"

#include "esp_err.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "sdkconfig.h"

/* ---------------- Experiment settings ---------------- */

#define SETTLING_IDLE_MS        3000

#define INITIAL_IDLE_MS         10000
#define TEST_WINDOW_MS          20000
#define FINAL_IDLE_MS           10000

#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ \
    ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

#define WORKLOAD_CORE           1

/* Check time only once per this many ADC reads */
#define ADC_READS_PER_TIME_CHECK 256

/* ADC configuration */
#define ADC_CHANNEL             ADC1_CHANNEL_6   /* GPIO34 */
#define ADC_WIDTH               ADC_WIDTH_BIT_12
#define ADC_ATTENUATION         ADC_ATTEN_DB_11

#define ENABLE_PHASE_LOGS       0
#define USE_MARKER              0
#define MARKER_GPIO             25

static const char *TAG =
    "adc_continuous_read_preconditioned_no_sync";

static volatile int g_adc_value = 0;
static volatile uint64_t g_adc_read_count = 0;

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
    ESP_ERROR_CHECK(gpio_set_level(MARKER_GPIO, level));
#else
    (void)level;
#endif
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

static void adc_precondition(void)
{
    /*
     * One ADC conversion is executed before all measured phases.
     * This moves the first-read state transition outside the
     * initial-idle / active / final-idle sequence.
     */
    g_adc_value = adc1_get_raw(ADC_CHANNEL);
}

/* ---------------- Continuous ADC workload ---------------- */

static void run_continuous_adc_read_ms(uint32_t duration_ms)
{
    const int64_t start_us = esp_timer_get_time();
    const int64_t duration_us = (int64_t)duration_ms * 1000;

    uint64_t read_count = 0;
    int last_value = 0;

    while (true) {
        for (uint32_t i = 0;
             i < ADC_READS_PER_TIME_CHECK;
             i++) {
            last_value = adc1_get_raw(ADC_CHANNEL);
        }

        read_count += ADC_READS_PER_TIME_CHECK;

        if ((esp_timer_get_time() - start_us) >= duration_us) {
            break;
        }
    }

    g_adc_value = last_value;
    g_adc_read_count = read_count;
}

/* ---------------- Core 1 experiment task ---------------- */

static void adc_experiment_task(void *arg)
{
    (void)arg;

#if ENABLE_PHASE_LOGS
    printf("# phase=settling_idle_adc_preconditioned\n");
#endif

    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(SETTLING_IDLE_MS));

    /*
     * CPU sync pulse and recovery idle were intentionally removed
     * so that the ADC transition is the first rising transition.
     */

#if ENABLE_PHASE_LOGS
    printf(
        "# phase=initial_idle_adc_preconditioned_no_sampling\n"
    );
#endif

    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=test_window_continuous_adc_read\n");
#endif

    marker_set(1);

    const int64_t active_start_us = esp_timer_get_time();

    run_continuous_adc_read_ms(TEST_WINDOW_MS);

    const int64_t active_elapsed_us =
        esp_timer_get_time() - active_start_us;

    marker_set(0);

#if ENABLE_PHASE_LOGS
    printf(
        "# phase=final_idle_adc_preconditioned_no_sampling\n"
    );
#endif

    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

#if ENABLE_PHASE_LOGS
    const double active_elapsed_s =
        (double)active_elapsed_us / 1000000.0;

    const double average_adc_reads_per_second =
        active_elapsed_s > 0.0
            ? (double)g_adc_read_count / active_elapsed_s
            : 0.0;

    printf("# experiment_complete\n");
    printf("# adc_value=%d\n", g_adc_value);
    printf(
        "# adc_read_count=%llu\n",
        (unsigned long long)g_adc_read_count
    );
    printf(
        "# average_adc_reads_per_second=%.2f\n",
        average_adc_reads_per_second
    );
#endif

    vTaskDelete(NULL);
}

/* ---------------- Main ---------------- */

void app_main(void)
{
    const uint32_t cpu_freq_hz = CPU_FREQ_HZ;

#if ENABLE_PHASE_LOGS
    printf(
        "# experiment="
        "oscilloscope_adc_continuous_read_preconditioned_no_sync\n"
    );
    printf("# measurement_device=oscilloscope\n");
    printf("# internal_current_sensor=disabled\n");

    printf(
        "# cpu_freq_hz=%lu\n",
        (unsigned long)cpu_freq_hz
    );
    printf(
        "# expected_cpu_freq_hz=%lu\n",
        (unsigned long)EXPECTED_CPU_FREQ_HZ
    );
    printf("# workload_core=%d\n", WORKLOAD_CORE);

    printf("# adc_unit=ADC1\n");
    printf("# adc_gpio=34\n");
    printf("# adc_channel=ADC1_CH6\n");
    printf("# adc_resolution_bits=12\n");
    printf("# adc_attenuation=11dB\n");
    printf("# adc_access_method=cpu_driven_no_dma\n");
    printf("# adc_workload=continuous_read\n");
    printf(
        "# adc_preconditioning="
        "one_dummy_read_before_measured_phases\n"
    );
    printf("# adc_read_delay=none\n");
    printf(
        "# adc_reads_per_time_check=%d\n",
        ADC_READS_PER_TIME_CHECK
    );

    printf("# cpu_sync_pulse=disabled\n");
    printf("# settling_idle_ms=%d\n", SETTLING_IDLE_MS);
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

    /*
     * The dummy read occurs before the task and before the measured
     * phases, preventing the first ADC read from contaminating the
     * initial-idle-to-active transition.
     */
    adc_precondition();

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