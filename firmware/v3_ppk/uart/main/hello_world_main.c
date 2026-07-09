#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/uart.h"

#include "sdkconfig.h"
#include "esp_cpu.h"
#include "esp_timer.h"

/* ---------------- Experiment settings ---------------- */

#define SETTLING_IDLE_MS        3000
#define SYNC_PULSE_MS           1000
#define RECOVERY_IDLE_MS        5000

#define INITIAL_IDLE_MS         10000
#define FINAL_IDLE_MS           10000

#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

#define WORKLOAD_CORE           1

/* UART workload settings */
#define UART_WORKLOAD_NUM       UART_NUM_1
#define UART_TX_GPIO            17
#define UART_RX_GPIO            UART_PIN_NO_CHANGE

#define UART_BAUD_RATE          115200
#define UART_TX_BYTES           512      // change to 64, 256, or 512
#define UART_PERIOD_MS          100
#define UART_ACTIVE_TX_COUNT    200
#define ACTIVE_DURATION_MS      (UART_ACTIVE_TX_COUNT * UART_PERIOD_MS)

#define UART_TX_BUFFER_SIZE     2048
#define UART_RX_BUFFER_SIZE     1024

/* Phase logs:
   1 = print phase logs for debugging
   0 = no phase logs for official measurement
*/
#define ENABLE_PHASE_LOGS       0

/* GPIO marker:
   1 = use marker for timing check
   0 = no marker for official measurement
*/
#define USE_MARKER              0
#define MARKER_GPIO             25

/* 0 = idle, 1 = UART active */
static volatile int g_workload_state = 0;
static volatile bool g_experiment_running = true;

/* Counters / anti-optimization */
static volatile uint32_t g_workload_counter = 0;
static volatile uint32_t g_sync_result = 1;

/* Fixed UART buffer */
static uint8_t g_uart_tx_buffer[UART_TX_BYTES];

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

    gpio_config(&io_conf);
    gpio_set_level(MARKER_GPIO, 0);
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
    uint32_t target_cycles =
        (uint32_t)(((uint64_t)CPU_FREQ_HZ * busy_us) / 1000000ULL);

    uint32_t start_cycles = esp_cpu_get_cycle_count();

    while ((uint32_t)(esp_cpu_get_cycle_count() - start_cycles) < target_cycles) {
        do_sync_workload_operation();
    }
}

static void run_sync_cpu_pulse_ms(uint32_t duration_ms)
{
    int64_t start_us = esp_timer_get_time();
    int64_t duration_us = (int64_t)duration_ms * 1000;

    while ((esp_timer_get_time() - start_us) < duration_us) {
        /*
           Run short busy chunks instead of one long blocking loop.
           This creates a visible CPU-current pulse while reducing watchdog risk.
        */
        run_busy_ccount_us(1000);
        taskYIELD();
    }
}

/* ---------------- UART workload init ---------------- */

static void uart_workload_init(void)
{
    uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    uart_driver_install(
        UART_WORKLOAD_NUM,
        UART_RX_BUFFER_SIZE,
        UART_TX_BUFFER_SIZE,
        0,
        NULL,
        0
    );

    uart_param_config(UART_WORKLOAD_NUM, &uart_config);

    uart_set_pin(
        UART_WORKLOAD_NUM,
        UART_TX_GPIO,
        UART_RX_GPIO,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE
    );

    /* Fill fixed buffer with deterministic data */
    for (int i = 0; i < UART_TX_BYTES; i++) {
        g_uart_tx_buffer[i] = (uint8_t)('A' + (i % 26));
    }
}

/* ---------------- UART workload task on Core 1 ---------------- */

static void uart_workload_task(void *parameter)
{
    (void)parameter;

    while (g_experiment_running) {

        if (g_workload_state == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        /*
           UART active phase:
           Send exactly UART_ACTIVE_TX_COUNT times.
        */
        for (int i = 0; i < UART_ACTIVE_TX_COUNT; i++) {

            if (!g_experiment_running || g_workload_state == 0) {
                break;
            }

            int64_t cycle_start_us = esp_timer_get_time();

            int written = uart_write_bytes(
                UART_WORKLOAD_NUM,
                (const char *)g_uart_tx_buffer,
                UART_TX_BYTES
            );

            if (written > 0) {
                g_workload_counter += (uint32_t)written;
            }

            uart_wait_tx_done(UART_WORKLOAD_NUM, pdMS_TO_TICKS(1000));

            int64_t elapsed_us = esp_timer_get_time() - cycle_start_us;
            int64_t period_us = (int64_t)UART_PERIOD_MS * 1000;

            if (elapsed_us < period_us) {
                int64_t remaining_us = period_us - elapsed_us;
                vTaskDelay(pdMS_TO_TICKS((uint32_t)(remaining_us / 1000)));
            } else {
                taskYIELD();
            }
        }

        /*
           After exactly 200 transmissions, stay idle until app_main changes phase.
        */
        while (g_experiment_running && g_workload_state == 1) {
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }

    vTaskDelete(NULL);
}

/* ---------------- Main experiment ---------------- */

void app_main(void)
{
    uint32_t cpu_freq_hz = CPU_FREQ_HZ;

#if ENABLE_PHASE_LOGS
    printf("# experiment=ppk2_uart_tx_only_with_cpu_sync_pulse\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n", (unsigned long)EXPECTED_CPU_FREQ_HZ);
    printf("# workload_core=%d\n", WORKLOAD_CORE);
    printf("# uart_num=%d\n", UART_WORKLOAD_NUM);
    printf("# uart_tx_gpio=%d\n", UART_TX_GPIO);
    printf("# uart_baud_rate=%d\n", UART_BAUD_RATE);
    printf("# uart_format=8N1\n");
    printf("# uart_tx_bytes=%d\n", UART_TX_BYTES);
    printf("# uart_period_ms=%d\n", UART_PERIOD_MS);
    printf("# uart_active_tx_count=%d\n", UART_ACTIVE_TX_COUNT);
    printf("# settling_idle_ms=%d\n", SETTLING_IDLE_MS);
    printf("# sync_pulse_ms=%d\n", SYNC_PULSE_MS);
    printf("# recovery_idle_ms=%d\n", RECOVERY_IDLE_MS);
    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# active_duration_ms=%d\n", ACTIVE_DURATION_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);
    printf("# use_marker=%d\n", USE_MARKER);
    printf("# marker_gpio=%d\n", MARKER_GPIO);
#endif

    if (cpu_freq_hz != EXPECTED_CPU_FREQ_HZ) {
#if ENABLE_PHASE_LOGS
        printf("# WARNING: CPU frequency is not 240MHz\n");
#endif
    }

    marker_init();
    uart_workload_init();

    g_workload_state = 0;
    g_experiment_running = true;

    BaseType_t workload_created =
        xTaskCreatePinnedToCore(
            uart_workload_task,
            "uart_workload_task",
            4096,
            NULL,
            5,
            NULL,
            WORKLOAD_CORE
        );

    if (workload_created != pdPASS) {
#if ENABLE_PHASE_LOGS
        printf("# workload_task_creation_failed\n");
#endif
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(1000));

#if ENABLE_PHASE_LOGS
    printf("# phase=settling_idle\n");
#endif
    g_workload_state = 0;
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(SETTLING_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=sync_cpu_pulse\n");
#endif
    marker_set(1);
    run_sync_cpu_pulse_ms(SYNC_PULSE_MS);
    marker_set(0);

#if ENABLE_PHASE_LOGS
    printf("# phase=recovery_idle\n");
#endif
    g_workload_state = 0;
    vTaskDelay(pdMS_TO_TICKS(RECOVERY_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=initial_idle\n");
#endif
    g_workload_state = 0;
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=uart_active\n");
#endif
    g_workload_state = 1;
    marker_set(1);
    vTaskDelay(pdMS_TO_TICKS(ACTIVE_DURATION_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=final_idle\n");
#endif
    g_workload_state = 0;
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    g_experiment_running = false;

    vTaskDelay(pdMS_TO_TICKS(1000));

#if ENABLE_PHASE_LOGS
    printf("# experiment_complete\n");
    printf("# uart_total_bytes_written=%lu\n", (unsigned long)g_workload_counter);
    printf("# sync_result=%lu\n", (unsigned long)g_sync_result);
#endif
}