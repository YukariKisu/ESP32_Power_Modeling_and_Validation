#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"

#include "sdkconfig.h"

/* ---------------- Experiment settings ---------------- */

#define INITIAL_IDLE_MS         10000
#define ACTIVE_DURATION_MS      20000
#define FINAL_IDLE_MS           10000


#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

#define WORKLOAD_CORE           1

/* GPIO marker:
   1 = use marker for timing check
   0 = no marker for official measurement
*/
#define USE_MARKER              0
#define MARKER_GPIO             25


/* 0 = idle, 1 = active */
static volatile int g_workload_state = 0;
static volatile bool g_experiment_running = true;


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


/* ---------------- CCOUNT workload control ---------------- */

static inline void do_workload_operation(void)
{
    asm volatile (
        "nop\n"
        "nop\n"
        "nop\n"
        "nop\n"
        "nop\n"
        "nop\n"
        "nop\n"
        "nop\n"
        ::: "memory"
    );
}

static void run_busy_ccount_us(uint32_t busy_us)
{
    uint32_t target_cycles =
        (uint32_t)(((uint64_t)CPU_FREQ_HZ * busy_us) / 1000000ULL);

    uint32_t start_cycles = esp_cpu_get_cycle_count();

    while ((uint32_t)(esp_cpu_get_cycle_count() - start_cycles) < target_cycles) {
        do_workload_operation();
    }
}

/* ---------------- Workload task on Core 1 ---------------- */

static void cpu_workload_task(void *parameter)
{
    (void)parameter;

    while (g_experiment_running) {

        if (g_workload_state == 0) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        while (g_experiment_running && g_workload_state == 1) {
            asm volatile ("nop");
        }
    }

    vTaskDelete(NULL);
}

/* ---------------- Main experiment ---------------- */

void app_main(void)
{
    uint32_t cpu_freq_hz = CPU_FREQ_HZ;

    printf("# experiment=ppk2_cpu_only_minimum_nop_100\n");
    printf("# sleep_mode=disabled\n");
    printf("# peripherals=disabled\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# workload_definition=minimum_core_only_workload\n");
    printf("# workload_components=nop_loop\n");
    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n", (unsigned long)EXPECTED_CPU_FREQ_HZ);
    printf("# workload_core=%d\n", WORKLOAD_CORE);
    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# active_duration_ms=%d\n", ACTIVE_DURATION_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);
    printf("# use_marker=%d\n", USE_MARKER);
    printf("# marker_gpio=%d\n", MARKER_GPIO);
    printf("# workload_mode=simple_continuous_nop_loop\n");
    printf("# ccount_control=disabled\n");

    if (cpu_freq_hz != EXPECTED_CPU_FREQ_HZ) {
        printf("# WARNING: CPU frequency is not 240MHz\n");
    }

    marker_init();

    g_workload_state = 0;
    g_experiment_running = true;

    BaseType_t workload_created =
        xTaskCreatePinnedToCore(
            cpu_workload_task,
            "cpu_workload_task",
            4096,
            NULL,
            5,
            NULL,
            WORKLOAD_CORE
        );

    if (workload_created != pdPASS) {
        printf("# workload_task_creation_failed\n");
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# phase=initial_idle\n");
    g_workload_state = 0;
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

    printf("# phase=active\n");
    g_workload_state = 1;
    marker_set(1);
    vTaskDelay(pdMS_TO_TICKS(ACTIVE_DURATION_MS));

    printf("# phase=final_idle\n");
    g_workload_state = 0;
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    g_experiment_running = false;

    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# experiment_complete\n");
}