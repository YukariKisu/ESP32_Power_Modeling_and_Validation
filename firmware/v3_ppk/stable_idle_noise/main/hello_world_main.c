#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"

#include "sdkconfig.h"
#include "esp_cpu.h"

/* ---------------- Experiment settings ---------------- */

#define STABLE_IDLE_DURATION_MS 60000

#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

/*
   0 = no workload task
       This measures a simple ESP-IDF / FreeRTOS idle-only baseline.

   1 = create workload task, but keep it inactive
       This is closer to the idle phase of the normal experiment firmware.
*/
#define CREATE_INACTIVE_WORKLOAD_TASK  0

#define WORKLOAD_CORE           1

/* GPIO marker:
   1 = use marker for timing check
   0 = no marker for official PPK measurement
*/
#define USE_MARKER              0
#define MARKER_GPIO             25

/* 0 = idle, 1 = active */
static volatile int g_workload_state = 0;
static volatile bool g_experiment_running = true;

/* Prevent optimization */
static volatile uint32_t g_workload_result = 1;


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


/* ---------------- Optional inactive workload task ---------------- */

static inline void do_minimal_operation(void)
{
    /*
       This function is not used for active workload here.
       It only exists to keep the structure similar to the main firmware
       if CREATE_INACTIVE_WORKLOAD_TASK is enabled.
    */
    g_workload_result =
        (g_workload_result * 1664525U) + 1013904223U;
}

static void inactive_workload_task(void *parameter)
{
    (void)parameter;

    while (g_experiment_running) {

        if (g_workload_state == 0) {
            /*
               Workload is inactive.
               This represents the idle phase of the normal experiment.
            */
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /*
           This branch should not be used in this idle-only experiment.
           It is kept only as a safety fallback.
        */
        do_minimal_operation();
        taskYIELD();
    }

    vTaskDelete(NULL);
}


/* ---------------- Main experiment ---------------- */

void app_main(void)
{
    uint32_t cpu_freq_hz = CPU_FREQ_HZ;

    printf("# experiment=ppk2_stable_idle_baseline\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n", (unsigned long)EXPECTED_CPU_FREQ_HZ);
    printf("# stable_idle_duration_ms=%d\n", STABLE_IDLE_DURATION_MS);
    printf("# create_inactive_workload_task=%d\n", CREATE_INACTIVE_WORKLOAD_TASK);
    printf("# workload_core=%d\n", WORKLOAD_CORE);
    printf("# use_marker=%d\n", USE_MARKER);
    printf("# marker_gpio=%d\n", MARKER_GPIO);

    if (cpu_freq_hz != EXPECTED_CPU_FREQ_HZ) {
        printf("# WARNING: CPU frequency is not 240MHz\n");
    }

    marker_init();

    g_workload_state = 0;
    g_experiment_running = true;

#if CREATE_INACTIVE_WORKLOAD_TASK
    BaseType_t workload_created =
        xTaskCreatePinnedToCore(
            inactive_workload_task,
            "inactive_workload_task",
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
#endif

    /*
       Let the system settle before the official stable idle phase.
    */
    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# phase=stable_idle\n");
    g_workload_state = 0;
    marker_set(0);

    vTaskDelay(pdMS_TO_TICKS(STABLE_IDLE_DURATION_MS));

    g_experiment_running = false;

    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# experiment_complete\n");
    printf("# workload_result=%lu\n", (unsigned long)g_workload_result);
}