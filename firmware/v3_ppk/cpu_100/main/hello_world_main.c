#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/portmacro.h"

#include "sdkconfig.h"
#include "esp_cpu.h"

/* ---------------- Experiment settings ---------------- */

#define INITIAL_IDLE_MS             10000
#define FINAL_IDLE_MS               10000

/* Clean rise section */
#define INTERRUPT_DISABLED_PULSE_US 10000    // 10 ms

/* Normal busy plateau after the clean rise */
#define ACTIVE_PLATEAU_US           100000   // 100 ms

#define CONTROL_INTERVAL_US         5

#define EXPECTED_CPU_FREQ_HZ        240000000UL
#define CPU_FREQ_HZ ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

/* Match original CPU100 training condition */
#define WORKLOAD_CORE               1

/* Experiment state */
static volatile bool g_experiment_done = false;

/* Prevent optimization */
static volatile uint32_t g_workload_result = 1;


/* ---------------- CCOUNT workload ---------------- */

static inline void do_workload_operation(void)
{
    g_workload_result =
        (g_workload_result * 1664525U) + 1013904223U;
    g_workload_result ^= g_workload_result >> 13;
    g_workload_result *= 2654435761U;
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

static void run_busy_plateau_us(uint32_t plateau_us)
{
    uint32_t target_cycles =
        (uint32_t)(((uint64_t)CPU_FREQ_HZ * plateau_us) / 1000000ULL);

    uint32_t start_cycles = esp_cpu_get_cycle_count();

    while ((uint32_t)(esp_cpu_get_cycle_count() - start_cycles) < target_cycles) {
        run_busy_ccount_us(CONTROL_INTERVAL_US);

        /*
           Interrupts are enabled here.
           Yield occasionally so the system remains stable.
        */
        taskYIELD();
    }
}


/* ---------------- Workload task ---------------- */

static void interrupt_disabled_clean_rise_task(void *parameter)
{
    (void)parameter;

    /*
       Part 1:
       Interrupt-disabled clean rise.
       This is the section used for tau_rise estimation.
    */
    portDISABLE_INTERRUPTS();

    run_busy_ccount_us(INTERRUPT_DISABLED_PULSE_US);

    portENABLE_INTERRUPTS();

    /*
       Part 2:
       Normal busy plateau.
       This makes the waveform easier to identify,
       but tau_rise should still be estimated from the first 10 ms.
    */
    run_busy_plateau_us(ACTIVE_PLATEAU_US);

    g_experiment_done = true;

    vTaskDelete(NULL);
}



void app_main(void)
{
    uint32_t cpu_freq_hz = CPU_FREQ_HZ;

    printf("# experiment=ppk2_cpu100_interrupt_disabled_clean_rise_with_plateau\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# purpose=tau_rise_estimation_clean_step_response\n");

    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n", (unsigned long)EXPECTED_CPU_FREQ_HZ);

    printf("# workload_core=%d\n", WORKLOAD_CORE);
    printf("# interrupt_disabled_pulse_us=%d\n", INTERRUPT_DISABLED_PULSE_US);
    printf("# active_plateau_us=%d\n", ACTIVE_PLATEAU_US);
    printf("# control_interval_us=%d\n", CONTROL_INTERVAL_US);

    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);

    printf("# gpio_marker=disabled\n");

    if (cpu_freq_hz != EXPECTED_CPU_FREQ_HZ) {
        printf("# WARNING: CPU frequency is not 240MHz\n");
    }

    g_experiment_done = false;

    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# phase=initial_idle\n");
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

    printf("# phase=interrupt_disabled_clean_rise_then_busy_plateau\n");

    /*
       Give UART a short time to finish printing before the clean pulse starts.
    */
    vTaskDelay(pdMS_TO_TICKS(100));

    /*
       Create workload task only after the initial idle phase.
       This prevents the workload task from waking up during initial idle.
    */
    BaseType_t workload_created =
        xTaskCreatePinnedToCore(
            interrupt_disabled_clean_rise_task,
            "clean_rise_task",
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

    while (!g_experiment_done) {
        vTaskDelay(pdMS_TO_TICKS(1));
    }

    printf("# phase=final_idle\n");
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    printf("# experiment_complete\n");
    printf("# workload_result=%lu\n", (unsigned long)g_workload_result);
}

