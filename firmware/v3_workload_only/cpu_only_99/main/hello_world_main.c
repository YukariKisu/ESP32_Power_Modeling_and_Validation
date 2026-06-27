#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#define IDLE_BEFORE_MS   10000   // 10 sec
#define ACTIVE_MS        20000   // 20 sec
#define IDLE_AFTER_MS    10000   // 10 sec

// 99% busy cycle
#define ACTIVE_CYCLE_MS  1000
#define BUSY_MS          990
#define IDLE_MS          10

static volatile uint32_t g_sink = 0;

static void busy_workload_ms(uint32_t duration_ms)
{
    int64_t start = esp_timer_get_time();

    while ((esp_timer_get_time() - start) < (int64_t)duration_ms * 1000)
    {
        uint32_t x = g_sink;

        for (int i = 0; i < 1000; i++)
        {
            x += i;
            x ^= (x << 1);
            x += 12345;
        }

        g_sink = x;
    }
}

static void active_99_percent_ms(uint32_t duration_ms)
{
    uint32_t elapsed = 0;

    while (elapsed < duration_ms)
    {
        busy_workload_ms(BUSY_MS);

        // Let FreeRTOS idle/task watchdog breathe.
        vTaskDelay(pdMS_TO_TICKS(IDLE_MS));

        elapsed += ACTIVE_CYCLE_MS;
    }
}

void app_main(void)
{
    while (true)
    {
        // Idle before active phase
        vTaskDelay(pdMS_TO_TICKS(IDLE_BEFORE_MS));

        // Active phase: 99% busy, 1% idle/yield
        active_99_percent_ms(ACTIVE_MS);

        // Idle after active phase
        vTaskDelay(pdMS_TO_TICKS(IDLE_AFTER_MS));
    }
}