#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void app_main(void)
{
    uint32_t count = 0;

    printf("ESP32 serial test started\n");

    while (1) {
        printf("serial_test,count=%lu\n", (unsigned long)count++);
        fflush(stdout);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
