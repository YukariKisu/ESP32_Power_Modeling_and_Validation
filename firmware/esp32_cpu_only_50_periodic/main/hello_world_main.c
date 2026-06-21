#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_timer.h"

#define I2C_MASTER_SCL_IO       22
#define I2C_MASTER_SDA_IO       21
#define I2C_MASTER_NUM          I2C_NUM_0
#define I2C_MASTER_FREQ_HZ      400000
#define I2C_TIMEOUT_MS          1000

#define INA226_ADDR             0x40
#define REG_CONFIG              0x00
#define REG_SHUNT_VOLTAGE       0x01
#define REG_BUS_VOLTAGE         0x02
#define REG_POWER               0x03
#define REG_CURRENT             0x04
#define REG_CALIBRATION         0x05
#define REG_MANUFACTURER_ID     0xFE
#define REG_DIE_ID              0xFF

#define CONFIG_VALUE            0x4127
#define CALIB_VALUE             0x2800

#define CURRENT_LSB_A           0.00005f
#define SHUNT_LSB_V             0.0000025f
#define BUS_LSB_V               0.00125f
#define POWER_LSB_W             (25.0f * CURRENT_LSB_A)

#define INITIAL_IDLE_MS         10000
#define PERIODIC_BUSY_MS        2000
#define PERIODIC_IDLE_MS        2000
#define PERIODIC_REPEAT_COUNT   3
#define FINAL_IDLE_MS           10000

#define WORKLOAD_CYCLE_MS       100
#define WORKLOAD_BUSY_MS        99
#define MEASUREMENT_INTERVAL_MS 20

static volatile int g_workload_state = 0;
static volatile bool g_experiment_running = true;
static volatile uint32_t g_workload_result = 1;
static int64_t g_experiment_start_us = 0;

static esp_err_t i2c_master_init(void)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    esp_err_t err = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (err != ESP_OK) return err;

    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

static esp_err_t write_register(uint8_t reg, uint16_t value)
{
    uint8_t data[3] = {
        reg,
        (uint8_t)((value >> 8) & 0xFF),
        (uint8_t)(value & 0xFF)
    };

    return i2c_master_write_to_device(
        I2C_MASTER_NUM,
        INA226_ADDR,
        data,
        sizeof(data),
        pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );
}

static esp_err_t read_register(uint8_t reg, uint16_t *value)
{
    uint8_t data[2] = {0};

    esp_err_t err = i2c_master_write_read_device(
        I2C_MASTER_NUM,
        INA226_ADDR,
        &reg,
        1,
        data,
        2,
        pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );

    if (err != ESP_OK) return err;

    *value = ((uint16_t)data[0] << 8) | data[1];
    return ESP_OK;
}

static esp_err_t ina226_init(void)
{
    uint16_t value = 0;
    esp_err_t err;

    err = write_register(REG_CONFIG, CONFIG_VALUE);
    if (err != ESP_OK) {
        printf("# config_write_failed=%s\n", esp_err_to_name(err));
        return err;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    err = read_register(REG_CONFIG, &value);
    if (err != ESP_OK) {
        printf("# config_read_failed=%s\n", esp_err_to_name(err));
        return err;
    }
    printf("# config=0x%04X\n", value);

    err = write_register(REG_CALIBRATION, CALIB_VALUE);
    if (err != ESP_OK) {
        printf("# calibration_write_failed=%s\n", esp_err_to_name(err));
        return err;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    err = read_register(REG_CALIBRATION, &value);
    if (err != ESP_OK) {
        printf("# calibration_read_failed=%s\n", esp_err_to_name(err));
        return err;
    }
    printf("# calibration=0x%04X\n", value);

    err = read_register(REG_MANUFACTURER_ID, &value);
    if (err == ESP_OK) {
        printf("# manufacturer_id=0x%04X\n", value);
    }

    err = read_register(REG_DIE_ID, &value);
    if (err == ESP_OK) {
        printf("# die_id=0x%04X\n", value);
    }

    return ESP_OK;
}

static void run_cpu_busy_period(void)
{
    int64_t busy_start_us = esp_timer_get_time();
    int64_t busy_duration_us = (int64_t)WORKLOAD_BUSY_MS * 1000;

    while ((esp_timer_get_time() - busy_start_us) < busy_duration_us) {
        g_workload_result =
            (g_workload_result * 1664525U) + 1013904223U;
        g_workload_result ^= g_workload_result >> 13;
        g_workload_result *= 2654435761U;
    }
}

static void cpu_workload_task(void *parameter)
{
    (void)parameter;

    while (g_experiment_running) {
        if (g_workload_state == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        TickType_t cycle_start = xTaskGetTickCount();

        run_cpu_busy_period();

        vTaskDelayUntil(
            &cycle_start,
            pdMS_TO_TICKS(WORKLOAD_CYCLE_MS)
        );
    }

    vTaskDelete(NULL);
}

static void measurement_task(void *parameter)
{
    (void)parameter;

    TickType_t last_wake_time = xTaskGetTickCount();

    while (g_experiment_running) {
        uint16_t shunt_raw = 0;
        uint16_t bus_raw = 0;
        uint16_t current_raw = 0;
        uint16_t power_raw = 0;

        esp_err_t e1 = read_register(REG_SHUNT_VOLTAGE, &shunt_raw);
        esp_err_t e2 = read_register(REG_BUS_VOLTAGE, &bus_raw);
        esp_err_t e3 = read_register(REG_CURRENT, &current_raw);
        esp_err_t e4 = read_register(REG_POWER, &power_raw);

        int16_t shunt_signed = (int16_t)shunt_raw;
        int16_t current_signed = (int16_t)current_raw;

        float shunt_voltage_mv =
            -shunt_signed * SHUNT_LSB_V * 1000.0f;

        float bus_voltage_v =
            bus_raw * BUS_LSB_V;

        float current_ma =
            -current_signed * CURRENT_LSB_A * 1000.0f;

        float power_mw =
            power_raw * POWER_LSB_W * 1000.0f;

        int64_t timestamp_ms =
            (esp_timer_get_time() - g_experiment_start_us) / 1000;

        bool read_ok =
            e1 == ESP_OK &&
            e2 == ESP_OK &&
            e3 == ESP_OK &&
            e4 == ESP_OK;

        printf(
            "%lld,%.6f,%.6f,%.3f,%.3f,%d,%s\n",
            timestamp_ms,
            shunt_voltage_mv,
            bus_voltage_v,
            current_ma,
            power_mw,
            g_workload_state,
            read_ok ? "OK" : "READ_FAIL"
        );

        fflush(stdout);

        vTaskDelayUntil(
            &last_wake_time,
            pdMS_TO_TICKS(MEASUREMENT_INTERVAL_MS)
        );
    }

    vTaskDelete(NULL);
}

void app_main(void)
{
    esp_err_t err = i2c_master_init();
    if (err != ESP_OK) {
        printf("I2C init failed: %s\n", esp_err_to_name(err));
        return;
    }

    err = ina226_init();
    if (err != ESP_OK) {
        printf("INA226 init failed: %s\n", esp_err_to_name(err));
        return;
    }

    printf("# experiment=cpu_only_50periodic_workload\n");
    printf("# composition=CPU_only\n");
    printf("# task_type=fixed_integer_arithmetic\n");
    printf("# workload_type=periodic_busy_idle\n");
    printf("# periodic_busy_ms=%d\n", PERIODIC_BUSY_MS);
    printf("# periodic_idle_ms=%d\n", PERIODIC_IDLE_MS);
    printf("# periodic_repeat_count=%d\n", PERIODIC_REPEAT_COUNT);
    printf("# workload_core=1\n");
    printf("# busy_ms_inside_burst=%d\n", WORKLOAD_BUSY_MS);
    printf("# cycle_ms_inside_burst=%d\n", WORKLOAD_CYCLE_MS);
    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);
    printf("# measurement_interval_ms=%d\n", MEASUREMENT_INTERVAL_MS);

    printf(
        "timestamp_ms,"
        "shunt_voltage_mV,"
        "bus_voltage_V,"
        "current_mA,"
        "power_mW,"
        "workload_state,"
        "status\n"
    );

    g_workload_state = 0;
    g_experiment_running = true;
    g_experiment_start_us = esp_timer_get_time();

    BaseType_t measurement_created =
        xTaskCreatePinnedToCore(
            measurement_task,
            "measurement_task",
            4096,
            NULL,
            5,
            NULL,
            0
        );

    BaseType_t workload_created =
        xTaskCreatePinnedToCore(
            cpu_workload_task,
            "cpu_workload_task",
            4096,
            NULL,
            5,
            NULL,
            1
        );

    if (measurement_created != pdPASS ||
        workload_created != pdPASS) {
        printf("# task_creation_failed\n");
        g_experiment_running = false;
        return;
    }

    printf("# phase=initial_idle\n");
    g_workload_state = 0;
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

    for (int i = 0; i < PERIODIC_REPEAT_COUNT; i++) {
        printf("# phase=periodic_busy_%d\n", i + 1);
        g_workload_state = 1;
        vTaskDelay(pdMS_TO_TICKS(PERIODIC_BUSY_MS));

        printf("# phase=periodic_idle_%d\n", i + 1);
        g_workload_state = 0;
        vTaskDelay(pdMS_TO_TICKS(PERIODIC_IDLE_MS));
    }

    printf("# phase=final_idle\n");
    g_workload_state = 0;
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    g_experiment_running = false;
    vTaskDelay(pdMS_TO_TICKS(200));

    printf("# experiment_complete\n");
    printf(
        "# workload_result=%lu\n",
        (unsigned long)g_workload_result
    );
}