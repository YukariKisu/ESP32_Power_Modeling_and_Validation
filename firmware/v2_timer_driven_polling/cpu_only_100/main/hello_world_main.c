#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/i2c.h"
#include "driver/gpio.h"

#include "esp_err.h"
#include "esp_timer.h"

/* ---------------- I2C / INA226 settings ---------------- */

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

/* Rshunt = 0.01 Ω, Current_LSB = 50 µA/count. */
#define CURRENT_LSB_A           0.00005f
#define SHUNT_LSB_V             0.0000025f
#define BUS_LSB_V               0.00125f
#define POWER_LSB_W             (25.0f * CURRENT_LSB_A)

/* ---------------- Experiment settings ---------------- */

#define INITIAL_IDLE_MS         10000
#define ACTIVE_DURATION_MS      20000
#define FINAL_IDLE_MS           10000

#define WORKLOAD_CYCLE_MS       100
#define WORKLOAD_BUSY_MS        99

#define MEASUREMENT_INTERVAL_MS 1

// Queue length
#define SAMPLE_QUEUE_LENGTH     2048

/* 0 = idle, 1 = active. */
static volatile int g_workload_state = 0;
static volatile bool g_experiment_running = true;

/* Volatile prevents the workload calculation from being optimized away. */
static volatile uint32_t g_workload_result = 1;

static int64_t g_experiment_start_us = 0;

/* ---------------- Sample buffer ---------------- */

typedef struct {
    int64_t timestamp_us;
    float shunt_voltage_mv;
    float bus_voltage_v;
    float current_ma;
    float power_mw;
    int workload_state;
    bool read_ok;
} sample_t;

static QueueHandle_t g_sample_queue = NULL;
static TaskHandle_t g_sampling_task_handle = NULL;
static esp_timer_handle_t g_sampling_timer = NULL;

static volatile uint32_t g_dropped_samples = 0;

/* ---------------- I2C functions ---------------- */

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
    if (err != ESP_OK) {
        return err;
    }

    return i2c_driver_install(
        I2C_MASTER_NUM,
        conf.mode,
        0,
        0,
        0
    );
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

    if (err != ESP_OK) {
        return err;
    }

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

/* ---------------- Workload task ---------------- */

/* Fixed arithmetic workload; intensity is controlled by busy duration. */
static void run_cpu_busy_period(void)
{
    int64_t busy_start_us = esp_timer_get_time();
    int64_t busy_duration_us =
        (int64_t)WORKLOAD_BUSY_MS * 1000;

    while ((esp_timer_get_time() - busy_start_us) < busy_duration_us) {
        g_workload_result =
            (g_workload_result * 1664525U) + 1013904223U;
        g_workload_result ^= g_workload_result >> 13;
        g_workload_result *= 2654435761U;
    }
}

/* CPU workload: 99 ms busy in each 100 ms cycle. */
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

/* ---------------- Timer-driven polling ---------------- */


static void sampling_timer_callback(void *arg)
{
    (void)arg;

    if (g_sampling_task_handle != NULL) {
        xTaskNotifyGive(g_sampling_task_handle);
    }
}

// Producer task:
static void sampling_task(void *parameter)
{
    (void)parameter;

    while (true) {

        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        if (!g_experiment_running) {
            break;
        }

        sample_t sample = {0};


        sample.timestamp_us =
            esp_timer_get_time() - g_experiment_start_us;

        uint16_t shunt_raw = 0;
        uint16_t bus_raw = 0;
        uint16_t current_raw = 0;
        uint16_t power_raw = 0;

        esp_err_t e1 =
            read_register(REG_SHUNT_VOLTAGE, &shunt_raw);
        esp_err_t e2 =
            read_register(REG_BUS_VOLTAGE, &bus_raw);
        esp_err_t e3 =
            read_register(REG_CURRENT, &current_raw);
        esp_err_t e4 =
            read_register(REG_POWER, &power_raw);

        int16_t shunt_signed = (int16_t)shunt_raw;
        int16_t current_signed = (int16_t)current_raw;

        sample.shunt_voltage_mv =
            -shunt_signed * SHUNT_LSB_V * 1000.0f;

        sample.bus_voltage_v =
            bus_raw * BUS_LSB_V;

        sample.current_ma =
            -current_signed * CURRENT_LSB_A * 1000.0f;

        sample.power_mw =
            power_raw * POWER_LSB_W * 1000.0f;

        sample.workload_state = g_workload_state;

        sample.read_ok =
            e1 == ESP_OK &&
            e2 == ESP_OK &&
            e3 == ESP_OK &&
            e4 == ESP_OK;


        if (xQueueSend(g_sample_queue, &sample, 0) != pdTRUE) {
            g_dropped_samples++;
        }
    }

    vTaskDelete(NULL);
}

// Consumer task part
static void logger_task(void *parameter)
{
    (void)parameter;

    sample_t sample;

    while (g_experiment_running ||
           uxQueueMessagesWaiting(g_sample_queue) > 0) {

        if (xQueueReceive(
                g_sample_queue,
                &sample,
                pdMS_TO_TICKS(100)
            ) == pdTRUE) {

            printf(
                "%lld,%.6f,%.6f,%.3f,%.3f,%d,%s\n",
                sample.timestamp_us,
                sample.shunt_voltage_mv,
                sample.bus_voltage_v,
                sample.current_ma,
                sample.power_mw,
                sample.workload_state,
                sample.read_ok ? "OK" : "READ_FAIL"
            );
        }
    }

    printf("# dropped_samples=%lu\n", (unsigned long)g_dropped_samples);

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

    g_sample_queue =
        xQueueCreate(SAMPLE_QUEUE_LENGTH, sizeof(sample_t));

    if (g_sample_queue == NULL) {
        printf("# sample_queue_creation_failed\n");
        return;
    }

    printf("# experiment=cpu_only_100busy_step\n");
    printf("# composition=CPU_only\n");
    printf("# task_type=fixed_integer_arithmetic\n");
    printf("# intensity=high_100busy\n");
    printf("# workload_core=1\n");
    printf("# sampling_core=0\n");
    printf("# logger_core=0\n");
    printf("# busy_ms=%d\n", WORKLOAD_BUSY_MS);
    printf("# cycle_ms=%d\n", WORKLOAD_CYCLE_MS);
    printf("# initial_idle_ms=%d\n", INITIAL_IDLE_MS);
    printf("# active_duration_ms=%d\n", ACTIVE_DURATION_MS);
    printf("# final_idle_ms=%d\n", FINAL_IDLE_MS);
    printf("# measurement_interval_ms=%d\n", MEASUREMENT_INTERVAL_MS);
    printf("# sampling_method=timer_driven_polling\n");
    printf("# buffer_type=FreeRTOS_queue\n");
    printf("# sample_queue_length=%d\n", SAMPLE_QUEUE_LENGTH);
    printf("# uart_usage=log_transfer_only\n");
    printf("# timestamp_source=ESP32_esp_timer_get_time\n");

    printf(
        "timestamp_us,"
        "shunt_voltage_mV,"
        "bus_voltage_V,"
        "current_mA,"
        "power_mW,"
        "workload_state,"
        "status\n"
    );

    const esp_timer_create_args_t sampling_timer_args = {
        .callback = &sampling_timer_callback,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "sampling_timer"
    };

    err = esp_timer_create(
        &sampling_timer_args,
        &g_sampling_timer
    );

    if (err != ESP_OK) {
        printf("# sampling_timer_create_failed=%s\n", esp_err_to_name(err));
        return;
    }

    g_workload_state = 0;
    g_experiment_running = true;
    g_experiment_start_us = esp_timer_get_time();

    BaseType_t sampling_created =
        xTaskCreatePinnedToCore(
            sampling_task,
            "sampling_task",
            4096,
            NULL,
            6,
            &g_sampling_task_handle,
            0
        );

    BaseType_t logger_created =
        xTaskCreatePinnedToCore(
            logger_task,
            "logger_task",
            4096,
            NULL,
            3,
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

    if (sampling_created != pdPASS ||
        logger_created != pdPASS ||
        workload_created != pdPASS) {
        printf("# task_creation_failed\n");
        g_experiment_running = false;
        return;
    }


    err = esp_timer_start_periodic(
        g_sampling_timer,
        MEASUREMENT_INTERVAL_MS * 1000
    );

    if (err != ESP_OK) {
        printf("# sampling_timer_start_failed=%s\n", esp_err_to_name(err));
        g_experiment_running = false;
        return;
    }

    printf("# phase=initial_idle\n");
    g_workload_state = 0;
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

    printf("# phase=active\n");
    g_workload_state = 1;
    vTaskDelay(pdMS_TO_TICKS(ACTIVE_DURATION_MS));

    printf("# phase=final_idle\n");
    g_workload_state = 0;
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));


    g_experiment_running = false;

    if (g_sampling_timer != NULL) {
        esp_timer_stop(g_sampling_timer);
        esp_timer_delete(g_sampling_timer);
        g_sampling_timer = NULL;
    }


    if (g_sampling_task_handle != NULL) {
        xTaskNotifyGive(g_sampling_task_handle);
    }


    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("# experiment_complete\n");
    printf(
        "# workload_result=%lu\n",
        (unsigned long)g_workload_result
    );
}