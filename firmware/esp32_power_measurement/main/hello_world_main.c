#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_timer.h"

#define I2C_MASTER_SCL_IO      22
#define I2C_MASTER_SDA_IO      21
#define I2C_MASTER_NUM         I2C_NUM_0
#define I2C_MASTER_FREQ_HZ     10000
#define I2C_TIMEOUT_MS         1000

#define INA226_ADDR            0x40

#define REG_CONFIG             0x00
#define REG_SHUNT_VOLTAGE      0x01
#define REG_BUS_VOLTAGE        0x02
#define REG_POWER              0x03
#define REG_CURRENT            0x04
#define REG_CALIBRATION        0x05
#define REG_MANUFACTURER_ID    0xFE
#define REG_DIE_ID             0xFF

#define CONFIG_VALUE           0x4127
#define CALIB_VALUE            0x2800
#define CURRENT_LSB_A          0.00005f

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

    esp_err_t ret = i2c_master_write_read_device(
        I2C_MASTER_NUM,
        INA226_ADDR,
        &reg,
        1,
        data,
        2,
        pdMS_TO_TICKS(I2C_TIMEOUT_MS)
    );

    if (ret != ESP_OK) return ret;

    *value = ((uint16_t)data[0] << 8) | data[1];
    return ESP_OK;
}

static void ina226_init_or_report(void)
{
    uint16_t value = 0;
    esp_err_t err;

    printf("INA226 power measurement firmware started\n");
    printf("time_ms,shunt_raw,bus_raw,current_raw,current_mA,power_raw,status\n");

    err = write_register(REG_CONFIG, CONFIG_VALUE);
    printf("# write_config=0x%04X,%s\n", CONFIG_VALUE, esp_err_to_name(err));

    vTaskDelay(pdMS_TO_TICKS(100));

    err = read_register(REG_CONFIG, &value);
    if (err == ESP_OK) {
        printf("# config_read=0x%04X\n", value);
    } else {
        printf("# config_read_failed=%s\n", esp_err_to_name(err));
    }

    err = write_register(REG_CALIBRATION, CALIB_VALUE);
    printf("# write_calibration=0x%04X,%s\n", CALIB_VALUE, esp_err_to_name(err));

    vTaskDelay(pdMS_TO_TICKS(100));

    err = read_register(REG_CALIBRATION, &value);
    if (err == ESP_OK) {
        printf("# calibration_read=0x%04X\n", value);
    } else {
        printf("# calibration_read_failed=%s\n", esp_err_to_name(err));
    }

    err = read_register(REG_MANUFACTURER_ID, &value);
    if (err == ESP_OK) {
        printf("# manufacturer_id=0x%04X\n", value);
    } else {
        printf("# manufacturer_id_failed=%s\n", esp_err_to_name(err));
    }

    err = read_register(REG_DIE_ID, &value);
    if (err == ESP_OK) {
        printf("# die_id=0x%04X\n", value);
    } else {
        printf("# die_id_failed=%s\n", esp_err_to_name(err));
    }
}

void app_main(void)
{
    esp_err_t err = i2c_master_init();
    if (err != ESP_OK) {
        printf("I2C init failed: %s\n", esp_err_to_name(err));
        return;
    }

    ina226_init_or_report();

    while (1) {
        uint16_t shunt_raw = 0;
        uint16_t bus_raw = 0;
        uint16_t current_raw = 0;
        uint16_t power_raw = 0;

        esp_err_t e1 = read_register(REG_SHUNT_VOLTAGE, &shunt_raw);
        esp_err_t e2 = read_register(REG_BUS_VOLTAGE, &bus_raw);
        esp_err_t e3 = read_register(REG_CURRENT, &current_raw);
        esp_err_t e4 = read_register(REG_POWER, &power_raw);
        
        int16_t current_signed = (int16_t)current_raw;
        float current_a = current_signed * CURRENT_LSB_A;
        float current_ma = current_a * 1000.0f;

        int64_t time_ms = esp_timer_get_time() / 1000;

        if (e1 == ESP_OK && e2 == ESP_OK && e3 == ESP_OK && e4 == ESP_OK) {
            printf("%lld,0x%04X,0x%04X,0x%04X,%.3f,0x%04X,OK\n",
                   time_ms,
                   shunt_raw,
                   bus_raw,
                   current_raw,
                   current_ma,
                   power_raw);
        } else {
            printf("%lld,0x%04X,0x%04X,0x%04X,%.3f,0x%04X,READ_FAIL\n",
                   time_ms,
                   shunt_raw,
                   bus_raw,
                   current_raw,
                   current_ma,
                   power_raw);
        }

        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
