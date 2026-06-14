#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_err.h"

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

static void print_register(uint8_t reg, const char *name)
{
    uint16_t value = 0;
    esp_err_t ret = read_register(reg, &value);

    if (ret == ESP_OK) {
        printf("%s register 0x%02X = 0x%04X\n", name, reg, value);
    } else {
        printf("%s register 0x%02X read failed: %s\n",
               name, reg, esp_err_to_name(ret));
    }
}

void app_main(void)
{
    printf("INA226 simple I2C debug started\n");
    printf("SDA = GPIO21, SCL = GPIO22, address = 0x40, speed = 10 kHz\n");

    esp_err_t err = i2c_master_init();
    if (err != ESP_OK) {
        printf("I2C init failed: %s\n", esp_err_to_name(err));
        return;
    }

    while (1) {
        printf("\n--- Config write check ---\n");
        err = write_register(REG_CONFIG, 0x4127);
        printf("Write config 0x4127: %s\n", esp_err_to_name(err));
        vTaskDelay(pdMS_TO_TICKS(100));
        print_register(REG_CONFIG, "Config");
        
        printf("\n--- Basic read check ---\n");
        print_register(REG_CONFIG, "Config");
        print_register(REG_MANUFACTURER_ID, "Manufacturer ID");
        print_register(REG_DIE_ID, "Die ID");

        printf("\n--- Write/read check ---\n");
        err = write_register(REG_CALIBRATION, 0x0001);
        printf("Write calibration 0x0001: %s\n", esp_err_to_name(err));

        vTaskDelay(pdMS_TO_TICKS(100));

        print_register(REG_CALIBRATION, "Calibration");
        print_register(REG_SHUNT_VOLTAGE, "Shunt voltage");
        print_register(REG_BUS_VOLTAGE, "Bus voltage");
        print_register(REG_CURRENT, "Current");

        printf("---\n");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
