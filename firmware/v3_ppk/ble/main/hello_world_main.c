#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "esp_cpu.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "sdkconfig.h"

#include "esp_bt.h"
#include "esp_bt_main.h"
#include "esp_gap_ble_api.h"

/* ---------------- Experiment settings ---------------- */

#define SETTLING_IDLE_MS        3000
#define SYNC_PULSE_MS           1000
#define RECOVERY_IDLE_MS        5000

#define INITIAL_IDLE_MS         10000
#define ACTIVE_DURATION_MS      20000
#define FINAL_IDLE_MS           10000

#define EXPECTED_CPU_FREQ_HZ    240000000UL
#define CPU_FREQ_HZ ((uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000UL)

#define WORKLOAD_CORE           1

/* BLE advertising workload settings */
#define BLE_ADV_INTERVAL_MS     100
#define BLE_ADV_INTERVAL_UNITS  ((uint16_t)((BLE_ADV_INTERVAL_MS * 1000) / 625))
#define BLE_ADV_PAYLOAD_SIZE    31
#define BLE_TX_POWER_LEVEL      ESP_PWR_LVL_P3

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

static const char *TAG = "ble_adv_100ms";

static volatile uint32_t g_sync_result = 1;
static volatile bool g_ble_raw_adv_config_done = false;
static volatile bool g_ble_adv_started = false;
static volatile bool g_ble_adv_stopped = false;
static uint8_t g_ble_adv_payload[BLE_ADV_PAYLOAD_SIZE];

static esp_ble_adv_params_t g_adv_params = {
    .adv_int_min       = BLE_ADV_INTERVAL_UNITS,
    .adv_int_max       = BLE_ADV_INTERVAL_UNITS,
    .adv_type          = ADV_TYPE_NONCONN_IND,
    .own_addr_type     = BLE_ADDR_TYPE_PUBLIC,
    .channel_map       = ADV_CHNL_ALL,
    .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
};

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
        run_busy_ccount_us(1000);
        taskYIELD();
    }
}

/* ---------------- BLE advertising setup ---------------- */

static void gap_event_handler(esp_gap_ble_cb_event_t event,
                              esp_ble_gap_cb_param_t *param)
{
    switch (event) {
    case ESP_GAP_BLE_ADV_DATA_RAW_SET_COMPLETE_EVT:
        g_ble_raw_adv_config_done = true;
#if ENABLE_PHASE_LOGS
        printf("# ble_raw_adv_data_configured status=%d\n",
               param->adv_data_raw_cmpl.status);
#endif
        break;

    case ESP_GAP_BLE_ADV_START_COMPLETE_EVT:
        g_ble_adv_started = (param->adv_start_cmpl.status == ESP_BT_STATUS_SUCCESS);
#if ENABLE_PHASE_LOGS
        printf("# adv_start_complete status=%d\n",
               param->adv_start_cmpl.status);
#endif
        break;

    case ESP_GAP_BLE_ADV_STOP_COMPLETE_EVT:
        g_ble_adv_stopped = (param->adv_stop_cmpl.status == ESP_BT_STATUS_SUCCESS);
#if ENABLE_PHASE_LOGS
        printf("# adv_stop_complete status=%d\n",
               param->adv_stop_cmpl.status);
#endif
        break;

    default:
        break;
    }
}

static void disable_wifi_for_measurement(void)
{
    esp_err_t err;

    err = esp_wifi_stop();
    if (err != ESP_OK && err != ESP_ERR_WIFI_NOT_INIT) {
        ESP_LOGW(TAG, "esp_wifi_stop: %s", esp_err_to_name(err));
    }

    err = esp_wifi_deinit();
    if (err != ESP_OK && err != ESP_ERR_WIFI_NOT_INIT) {
        ESP_LOGW(TAG, "esp_wifi_deinit: %s", esp_err_to_name(err));
    }
}

static void fill_fixed_ble_payload(void)
{
    /* 31-byte raw advertising payload:
       3 bytes Flags AD structure + 28 bytes Manufacturer Specific Data. */
    g_ble_adv_payload[0] = 0x02;
    g_ble_adv_payload[1] = 0x01;
    g_ble_adv_payload[2] = 0x06;

    g_ble_adv_payload[3] = 0x1B;
    g_ble_adv_payload[4] = 0xFF;
    for (int i = 5; i < BLE_ADV_PAYLOAD_SIZE; i++) {
        g_ble_adv_payload[i] = (uint8_t)('A' + ((i - 5) % 26));
    }
}

static void wait_for_flag(volatile bool *flag, uint32_t timeout_ms)
{
    uint32_t elapsed_ms = 0;

    while (!(*flag) && elapsed_ms < timeout_ms) {
        vTaskDelay(pdMS_TO_TICKS(10));
        elapsed_ms += 10;
    }
}

static void ble_stack_init_for_advertising(void)
{
    esp_err_t err;

    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(err);
    }

    disable_wifi_for_measurement();
    ESP_ERROR_CHECK(esp_sleep_disable_wakeup_source(ESP_SLEEP_WAKEUP_ALL));

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT));

    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_BLE));

    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());

    ESP_ERROR_CHECK(esp_ble_gap_register_callback(gap_event_handler));
    ESP_ERROR_CHECK(esp_ble_tx_power_set(ESP_BLE_PWR_TYPE_ADV, BLE_TX_POWER_LEVEL));

    fill_fixed_ble_payload();
    ESP_ERROR_CHECK(esp_ble_gap_config_adv_data_raw(
        g_ble_adv_payload,
        BLE_ADV_PAYLOAD_SIZE
    ));
    wait_for_flag(&g_ble_raw_adv_config_done, 1000);
}

static void ble_advertising_start(void)
{
    g_ble_adv_started = false;
    g_ble_adv_stopped = false;
    ESP_ERROR_CHECK(esp_ble_gap_start_advertising(&g_adv_params));
    wait_for_flag(&g_ble_adv_started, 1000);
}

static void ble_advertising_stop(void)
{
    g_ble_adv_stopped = false;
    ESP_ERROR_CHECK(esp_ble_gap_stop_advertising());
    wait_for_flag(&g_ble_adv_stopped, 1000);
}

/* ---------------- Main experiment ---------------- */

void app_main(void)
{
    uint32_t cpu_freq_hz = CPU_FREQ_HZ;

#if ENABLE_PHASE_LOGS
    printf("# experiment=ppk2_ble_adv_100ms_with_cpu_sync_pulse\n");
    printf("# measurement_device=PPK2\n");
    printf("# internal_current_sensor=disabled\n");
    printf("# cpu_freq_hz=%lu\n", (unsigned long)cpu_freq_hz);
    printf("# expected_cpu_freq_hz=%lu\n", (unsigned long)EXPECTED_CPU_FREQ_HZ);
    printf("# workload_core=%d\n", WORKLOAD_CORE);
    printf("# ble_mode=broadcast_advertising\n");
    printf("# ble_condition=advertising_enabled_during_active_phase\n");
    printf("# ble_adv_type=non_connectable_non_scannable\n");
    printf("# ble_adv_payload_size=%d\n", BLE_ADV_PAYLOAD_SIZE);
    printf("# ble_adv_interval_ms=%d\n", BLE_ADV_INTERVAL_MS);
    printf("# ble_adv_channels=37,38,39\n");
    printf("# ble_tx_power_level=%d\n", BLE_TX_POWER_LEVEL);
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

#if ENABLE_PHASE_LOGS
    printf("# phase=ble_stack_init\n");
#endif
    ble_stack_init_for_advertising();

    vTaskDelay(pdMS_TO_TICKS(1000));

#if ENABLE_PHASE_LOGS
    printf("# phase=settling_idle\n");
#endif
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
    vTaskDelay(pdMS_TO_TICKS(RECOVERY_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=initial_idle_ble_initialized_adv_stopped\n");
#endif
    marker_set(0);
    vTaskDelay(pdMS_TO_TICKS(INITIAL_IDLE_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=ble_advertising_active_100ms\n");
#endif
    marker_set(1);
    ble_advertising_start();
    vTaskDelay(pdMS_TO_TICKS(ACTIVE_DURATION_MS));

#if ENABLE_PHASE_LOGS
    printf("# phase=ble_advertising_stop\n");
#endif
    ble_advertising_stop();
    marker_set(0);

#if ENABLE_PHASE_LOGS
    printf("# phase=final_idle_ble_initialized_adv_stopped\n");
#endif
    vTaskDelay(pdMS_TO_TICKS(FINAL_IDLE_MS));

    vTaskDelay(pdMS_TO_TICKS(1000));

#if ENABLE_PHASE_LOGS
    printf("# experiment_complete\n");
    printf("# sync_result=%lu\n", (unsigned long)g_sync_result);
#endif
}