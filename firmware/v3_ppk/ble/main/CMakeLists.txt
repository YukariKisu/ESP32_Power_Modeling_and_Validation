idf_component_register(
    SRCS "hello_world_main.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        bt
        nvs_flash
        esp_wifi
        driver
        esp_timer
        esp_hw_support
)