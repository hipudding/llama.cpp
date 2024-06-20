#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#define GGML_CANN_NAME "CANN"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_CANN_MAX_DEVICES 16

#define QK4_0 32
typedef struct {
    uint16_t d;             // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct {
    uint16_t d;        // delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cann_init(int32_t device);

GGML_API GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cann_buffer_type(int32_t device);

GGML_API GGML_CALL int32_t ggml_backend_cann_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cann_get_device_description(
    int32_t device, char* description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cann_get_device_memory(int32_t device,
                                                            size_t* free,
                                                            size_t* total);
void ggml_cann_backend_init(void);
void ggml_cann_backend_free(void);
#ifdef __cplusplus
}
#endif
