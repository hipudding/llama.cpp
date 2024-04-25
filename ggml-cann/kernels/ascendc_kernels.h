#ifndef ASCENDC_KERNELS_H
#define ASCENDC_KERNELS_H


#include <stdint.h>

void cann_dequantize_q4_0(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint8_t* size);
void cann_quantize_q4_0(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint8_t* size);

#endif //ASCENDC_KERNELS_H