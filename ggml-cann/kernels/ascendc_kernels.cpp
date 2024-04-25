#include "ascendc_kernels.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#else
#include "aclrtlaunch_ascendc_dequantize_q4_0.h"
#include "aclrtlaunch_ascendc_quantize_q4_0.h"
#endif


#ifdef __CCE_KT_TEST__
#include <acl/acl.h>

uint8_t* to_gm(uint8_t* ptr, size_t size) {
    uint8_t* gm = (uint8_t*)AscendC::GmAlloc(size);
    aclrtMemcpy(gm, size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    return gm;
}

void free_gm(uint8_t* ptr) {
    aclrtFree(ptr);
}

extern "C" __global__ __aicore__ void ascendc_dequantize_q4_0(GM_ADDR x, GM_ADDR y, GM_ADDR size);
extern "C" __global__ __aicore__ void ascendc_quantize_q4_0(GM_ADDR x, GM_ADDR y, GM_ADDR size);
#endif

void cann_dequantize_q4_0(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint8_t* size) {
#ifdef __CCE_KT_TEST__
    uint8_t* size_host = to_gm(size, sizeof(size_t));
    uint8_t* x_host = to_gm(x, *((size_t*)size_host));
    uint8_t* y_host = to_gm(y, *((size_t*)size_host));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ascendc_dequantize_q4_0, 1, x_host, y_host, size_host);
    free_gm(size_host);
    free_gm(x_host);
    free_gm(y_host);
#else
    aclrtlaunch_ascendc_dequantize_q4_0(block_dim, stream, x, y, size);
#endif
}

void cann_quantize_q4_0(uint32_t block_dim, void* stream, uint8_t* x, uint8_t* y, uint8_t* size) {
#ifdef __CCE_KT_TEST__
    uint8_t* size_host = to_gm(size, sizeof(size_t));
    uint8_t* x_host = to_gm(x, *((size_t*)size_host));
    uint8_t* y_host = to_gm(y, *((size_t*)size_host));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ascendc_quantize_q4_0, 1, x_host, y_host, size_host);
    free_gm(size_host);
    free_gm(x_host);
    free_gm(y_host);
#else
    aclrtlaunch_ascendc_quantize_q4_0(block_dim, stream, x, y, size);
#endif
}