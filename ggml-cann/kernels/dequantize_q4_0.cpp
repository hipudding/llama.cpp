#include "dequantize_q4_0.h"

using namespace AscendC;

#define BUFFER_NUM 2

__aicore__ inline int32_t align_ceil(int32_t n, int32_t align) { return ((n + align) & ~(align-1)); }

__aicore__ inline int32_t align_floor(int32_t n, int32_t align) { return (n & ~(align-1)); }


#define QK4_0 32
typedef struct {
    uint16_t d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

class KernelDequantizeQ4_0
{
public:
    __aicore__ inline KernelDequantizeQ4_0() {}
    __aicore__ inline void init(GM_ADDR x, GM_ADDR y, size_t size) {
        uint64_t src_block_size =
            align_ceil(size / GetBlockNum(), sizeof(block_q4_0));
        uint64_t src_offset = GetBlockIdx() * src_block_size;
        src_block_size =
            (src_offset + src_block_size > (size / 32 * sizeof(block_q4_0)))
                ? (size / 32 * sizeof(block_q4_0) - src_offset)
                : src_block_size;
        uint64_t dst_block_size =
            align_ceil(size / GetBlockNum(), QK4_0 * sizeof(float));
        uint64_t dst_offset = GetBlockIdx() * dst_block_size;
        dst_block_size =
            (dst_offset + dst_block_size > size * sizeof(float))
                ? (size * sizeof(float) - dst_offset)
                : dst_block_size;
        
        xGM.SetGlobalBuffer((__gm__ int4b_t*)x + src_offset, src_block_size);
        yGM.SetGlobalBuffer((__gm__ float*)y + dst_offset, dst_block_size);

        pipe.InitBuffer(input_queue, BUFFER_NUM, QK4_0 * sizeof(int4b_t));
        // Ascendc do not support cast int4b_t -> float, but support int4b_t ->
        // half -> float.
        pipe.InitBuffer(cast_queue, BUFFER_NUM, QK4_0 * sizeof(half));
        pipe.InitBuffer(copy_queue, BUFFER_NUM, QK4_0 * sizeof(float));
        pipe.InitBuffer(output_queue, BUFFER_NUM, QK4_0 * sizeof(float));
    }

    __aicore__ inline void copy_in(uint32_t offset) {
        LocalTensor<int4b_t> x_local = input_queue.AllocTensor<int4b_t>();
        // offset + 2 to skip scale.
        DataCopy(x_local, xGM[offset + 2], QK4_0);
        input_queue.EnQue(x_local);
    }

    __aicore__ inline void copy_out(uint32_t offset) {
        LocalTensor<float> y_local = output_queue.DeQue<float>();
        DataCopy(yGM[offset], y_local, QK4_0);
        output_queue.FreeTensor(y_local);
    }

    __aicore__ inline void calculate(uint32_t offset, uint32_t len) {
        copy_in(offset);

        LocalTensor<int4b_t> x_local = input_queue.DeQue<int4b_t>();
        LocalTensor<half> cast_local = cast_queue.AllocTensor<half>();
        LocalTensor<float> copy_local = copy_queue.AllocTensor<float>();
        LocalTensor<float> y_local = output_queue.AllocTensor<float>();

        Cast(x_local, cast_local, RoundMode::CAST_NONE, QK4_0);
        Cast(cast_local, copy_local, RoundMode::CAST_NONE, QK4_0);

        
    }

    __aicore__ inline void run() {
        calculate(0, 10);
    }

private:
    uint64_t block_size;
    uint64_t offset;

    TPipe pipe;
    GlobalTensor<int4b_t> xGM;
    GlobalTensor<float> yGM;
    TQue<QuePosition::VECIN, BUFFER_NUM> input_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cast_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> copy_queue;
};

extern "C" __global__ __aicore__ void ascendc_dequantize_q4_0(GM_ADDR x, GM_ADDR y, GM_ADDR size)
{
    size_t size_ub;
    auto size_gm_ptr = (__gm__ uint8_t*)size;
    auto size_ub_ptr = (uint8_t*)&size_ub;

    for (int32_t i = 0; i < sizeof(size_t) / sizeof(uint8_t);
         ++i, ++size_gm_ptr, ++size_ub_ptr)
    {
        *size_ub_ptr = *size_gm_ptr;
    }

    KernelDequantizeQ4_0 dequantize_q4_0;
    dequantize_q4_0.init(x, y, size_ub);
    dequantize_q4_0.run();
}