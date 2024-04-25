#include "quantize_q4_0.h"

using namespace AscendC;

#define BUFFER_NUM 2

__aicore__ inline int32_t align_ceil(int32_t n, int32_t align) { return ((n + align) & ~(align-1)); }

__aicore__ inline int32_t align_floor(int32_t n, int32_t align) { return (n & ~(align-1)); }


#define QK4_0 32
typedef struct {
    uint16_t d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

template<typename T>
class QuantizeQ4_0
{
public:
    __aicore__ inline QuantizeQ4_0() {}
    __aicore__ inline void init(GM_ADDR x, GM_ADDR y, size_t size) {
        uint64_t src_block_size = align_ceil(size / GetBlockNum(), QK4_0 * sizeof(T));
        uint64_t src_offset = GetBlockIdx() * src_block_size;
        uint64_t data_length = size * sizeof(T);
        if (src_offset + src_block_size > data_length) {
            src_block_size = data_length - src_offset;
        }
        uint64_t n_elem = src_block_size / sizeof(T);

        uint64_t dst_block_size = n_elem / 32 * sizeof(block_q4_0);
        uint64_t dst_offset = GetBlockIdx() * dst_block_size;
        
        xGM.SetGlobalBuffer((__gm__ T*)x + src_offset, src_block_size);
        yGM.SetGlobalBuffer((__gm__ int4b_t*)y + dst_offset, dst_block_size);

        pipe.InitBuffer(input_queue, BUFFER_NUM, QK4_0 * sizeof(T));
        // Ascendc do not support cast int4b_t -> float, but support int4b_t ->
        // half -> float.
        pipe.InitBuffer(copy_queue, BUFFER_NUM, QK4_0 * sizeof(T));
        pipe.InitBuffer(cast_queue, BUFFER_NUM, QK4_0 * sizeof(half));
        pipe.InitBuffer(output_queue, BUFFER_NUM, QK4_0 * sizeof(int4b_t));
    }

    __aicore__ inline void copy_in(uint32_t offset) {
        LocalTensor<T> x_local = input_queue.AllocTensor<T>();
        DataCopy(x_local, xGM[offset], QK4_0);
        input_queue.EnQue(x_local);
    }

    __aicore__ inline void copy_out(uint32_t offset) {
        
    }

    __aicore__ inline void calculate(uint32_t src_offset, uint32_t dst_offset) {
        copy_in(src_offset);

        LocalTensor<T> x_local = input_queue.DeQue<T>();
        
        //LocalTensor<T> copy_local = copy_queue.AllocTensor<T>();

        
    }

    __aicore__ inline void run() {
        calculate(0, 10);
    }

private:
    uint64_t block_size;
    uint64_t offset;

    TPipe pipe;
    GlobalTensor<T> xGM;
    GlobalTensor<int4b_t> yGM;
    TQue<QuePosition::VECIN, BUFFER_NUM> input_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cast_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> copy_queue;
};

extern "C" __global__ __aicore__ void ascendc_quantize_q4_0(GM_ADDR x, GM_ADDR y, GM_ADDR size)
{
    size_t size_ub;
    auto size_gm_ptr = (__gm__ uint8_t*)size;
    auto size_ub_ptr = (uint8_t*)&size_ub;

    for (int32_t i = 0; i < sizeof(size_t) / sizeof(uint8_t);
         ++i, ++size_gm_ptr, ++size_ub_ptr)
    {
        *size_ub_ptr = *size_gm_ptr;
    }

    // QuantizeQ4_0<float> quantize_q4_0;
    // quantize_q4_0.init(x, y, size_ub);
    // quantize_q4_0.run();
}