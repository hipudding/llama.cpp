#include "kernel_operator.h"
#include "get_row_q8_0.h"

using namespace AscendC;

#define BUFFER_NUM 2
#define QK8_0 32

class GET_ROW_Q8_0 {
   public:
    __aicore__ inline GET_ROW_Q8_0() {}
    __aicore__ inline void init(GM_ADDR input, GM_ADDR indices, GM_ADDR output,
                                get_row_param& param) {
        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        // Input has three dims. [batch, height, weight].
        input_row_elems = param.input_ne[0];
        input_batch_elems = param.input_ne[1] * input_row_elems;
        input_length = param.input_ne[2] * input_batch_elems;

        scale_row_elems = param.input_ne[0] / QK8_0;
        scale_batch_elems = param.input_ne[1] * scale_row_elems;

        output_row_elems = param.input_ne[0];

        // Indices has two dims. n_rows = all rows should get.
        // get_row_count, all rows should this thread get.
        uint64_t n_rows = param.indices_ne[0] * param.indices_ne[1];
        get_row_count = n_rows / op_block_num;
        indices_batch_size = param.indices_ne[0] * sizeof(int32_t);

        // Last block, need to handle the tail data.
        if (op_block_idx == op_block_num - 1) {
            get_row_count += n_rows % op_block_num;
        }

        // Get the start position of the current block.
        indices_start_idx = get_row_count * op_block_idx;

        input_gm.SetGlobalBuffer((__gm__ int8_t*)input);
        scale_gm.SetGlobalBuffer((__gm__ half*)(input + input_length));
        indices_gm.SetGlobalBuffer((__gm__ int32_t*)indices);
        output_gm.SetGlobalBuffer((__gm__ float*)output);

        pipe.InitBuffer(input_queue, BUFFER_NUM, QK8_0 * sizeof(int8_t));
        pipe.InitBuffer(cast_queue, BUFFER_NUM, QK8_0 * sizeof(half));
        pipe.InitBuffer(output_queue, BUFFER_NUM, QK8_0 * sizeof(float));
    }

    __aicore__ inline void copy_in(uint32_t offset) {
        LocalTensor<int8_t> input_local = input_queue.AllocTensor<int8_t>();
        DataCopy(input_local, input_gm[offset], QK8_0);
        input_queue.EnQue(input_local);
    }

    __aicore__ inline void copy_out(uint32_t offset) {
        LocalTensor<float> output_local = output_queue.DeQue<float>();
        DataCopy(output_gm[offset], output_local, QK8_0);
        output_queue.FreeTensor(output_local);
    }

    __aicore__ inline void calculate_group(int64_t batch_idx, int64_t row_idx,
                                           int64_t indices_idx,
                                           int64_t group_idx) {
        int64_t input_offset = batch_idx * input_batch_elems +
                               row_idx * input_row_elems + group_idx * QK8_0;
        int64_t scale_elem_idx = batch_idx * scale_batch_elems +
                                 row_idx * scale_row_elems + group_idx;
        int64_t output_offset =
            indices_idx * output_row_elems + group_idx * QK8_0;

        half group_scale = scale_gm.GetValue(scale_elem_idx);

        copy_in(input_offset);
        LocalTensor<int8_t> input_local = input_queue.DeQue<int8_t>();

        LocalTensor<half> cast_local = cast_queue.AllocTensor<half>();
        LocalTensor<float> output_local = output_queue.AllocTensor<float>();

        Cast(cast_local, input_local, RoundMode::CAST_NONE, QK8_0);
        Cast(output_local, cast_local, RoundMode::CAST_NONE, QK8_0);

        Muls(output_local, output_local, (float)group_scale, QK8_0);

        input_queue.FreeTensor(input_local);
        cast_queue.FreeTensor(cast_local);
        output_queue.EnQue(output_local);

        copy_out(output_offset);
    }

    __aicore__ inline void calculate() {
        for (int64_t indices_idx = indices_start_idx;
             indices_idx < indices_start_idx + get_row_count; indices_idx++) {
            int64_t batch_idx = indices_idx / indices_batch_size;
            int64_t row_idx = indices_gm.GetValue(indices_idx);
            int64_t group_size_in_row = input_row_elems / QK8_0;

            for (int64_t i = 0; i < group_size_in_row; i++) {
                calculate_group(batch_idx, row_idx, indices_idx, i);
            }
        }
    }

   private:
    int64_t indices_batch_size;
    int64_t indices_start_idx;
    int64_t get_row_count;
    int64_t input_row_elems;
    int64_t input_batch_elems;
    int64_t scale_row_elems;
    int64_t scale_batch_elems;
    int64_t input_length;
    int64_t output_row_elems;

    TPipe pipe;
    GlobalTensor<int8_t> input_gm;
    GlobalTensor<half> scale_gm;
    GlobalTensor<int32_t> indices_gm;
    GlobalTensor<float> output_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> input_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cast_queue;
};

extern "C" __global__ __aicore__ void ascendc_get_row_q8_0(GM_ADDR input_gm,
                                                   GM_ADDR indices_gm,
                                                   GM_ADDR output_gm,
                                                   GM_ADDR param) {
    // copy params from gm to ub.
    get_row_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(get_row_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    GET_ROW_Q8_0 op;
    op.init(input_gm, indices_gm, output_gm, param_ub);
    op.calculate();
}