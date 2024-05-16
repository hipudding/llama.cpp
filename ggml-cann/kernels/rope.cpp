#include "kernel_operator.h"
#include "rope.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 2
#define QK8_0 32

class InitCache {
   public:
    __aicore__ inline InitCache() {}
    __aicore__ inline void init(GM_ADDR position,
                                GM_ADDR sin_output,
                                GM_ADDR cos_output,
                                rope_param& param) {
        // Input has four dims. [batch, seq_len, heads, head_dim].

        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        // arange param
        head_dim = param.input_ne[0];
        first_value_ = 0;
        diff_value_ = 1;
        count_ = head_dim / 2;
        //count_ = 64;

        // power param
        theta_scale = param.theta_scale;
        // theta_scale = 0.5;
        // PRINTF("scale", theta_scale);
        // broadcast param
        broadcast_shape[0] = param.position_ne[0];
        broadcast_shape[1] = count_;
        broadcast_size  = broadcast_shape[0] * broadcast_shape[1];

        position_shape[0] = param.position_ne[0];
        position_shape[1] = 1;
        position_size = position_shape[0] * position_shape[1];

        arange_shape[0] = 1;
        arange_shape[1] = count_;
        arange_size = arange_shape[0] * arange_shape[1];        

        PRINTF("broadcast_size: %d \n", broadcast_size);
        PRINTF("broadcast_bf_position_shape: %d, %d \n", position_shape[0], position_shape[1]);
        PRINTF("broadcast_bf_arange_shape: %d, %d \n", arange_shape[0], arange_shape[1]);
        PRINTF("broadcast_af_shape: %d, %d \n", broadcast_shape[0], broadcast_shape[1]);    
                
        //output_gm.SetGlobalBuffer((__gm__ float*)output, param.position_ne[0]*count_);
        position_gm.SetGlobalBuffer((__gm__ float*)position);
        output_sin_gm.SetGlobalBuffer((__gm__ float*)sin_output);
        output_cos_gm.SetGlobalBuffer((__gm__ float*)cos_output);
        
        pipe.InitBuffer(power_queue, BUFFER_NUM, (sizeof(float) * count_ + 32 -1)/32*32);
        pipe.InitBuffer(position_queue, BUFFER_NUM, (sizeof(float) *  param.position_ne[0] + 32 -1)/32*32);
        pipe.InitBuffer(arange_queue, BUFFER_NUM, (sizeof(float) * count_ + 32 -1)/32*32);
        pipe.InitBuffer(output_queue, BUFFER_NUM,  (sizeof(float) * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_tmp_buffer, (sizeof(float) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_power_buffer, (sizeof(float) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_position_buffer, (sizeof(float) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(theta_buffer, (sizeof(float) * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(sin_queue, BUFFER_NUM, (sizeof(float) * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(cos_queue, BUFFER_NUM, (sizeof(float) * broadcast_size + 32 -1)/32*32);
    }

    __aicore__ inline void copy_in() {
        LocalTensor<float> input_local = position_queue.AllocTensor<float>();
        int32_t BLOCK_NUM = 32 / sizeof(float);
        DataCopy(input_local, position_gm, (position_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        PRINTF("INPUT: %d \n", (position_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        position_queue.EnQue(input_local);
    }

    __aicore__ inline void copy_out() {
        LocalTensor<float> sin_local = sin_queue.DeQue<float>();
        int32_t BLOCK_NUM = 32 / sizeof(float);
        DataCopy(output_sin_gm, sin_local, (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        // PRINTF("OUTPUT: %d \n", (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);

        LocalTensor<float> cos_local = cos_queue.DeQue<float>();
        DataCopy(output_cos_gm, cos_local, (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);

        // LocalTensor<float> output_local = output_queue.DeQue<float>();
        // DataCopy(output_sin_gm, output_local, (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        
        sin_queue.FreeTensor(sin_local);
        cos_queue.FreeTensor(cos_local);

        // output_queue.FreeTensor(output_local);
    }

    __aicore__ inline void calculate() {
        // arange    
        LocalTensor<float> arange_local = arange_queue.AllocTensor<float>();
        ArithProgression<float>(arange_local, static_cast<float>(first_value_), 
                                static_cast<float>(diff_value_), count_);
        
        // pow
        LocalTensor<float> power_local = power_queue.AllocTensor<float>();
        Power<float, false>(power_local, theta_scale, arange_local);

        // boradcast1: position [10, 1] -> [10, 64]
        copy_in();
        LocalTensor<float> position_brcast_local = broadcast_position_buffer.Get<float>();
        LocalTensor<float> position_local = position_queue.DeQue<float>();
        // LocalTensor<uint8_t> tmp_buffer = broadcast_tmp_buffer.Get<uint8_t>(); 
        // BroadCast<float, 2, 1>(position_brcast_local, position_local, broadcast_shape, position_shape, tmp_buffer);
        BroadCast<float, 2, 1>(position_brcast_local, position_local, broadcast_shape, position_shape);
        pipe_barrier(PIPE_ALL);
        
        // broadcast2: arange [1, 64] -> [10, 64]
        LocalTensor<float> power_brcast_local = broadcast_power_buffer.Get<float>();
        // BroadCast<float, 2, 0>(power_brcast_local, power_local, broadcast_shape, arange_shape, tmp_buffer);
        BroadCast<float, 2, 0>(power_brcast_local, power_local, broadcast_shape, arange_shape);
        
        // // mul
        LocalTensor<float> theta_local = theta_buffer.Get<float>(); 
        Mul(theta_local, position_brcast_local, power_brcast_local, broadcast_size);

        // sin & cos
        LocalTensor<float> sin_local = sin_queue.AllocTensor<float>(); 
        Sin<float, false>(sin_local, theta_local);

        LocalTensor<float> cos_local = cos_queue.AllocTensor<float>(); 
        Cos<float, false>(cos_local, theta_local);

        // // cout
        // LocalTensor<float> output_local = output_queue.AllocTensor<float>();
        // Muls(output_local, position_brcast_local, (float)1, 64*16);

        // release
        arange_queue.FreeTensor(arange_local);
        power_queue.FreeTensor(power_local);
        broadcast_power_buffer.FreeTensor(power_brcast_local);
        position_queue.FreeTensor(position_local);
        broadcast_position_buffer.FreeTensor(position_brcast_local);
        // broadcast_tmp_buffer.FreeTensor(tmp_buffer);
        theta_buffer.FreeTensor(theta_local);
        
        // output
        sin_queue.EnQue<float>(sin_local);
        cos_queue.EnQue<float>(cos_local);
        // output_queue.EnQue<float>(output_local);
        copy_out();
    }

   private:

    int64_t head_dim;
    int64_t first_value_;
    float_t diff_value_;
    int64_t count_;
    float_t theta_scale;

    uint32_t broadcast_shape[2];
    uint32_t position_shape[2];
    uint32_t arange_shape[2];
    int64_t broadcast_size;
    int64_t position_size;
    int64_t arange_size;
  
    TPipe pipe;
    GlobalTensor<float> position_gm;
    GlobalTensor<float> output_sin_gm;
    GlobalTensor<float> output_cos_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> arange_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> power_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> position_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sin_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cos_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
    TBuf<QuePosition::VECCALC> broadcast_tmp_buffer;
    TBuf<QuePosition::VECCALC> broadcast_position_buffer;
    TBuf<QuePosition::VECCALC> broadcast_power_buffer;
    TBuf<QuePosition::VECCALC> theta_buffer;
    
};


extern "C" __global__ __aicore__ void ascendc_rope(GM_ADDR input_gm,
                                                   GM_ADDR position_gm,
                                                   GM_ADDR output_sin_gm,
                                                   GM_ADDR output_cos_gm,
                                                   GM_ADDR param) {
    // copy params from gm to ub.
    rope_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(rope_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    InitCache op;
    op.init(position_gm, output_sin_gm, output_cos_gm, param_ub);
    op.calculate(); 

    // ROPE op;
    // op.init(input_gm, position_gm, output_gm, param_ub);
    // op.calculate();
}
