#include "kernel_operator.h"
#include "rope.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 2

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

        // power param
        theta_scale = param.theta_scale;
        
        // broadcast param
        broadcast_shape0[0] = count_;
        broadcast_shape0[1] = 2;
        broadcast_shape1[0] = 1;
        broadcast_shape1[1] = head_dim;
        broadcast_shape2[0] = param.position_ne[0];;
        broadcast_shape2[1] = head_dim;
        broadcast_size  = broadcast_shape2[0] * broadcast_shape2[1];

        position_shape[0] = param.position_ne[0];
        position_shape[1] = 1;
        position_size = position_shape[0] * position_shape[1];

        arange_shape[0] = count_;
        arange_shape[1] = 1; 

        position_gm.SetGlobalBuffer((__gm__ float_t*)position);
        output_sin_gm.SetGlobalBuffer((__gm__ float_t*)sin_output);
        output_cos_gm.SetGlobalBuffer((__gm__ float_t*)cos_output);
        
        pipe.InitBuffer(power_queue, BUFFER_NUM, (sizeof(float_t) * count_ + 32 -1)/32*32);
        pipe.InitBuffer(position_queue, BUFFER_NUM, (sizeof(float_t) *  param.position_ne[0] + 32 -1)/32*32);
        pipe.InitBuffer(arange_queue, BUFFER_NUM, (sizeof(float_t) * count_ + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_tmp_buffer, (sizeof(float_t) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_power_buffer, (sizeof(float_t) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_power_buffer2, (sizeof(float_t) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(broadcast_position_buffer, (sizeof(float_t) * 2 * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(theta_buffer, (sizeof(float_t) * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(sin_queue, BUFFER_NUM, (sizeof(float_t) * broadcast_size + 32 -1)/32*32);
        pipe.InitBuffer(cos_queue, BUFFER_NUM, (sizeof(float_t) * broadcast_size + 32 -1)/32*32);
    }

    __aicore__ inline void copy_in() {
        LocalTensor<float_t> input_local = position_queue.AllocTensor<float_t>();
        int32_t BLOCK_NUM = 32 / sizeof(float_t);
        DataCopy(input_local, position_gm, (position_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        position_queue.EnQue(input_local);
    }

    __aicore__ inline void copy_out() {
        LocalTensor<float_t> sin_local = sin_queue.DeQue<float_t>();
        int32_t BLOCK_NUM = 32 / sizeof(float_t);
        DataCopy(output_sin_gm, sin_local, (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);

        LocalTensor<float_t> cos_local = cos_queue.DeQue<float_t>();
        DataCopy(output_cos_gm, cos_local, (broadcast_size + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
        
        sin_queue.FreeTensor(sin_local);
        cos_queue.FreeTensor(cos_local);
    }

    __aicore__ inline void calculate() {
        // arange    
        LocalTensor<float_t> arange_local = arange_queue.AllocTensor<float_t>();
        ArithProgression<float_t>(arange_local, first_value_, diff_value_, count_);
        
        // pow
        LocalTensor<float_t> power_local = power_queue.AllocTensor<float_t>();
        Power<float_t, false>(power_local, static_cast<float_t>(theta_scale), arange_local);

        // broadcast1: arange [64, 1] -> [64, 2] -> [1, 128] -> [10, 128]
        LocalTensor<float_t> power_brcast_local = broadcast_power_buffer.Get<float_t>();
        BroadCast<float_t, 2, 1>(power_brcast_local, power_local, broadcast_shape0, arange_shape);
        LocalTensor<float_t> power_brcast_local2 = broadcast_power_buffer2.Get<float_t>();
        BroadCast<float_t, 2, 0>(power_brcast_local2, power_brcast_local, broadcast_shape2, broadcast_shape1);

        // boradcast2: position [10, 1] -> [10, 128]
        copy_in();
        LocalTensor<float_t> position_brcast_local = broadcast_position_buffer.Get<float_t>();
        LocalTensor<float_t> position_local = position_queue.DeQue<float_t>();
        BroadCast<float_t, 2, 1>(position_brcast_local, position_local, broadcast_shape2, position_shape);
        
        // mul
        LocalTensor<float_t> theta_local = theta_buffer.Get<float_t>(); 
        Mul<float_t>(theta_local, position_brcast_local, power_brcast_local2, broadcast_size);

        // sin & cos
        LocalTensor<float_t> sin_local = sin_queue.AllocTensor<float_t>(); 
        Sin<float_t, false>(sin_local, theta_local);

        LocalTensor<float_t> cos_local = cos_queue.AllocTensor<float_t>(); 
        Cos<float_t, false>(cos_local, theta_local);

        // release
        arange_queue.FreeTensor(arange_local);
        power_queue.FreeTensor(power_local);
        broadcast_power_buffer.FreeTensor(power_brcast_local);
        position_queue.FreeTensor(position_local);
        broadcast_position_buffer.FreeTensor(position_brcast_local);
        theta_buffer.FreeTensor(theta_local);
        
        // output
        sin_queue.EnQue<float_t>(sin_local);
        cos_queue.EnQue<float_t>(cos_local);
        copy_out();
    }

   private:

    int64_t head_dim;
    float_t first_value_;
    float_t diff_value_;
    int32_t count_;
    float_t theta_scale;

    uint32_t broadcast_shape[2];
    uint32_t broadcast_shape0[2];
    uint32_t broadcast_shape1[2];
    uint32_t broadcast_shape2[2];
    uint32_t position_shape[2];
    uint32_t arange_shape[2];
    int64_t broadcast_size;
    int64_t position_size;
  
    TPipe pipe;
    GlobalTensor<float_t> position_gm;
    GlobalTensor<float_t> output_sin_gm;
    GlobalTensor<float_t> output_cos_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> arange_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> power_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> position_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sin_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cos_queue;
    TBuf<QuePosition::VECCALC> broadcast_tmp_buffer;
    TBuf<QuePosition::VECCALC> broadcast_position_buffer;
    TBuf<QuePosition::VECCALC> broadcast_power_buffer;
    TBuf<QuePosition::VECCALC> broadcast_power_buffer2;
    TBuf<QuePosition::VECCALC> theta_buffer;
    
};

extern "C" __global__ __aicore__ void ascendc_rope_init_cache(GM_ADDR position_gm,
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
}