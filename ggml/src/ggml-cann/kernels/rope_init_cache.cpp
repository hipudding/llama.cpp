#include "kernel_operator.h"
#include "rope.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 1

class InitCache {
   public:
    __aicore__ inline InitCache() {}
    __aicore__ inline void init(GM_ADDR position,
                                GM_ADDR sin_output,
                                GM_ADDR cos_output,
                                rope_param& param,
                                int64_t* input_ne_ub) {
        /*Init sin&cos cache for rope, impl of ggml_compute_forward_rope_f32().
        each kernel process input_ne[0]*1 cache.
        */

        // Input has four dims. [batch, seq_len, heads, head_dim].
        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        // arange param
        // head_dim = param.input_ne[0];
        // head_dim = param.input_ne[0];
        head_dim = input_ne_ub[0];
        first_value = 0;
        diff_value = 1;
        count = head_dim / 2;

        // power param
        theta_scale = param.theta_scale;
        
        // broadcast param
        // arange_shape: [count, 1] -> broadcast_shape0: [count, 2]
        arange_shape[0] = count;
        arange_shape[1] = 1;
        broadcast_shape0[0] = count;
        broadcast_shape0[1] = 2;

        // arange_shape1: [1, count] -> broadcast_shape2: [2, count]
        arange_shape1[0] = 1;
        arange_shape1[1] = count;
        broadcast_shape2[0] = 2;
        broadcast_shape2[1] = count;
        
        // position_shape: [1, 1] -> broadcast_shape1: [1, head_dim]
        position_shape[0] = 1;
        position_shape[1] = 1;
        broadcast_shape1[0] = 1;
        broadcast_shape1[1] = head_dim;

        // position raw and brcst size.
        position_size = 1;
        broadcast_size  = broadcast_shape1[0] * broadcast_shape1[1];
        
        // other param
        attn_factor = param.attn_factor;
        freq_scale = param.freq_scale;
        is_neox = param.is_neox;
        is_glm = param.is_glm;

        // stride
        position_stride = op_block_idx;
        output_stride = op_block_idx * broadcast_size;

        position_gm.SetGlobalBuffer((__gm__ float_t*)position + position_stride, 
                                    1);
        output_sin_gm.SetGlobalBuffer((__gm__ float_t*)sin_output + 
                                                        output_stride, 
                                                       broadcast_size);
        output_cos_gm.SetGlobalBuffer((__gm__ float_t*)cos_output + 
                                                        output_stride, 
                                                       broadcast_size);
        
        pipe.InitBuffer(power_queue, BUFFER_NUM, 
                        (sizeof(float_t)*count+32-1)/32*32);
        pipe.InitBuffer(position_queue, BUFFER_NUM, 
                        (sizeof(float_t)*position_size+32-1)/32*32);
        pipe.InitBuffer(arange_queue, BUFFER_NUM, 
                        (sizeof(float_t)*count+32-1)/32*32);
        pipe.InitBuffer(sin_mul_mscale_queue, BUFFER_NUM, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(cos_mul_mscale_queue, BUFFER_NUM, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(broadcast_power_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(theta_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(sin_buffer,
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(cos_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
    }

    __aicore__ inline void copy_in() {
        LocalTensor<float_t> input_local = 
                                        position_queue.AllocTensor<float_t>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = position_size * sizeof(float_t);
        DataCopyPadExtParams<float_t> padParams;
        DataCopyPad(input_local, position_gm, dataCopyParams, padParams);

        position_queue.EnQue(input_local);
    }

    __aicore__ inline void copy_out() {
        LocalTensor<float_t> sin_local = sin_mul_mscale_queue.DeQue<float_t>();
        int32_t BLOCK_NUM = 32 / sizeof(float_t);
        DataCopy(output_sin_gm, sin_local, (broadcast_size + BLOCK_NUM - 1) 
                                            / BLOCK_NUM * BLOCK_NUM);

        LocalTensor<float_t> cos_local = cos_mul_mscale_queue.DeQue<float_t>();
        DataCopy(output_cos_gm, cos_local, (broadcast_size + BLOCK_NUM - 1) 
                                           / BLOCK_NUM * BLOCK_NUM);
        
        sin_mul_mscale_queue.FreeTensor(sin_local);
        cos_mul_mscale_queue.FreeTensor(cos_local);
    }

    __aicore__ inline void calculate() {

        // arange    
        LocalTensor<float_t> arange_local = arange_queue.AllocTensor<float_t>();
        ArithProgression<float_t>(arange_local, first_value, diff_value, count);
        
        // pow
        LocalTensor<float_t> power_local = power_queue.AllocTensor<float_t>();
        Power<float_t, false>(power_local, static_cast<float_t>(theta_scale), 
                              arange_local);
        
        LocalTensor<float_t> power_brcast_local = 
                                       broadcast_power_buffer.Get<float_t>();

        //TODO: is_glm==true.
        if (!is_glm && !is_neox) {    
            // for :dst_data[0] = x0*cos_theta*zeta - x1*sin_theta*zeta;
            //      dst_data[1] = x0*sin_theta*zeta + x1*cos_theta*zeta;
            // the value of 0,1 or 2,3, ..., should be same.

            // broadcast: e.g. arange [64, 1] -> [64, 2]
            BroadCast<float_t, 2, 1>(power_brcast_local, power_local, 
                                     broadcast_shape0, arange_shape);
            // position: [1] 
            copy_in();
            LocalTensor<float_t> position_local = 
                                     position_queue.DeQue<float_t>();
            position_value = position_local.GetValue(0);
            position_queue.FreeTensor(position_local);
        }
        else {
            // for: dst_data[0]        = x0*cos_theta - x1*sin_theta;
            //      dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            // the value of 0,n_dims/2 or 1,n/dims/2+1 should be same.

            // broadcast: e.g. arange [1, 64] -> [2, 64]
            BroadCast<float_t, 2, 0>(power_brcast_local, power_local, 
                                     broadcast_shape2, arange_shape1);

            // position * freq_scale
            copy_in();
            LocalTensor<float_t> position_local = 
                                        position_queue.DeQue<float_t>();
            position_value = position_local.GetValue(0);
            position_value = position_value * freq_scale;  
            position_queue.FreeTensor(position_local);     
        }
        
        // theta
        LocalTensor<float_t> theta_local = theta_buffer.Get<float_t>(); 
        Muls(theta_local, power_brcast_local, position_value, 
             broadcast_size);

        // sin & cos
        // TODO: if ext_factor != 0
        LocalTensor<float_t> sin_local = sin_buffer.Get<float_t>(); 
        Sin<float_t, false>(sin_local, theta_local);
        LocalTensor<float_t> sin_mul_mscale_local = 
                                    sin_mul_mscale_queue.AllocTensor<float_t>(); 
        Muls(sin_mul_mscale_local, sin_local, attn_factor, broadcast_size);

        LocalTensor<float_t> cos_local = cos_buffer.Get<float_t>(); 
        Cos<float_t, false>(cos_local, theta_local);
        LocalTensor<float_t> cos_mul_mscale_local = 
                                    cos_mul_mscale_queue.AllocTensor<float_t>(); 
        Muls(cos_mul_mscale_local, cos_local, attn_factor, broadcast_size);

        // release, VECCALC not need.
        arange_queue.FreeTensor(arange_local);
        power_queue.FreeTensor(power_local);
        
        // output
        sin_mul_mscale_queue.EnQue<float_t>(sin_mul_mscale_local);
        cos_mul_mscale_queue.EnQue<float_t>(cos_mul_mscale_local);
        copy_out();
    }

   private:

    int64_t head_dim;
    float_t first_value;
    float_t diff_value;
    int32_t count;
    float_t theta_scale;
    float_t attn_factor;
    float_t freq_scale;
    bool is_neox;
    bool is_glm;

    uint32_t broadcast_shape0[2];
    uint32_t broadcast_shape1[2];
    uint32_t broadcast_shape2[2];
    uint32_t position_shape[2];
    uint32_t arange_shape[2];
    uint32_t arange_shape1[2];
    int64_t broadcast_size;
    int64_t position_size;
    int64_t position_stride;
    int64_t output_stride;
    float_t position_value;

    TPipe pipe;
    GlobalTensor<float_t> position_gm;
    GlobalTensor<float_t> output_sin_gm;
    GlobalTensor<float_t> output_cos_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> arange_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> power_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> position_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sin_mul_mscale_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cos_mul_mscale_queue;
    TBuf<QuePosition::VECCALC> broadcast_power_buffer;
    TBuf<QuePosition::VECCALC> theta_buffer;
    TBuf<QuePosition::VECCALC> sin_buffer;
    TBuf<QuePosition::VECCALC> cos_buffer;
    
};

template <typename T>
__aicore__ inline void copy_to_ub(GM_ADDR gm, T *ub, int32_t size) {
    auto gm_ptr = (__gm__ uint8_t *)gm;
    auto ub_ptr = (uint8_t *)(ub);
    for (int32_t i = 0; i < size; ++i, ++ub_ptr, ++gm_ptr) {
        *ub_ptr = *gm_ptr;
    }
}

extern "C" __global__ __aicore__ void ascendc_rope_init_cache(
                                                          GM_ADDR position_gm,
                                                          GM_ADDR output_sin_gm,
                                                          GM_ADDR output_cos_gm,
                                                          GM_ADDR param,
                                                          GM_ADDR input_ne_gm
                                                          ) {
    // copy params from gm to ub.
    rope_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < static_cast<int32_t>(sizeof(rope_param) / sizeof(uint8_t));
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    int64_t input_ne_ub[4];

    copy_to_ub(input_ne_gm, input_ne_ub, 32);

    InitCache op;
    op.init(position_gm, output_sin_gm, output_cos_gm, param_ub, input_ne_ub);
    op.calculate(); 
}