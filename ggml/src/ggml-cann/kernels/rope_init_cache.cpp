#include "kernel_operator.h"
#include "rope.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 1

class InitCache {
   public:
    __aicore__ inline InitCache() {}
    __aicore__ inline void init(GM_ADDR position,
                                GM_ADDR freq_factors,
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
        ext_factor = param.ext_factor;
        is_neox = param.is_neox;
        use_freq_factors = param.use_freq_factors;
        corr_dims0 = param.corr_dims0;
        corr_dims1 = param.corr_dims1;
        corr_dims0_neg = param.corr_dims0_neg;
        rope_yarn_ramp_max_inv = param.rope_yarn_ramp_max_inv;

        // stride
        position_stride = op_block_idx;
        output_stride = op_block_idx * broadcast_size;

        if (use_freq_factors) {
            freq_factors_gm.SetGlobalBuffer((__gm__ float_t*)freq_factors, count);
            pipe.InitBuffer(freq_factors_queue, BUFFER_NUM, 
                        (sizeof(float_t)*count+32-1)/32*32);
            pipe.InitBuffer(freq_factors_brcast_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        }

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
        pipe.InitBuffer(theta_base_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(theta_div_ff_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(theta_interp_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(arange_brcast_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(arange_brcast_div_buff, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(arange_brcast_div_add_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(rope_yarn_ramp_y_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(rope_yarn_ramp_y_max_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(rope_yarn_ramp_y_maxmin_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(rope_yarn_ramp_y_maxmin_neg_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(ramp_mix_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(theta0_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(theta1_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(arange_brcast_add_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(mscale_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
                        


        pipe.InitBuffer(theta_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(sin_buffer,
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
        pipe.InitBuffer(cos_buffer, 
                        (sizeof(float_t)*broadcast_size+32-1)/32*32);
    }

    __aicore__ inline void copy_position_in() {
        LocalTensor<float_t> input_local = 
                                        position_queue.AllocTensor<float_t>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = position_size * sizeof(float_t);
        DataCopyPadExtParams<float_t> padParams;
        DataCopyPad(input_local, position_gm, dataCopyParams, padParams);

        position_queue.EnQue(input_local);
    }

    __aicore__ inline void copy_freq_factors_in() {
        LocalTensor<float_t> freq_factors_local = 
                                     freq_factors_queue.AllocTensor<float_t>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = count * sizeof(float_t);
        DataCopyPadExtParams<float_t> padParams;
        DataCopyPad(freq_factors_local, freq_factors_gm, dataCopyParams, 
                    padParams);
        PRINTF("\nfreq_scale: \n");
        for (int i =0 ; i<64; i++) {
            PRINTF("%f,",  freq_factors_local.GetValue(i));
        }

        freq_factors_queue.EnQue(freq_factors_local);
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
        
        // theta stride
        LocalTensor<float_t> power_local = power_queue.AllocTensor<float_t>();
        Power<float_t, false>(power_local, static_cast<float_t>(theta_scale), 
                              arange_local);
        
        LocalTensor<float_t> power_brcast_local = 
                                       broadcast_power_buffer.Get<float_t>();
        if (!is_neox) {    
            // for :dst_data[0] = x0*cos_theta*zeta - x1*sin_theta*zeta;
            //      dst_data[1] = x0*sin_theta*zeta + x1*cos_theta*zeta;
            // the value of 0,1 or 2,3, ..., should be same.

            // broadcast: e.g. arange [64, 1] -> [64, 2]
            BroadCast<float_t, 2, 1>(power_brcast_local, power_local, 
                                     broadcast_shape0, arange_shape);
        }
        else {
            // for: dst_data[0]        = x0*cos_theta - x1*sin_theta;
            //      dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
            // the value of 0,n_dims/2 or 1,n/dims/2+1 should be same.

            // broadcast: e.g. arange [1, 64] -> [2, 64]
            BroadCast<float_t, 2, 0>(power_brcast_local, power_local, 
                                     broadcast_shape2, arange_shape1);
        }

        // position 
        copy_position_in();
        LocalTensor<float_t> position_local = 
                                    position_queue.DeQue<float_t>();
        position_value = position_local.GetValue(0);
        position_queue.FreeTensor(position_local);

        // theta_base 
        LocalTensor<float_t> theta_base_local = theta_base_buffer.Get<float_t>(); 
        Muls(theta_base_local, power_brcast_local, position_value, 
             broadcast_size);

        // theta_extrap = theta_base/ff
        LocalTensor<float_t> theta_div_ff_local;
        if (use_freq_factors) {
            copy_freq_factors_in();
            LocalTensor<float_t> freq_factors_local = 
                                        freq_factors_queue.DeQue<float_t>();
            LocalTensor<float_t> freq_factors_brcast_local = freq_factors_brcast_buffer.Get<float_t>(); 
            if (!is_neox) {    
                BroadCast<float_t, 2, 1>(freq_factors_brcast_local, 
                                     freq_factors_local, broadcast_shape0, 
                                     arange_shape);
            }
            else {
                BroadCast<float_t, 2, 0>(freq_factors_brcast_local, 
                                     freq_factors_local, broadcast_shape2, arange_shape1);
            }
            
            theta_div_ff_local = theta_div_ff_buffer.Get<float_t>(); 
            Div(theta_div_ff_local, theta_base_local, freq_factors_brcast_local,
                broadcast_size);
            freq_factors_queue.FreeTensor(freq_factors_local);
        }
        else {
            theta_div_ff_local = theta_base_local;
        }

        // theta_interp
        LocalTensor<float_t> theta_interp_local = theta_interp_buffer.Get<float_t>(); 
        PRINTF("freq_scale: %f \n", freq_scale);
        PRINTF("mscale: %f \n", attn_factor);
        Muls(theta_interp_local, theta_div_ff_local, freq_scale, broadcast_size);
        PRINTF("\n");
        for (int i =0; i<128; i++) {
            PRINTF("%d: %f, ", i, theta_interp_local.GetValue(i));
        }
        
        LocalTensor<float_t> theta_local =  theta_buffer.Get<float_t>();
        LocalTensor<float_t> sin_mul_mscale_local = 
                                    sin_mul_mscale_queue.AllocTensor<float_t>(); 
        LocalTensor<float_t> cos_mul_mscale_local = 
                                    cos_mul_mscale_queue.AllocTensor<float_t>(); 
        float_t mscale = 0;
        if (ext_factor != 0.0f && freq_scale != 1.0f) {
            // Need to check
            // rope_yarn_ramp
            LocalTensor<float_t> arange_brcast_local = 
                                       arange_brcast_buffer.Get<float_t>();
            BroadCast<float_t, 2, 1>(arange_brcast_local, arange_local, 
                                     broadcast_shape0, arange_shape);
            
            LocalTensor<float_t> arange_brcast_div_local = 
                                       arange_brcast_div_buff.Get<float_t>();
            float_t div_param = 0.5;
            Muls(arange_brcast_div_local, arange_brcast_local, div_param, broadcast_size);

            LocalTensor<float_t> arange_brcast_div_add_local = 
                                       arange_brcast_div_add_buffer.Get<float_t>();
            Adds(arange_brcast_div_add_local, arange_brcast_div_local, corr_dims0_neg, broadcast_size);
            
            LocalTensor<float_t> rope_yarn_ramp_y_local = 
                                       rope_yarn_ramp_y_buffer.Get<float_t>();
            Muls(rope_yarn_ramp_y_local, arange_brcast_div_add_local, rope_yarn_ramp_max_inv, broadcast_size);

            LocalTensor<float_t> rope_yarn_ramp_y_max_local = 
                                       rope_yarn_ramp_y_max_buffer.Get<float_t>();
            float_t max_param = 0.0f;
            Maxs(rope_yarn_ramp_y_max_local, rope_yarn_ramp_y_local, max_param, broadcast_size);  
            
            LocalTensor<float_t> rope_yarn_ramp_y_maxmin_local = 
                                       rope_yarn_ramp_y_maxmin_buffer.Get<float_t>();
            float_t mins_param = 1.0f;
            Mins(rope_yarn_ramp_y_maxmin_local, rope_yarn_ramp_y_max_local, mins_param, broadcast_size);

            LocalTensor<float_t> rope_yarn_ramp_y_maxmin_neg_local = 
                                       rope_yarn_ramp_y_maxmin_neg_buffer.Get<float_t>();
            float_t neg_param = -1.0f;
            Muls(rope_yarn_ramp_y_maxmin_neg_local, rope_yarn_ramp_y_maxmin_local, neg_param, broadcast_size);
            
            LocalTensor<float_t> ramp_mix_local = ramp_mix_buffer.Get<float_t>();
            float_t add_param = 1.0f;
            Adds(ramp_mix_local, rope_yarn_ramp_y_maxmin_neg_local, add_param, broadcast_size);

            // theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            LocalTensor<float_t> theta0_local =  theta0_buffer.Get<float_t>();
            Mul(theta0_local, theta_interp_local, rope_yarn_ramp_y_maxmin_local, broadcast_size);

            LocalTensor<float_t> theta1_local =  theta1_buffer.Get<float_t>();
            Mul(theta1_local, theta_div_ff_local, ramp_mix_local, broadcast_size);

            Add(theta_local, theta0_local, theta1_local, broadcast_size);

            // mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale)
            LocalTensor<float_t> arange_brcast_add_local = 
                                       arange_brcast_add_buffer.Get<float_t>();
            add_param = 1.0f;
            Adds(arange_brcast_add_local, arange_brcast_local, add_param, broadcast_size);

            LocalTensor<float_t> mscale_local = mscale_buffer.Get<float_t>();
            Power<float_t, false>(mscale_local, static_cast<float_t>(attn_factor), 
                                  arange_brcast_add_local);
            
            //
            LocalTensor<float_t> sin_local = sin_buffer.Get<float_t>(); 
            LocalTensor<float_t> cos_local = cos_buffer.Get<float_t>(); 
            Sin<float_t, false>(sin_local, theta_div_ff_local);
            Cos<float_t, false>(cos_local, theta_div_ff_local);

            Mul(sin_mul_mscale_local, sin_local, mscale_local, broadcast_size);
            Mul(cos_mul_mscale_local, cos_local, mscale_local, broadcast_size);
            
        }
        else {
            theta_local = theta_interp_local;
            mscale = attn_factor;

            LocalTensor<float_t> sin_local = sin_buffer.Get<float_t>(); 
            LocalTensor<float_t> cos_local = cos_buffer.Get<float_t>(); 
            Sin<float_t, false>(sin_local, theta_local);
            Cos<float_t, false>(cos_local, theta_local);

            Muls(sin_mul_mscale_local, sin_local, mscale, broadcast_size);
            Muls(cos_mul_mscale_local, cos_local, mscale, broadcast_size);
            // PRINTF("\n sin: \n");
            // for (int i =0; i<128; i++) {
            //     PRINTF("%d: %f, ", i, sin_local.GetValue(i));
            // }
            // PRINTF("\n sin * mscale: \n");
            // for (int i =0; i<128; i++) {
            //     PRINTF("%d: %f, ", i, sin_mul_mscale_local.GetValue(i));
            // }
        }

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
    float_t ext_factor;
    bool is_neox;
    bool use_freq_factors;
    float_t corr_dims0;
    float_t corr_dims1;
    float_t corr_dims0_neg;
    float_t rope_yarn_ramp_max_inv;


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
    GlobalTensor<float_t> freq_factors_gm;
    GlobalTensor<float_t> output_sin_gm;
    GlobalTensor<float_t> output_cos_gm;
    TQue<QuePosition::VECIN, BUFFER_NUM> arange_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> power_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> position_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> freq_factors_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> sin_mul_mscale_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cos_mul_mscale_queue;
    TBuf<QuePosition::VECCALC> broadcast_power_buffer;
    TBuf<QuePosition::VECCALC> theta_base_buffer;
    TBuf<QuePosition::VECCALC> theta_buffer;
    TBuf<QuePosition::VECCALC> sin_buffer;
    TBuf<QuePosition::VECCALC> cos_buffer;
    TBuf<QuePosition::VECCALC> freq_factors_brcast_buffer;
    TBuf<QuePosition::VECCALC> theta_div_ff_buffer;
    TBuf<QuePosition::VECCALC> theta_interp_buffer;
    TBuf<QuePosition::VECCALC> arange_brcast_buffer;
    TBuf<QuePosition::VECCALC> arange_brcast_div_buff;
    TBuf<QuePosition::VECCALC> arange_brcast_div_add_buffer;
    TBuf<QuePosition::VECCALC> rope_yarn_ramp_y_buffer;
    TBuf<QuePosition::VECCALC> rope_yarn_ramp_y_max_buffer;
    TBuf<QuePosition::VECCALC> rope_yarn_ramp_y_maxmin_buffer;
    TBuf<QuePosition::VECCALC> rope_yarn_ramp_y_maxmin_neg_buffer;
    TBuf<QuePosition::VECCALC> ramp_mix_buffer;
    TBuf<QuePosition::VECCALC> theta0_buffer;
    TBuf<QuePosition::VECCALC> theta1_buffer;
    TBuf<QuePosition::VECCALC> arange_brcast_add_buffer;
    TBuf<QuePosition::VECCALC> mscale_buffer;
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
                                                          GM_ADDR freq_factors_gm,
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
    op.init(position_gm, freq_factors_gm, output_sin_gm, output_cos_gm, param_ub, input_ne_ub);
    op.calculate(); 
}