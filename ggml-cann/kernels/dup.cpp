#include "kernel_operator.h"
#include "dup.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 2

template <typename SRC_T, typename DST_T>
class DupByRows {
   public:
    __aicore__ inline DupByRows() {}
    __aicore__ inline void init(GM_ADDR src, GM_ADDR dst, dup_param& param) {
        // Input has four dims.
        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        // param
        num_rows = param.src_ne[1] * param.src_ne[2] * param.src_ne[3];
        num_elem = param.src_ne[0];
        
        idx_0 = op_block_idx / (param.src_ne[1] * param.src_ne[2]);
        idx_1 = (op_block_idx - idx_0 * (param.src_ne[1] * param.src_ne[2])) 
                 / (param.src_ne[1]);
        idx_2 = op_block_idx - idx_0 * (param.src_ne[1] * param.src_ne[2]) 
                - idx_1 * param.src_ne[1];
                
        src_stride = param.src_nb[3] * idx_0 + param.src_nb[2] * idx_1
                     + param.src_nb[1] * idx_2;

        dst_stride = param.dst_nb[3] * idx_0 + param.dst_nb[2] * idx_1
                     + param.dst_nb[1] * idx_2;
        
        src_gm.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_T *>(src + src_stride));
        dst_gm.SetGlobalBuffer(reinterpret_cast<__gm__ DST_T *>(dst + dst_stride));

        pipe.InitBuffer(src_queue, BUFFER_NUM, (sizeof(SRC_T) * num_elem + 32 - 1) / 32 * 32);
        pipe.InitBuffer(dst_queue, BUFFER_NUM, (sizeof(DST_T) * num_elem + 32 - 1) / 32 * 32);
    }

    __aicore__ inline void copy_in() {
        LocalTensor<SRC_T> src_local = src_queue.AllocTensor<SRC_T>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_elem * sizeof(SRC_T);
        DataCopyPadExtParams<SRC_T> padParams;
        DataCopyPad(src_local, src_gm, dataCopyParams, padParams);
        
        src_queue.EnQue(src_local);
    }

    __aicore__ inline void copy_out() {
        LocalTensor<DST_T> dst_local = dst_queue.DeQue<DST_T>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_elem * sizeof(DST_T);
        DataCopyPad(dst_gm, dst_local, dataCopyParams);

        dst_queue.FreeTensor(dst_local);
    }

    __aicore__ inline void dup() {
        copy_in();
        
        LocalTensor<SRC_T> src_local = src_queue.DeQue<SRC_T>();
        LocalTensor<DST_T> dst_local = dst_queue.AllocTensor<DST_T>();
        
        int32_t BLOCK_NUM = 32 / sizeof(DST_T);
        DataCopy(dst_local, src_local, (num_elem + BLOCK_NUM - 1) 
                                        / BLOCK_NUM * BLOCK_NUM);    
        dst_queue.EnQue<DST_T>(dst_local);

        //
        src_queue.FreeTensor(src_local);
        copy_out();
    }

    __aicore__ inline void dup_with_cast() {
        copy_in();
        
        LocalTensor<SRC_T> src_local = src_queue.DeQue<SRC_T>();
        LocalTensor<DST_T> dst_local = dst_queue.AllocTensor<DST_T>();
        
        Cast(dst_local, src_local, RoundMode::CAST_NONE, num_elem); 
        dst_queue.EnQue<DST_T>(dst_local);

        //
        src_queue.FreeTensor(src_local);
        copy_out();
    }

   private:
  
    TPipe pipe;
    GlobalTensor<SRC_T> src_gm;
    GlobalTensor<DST_T> dst_gm;

    int64_t num_rows;
    int64_t num_elem;
    int64_t idx_0;
    int64_t idx_1;
    int64_t idx_2;
    int64_t src_stride;
    int64_t dst_stride;
    
    TQue<QuePosition::VECIN, BUFFER_NUM> src_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dst_queue;
};

template <typename T>
__aicore__ inline void copy_to_ub(GM_ADDR gm, T *ub, size_t size) {
    auto gm_ptr = (__gm__ uint8_t *)gm;
    auto ub_ptr = (uint8_t *)(ub);
    for (int32_t i = 0; i < size; ++i, ++ub_ptr, ++gm_ptr) {
        *ub_ptr = *gm_ptr;
    }
}

extern "C" __global__ __aicore__ void ascendc_dup_by_rows_fp16(GM_ADDR src_gm,
                                                               GM_ADDR dst_gm,
                                                               GM_ADDR param) {

    // copy params from gm to ub.
    dup_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(dup_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    DupByRows<half, half> op;
    op.init(src_gm, dst_gm, param_ub);
    op.dup(); 
}

extern "C" __global__ __aicore__ void ascendc_dup_by_rows_fp32(GM_ADDR src_gm,
                                                               GM_ADDR dst_gm,
                                                               GM_ADDR param) {

    // copy params from gm to ub.
    dup_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(dup_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    DupByRows<float_t, float_t> op;
    op.init(src_gm, dst_gm, param_ub);
    op.dup(); 
}

extern "C" __global__ __aicore__ void ascendc_dup_by_rows_fp32_to_fp16(GM_ADDR src_gm,
                                                               GM_ADDR dst_gm,
                                                               GM_ADDR param) {

    // copy params from gm to ub.
    dup_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(dup_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    DupByRows<float_t, half> op;
    op.init(src_gm, dst_gm, param_ub);
    op.dup_with_cast(); 
}

extern "C" __global__ __aicore__ void ascendc_dup_by_rows_fp16_to_fp32(GM_ADDR src_gm,
                                                               GM_ADDR dst_gm,
                                                               GM_ADDR param) {

    // copy params from gm to ub.
    dup_param param_ub;
    auto param_gm_ptr = (__gm__ uint8_t*)param;
    auto param_ub_ptr = (uint8_t*)&param_ub;

    for (int32_t i = 0; i < sizeof(dup_param) / sizeof(uint8_t);
         ++i, ++param_gm_ptr, ++param_ub_ptr) {
        *param_ub_ptr = *param_gm_ptr;
    }

    DupByRows<half, float_t> op;
    op.init(src_gm, dst_gm, param_ub);
    op.dup_with_cast(); 
}