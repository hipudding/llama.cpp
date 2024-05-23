#include "kernel_operator.h"

#include <cmath>

using namespace AscendC;

#define BUFFER_NUM 2

class DupByRows {
   public:
    __aicore__ inline DupByRows() {}
    __aicore__ inline void init(GM_ADDR src, GM_ADDR dst, int64_t *src_ne_ub, 
                                size_t *src_nb_ub,  int64_t *dst_ne_ub, 
                                size_t *dst_nb_ub) {
        // Input has four dims.
        int64_t op_block_num = GetBlockNum();
        int64_t op_block_idx = GetBlockIdx();

        // param
        num_rows = src_ne_ub[1] * src_ne_ub[2] * src_ne_ub[3];
        num_elem = src_ne_ub[0];
        
        idx_0 = op_block_idx / (src_ne_ub[1] * src_ne_ub[2]);
        idx_1 = (op_block_idx - idx_0 * (src_ne_ub[1] * src_ne_ub[2])) 
                 / (src_ne_ub[2]);
        idx_2 = op_block_idx - idx_0 * (src_ne_ub[1] * src_ne_ub[2]) 
                - idx_1 * src_ne_ub[2];
                
        src_stride = src_nb_ub[3] * idx_0 + src_nb_ub[2] * idx_1
                     + src_nb_ub[1] * idx_2;

        dst_stride = dst_nb_ub[3] * idx_0 + dst_nb_ub[2] * idx_1
                     + dst_nb_ub[1] * idx_2;
        
        src_gm.SetGlobalBuffer((__gm__ float_t*)(src + src_stride));
        dst_gm.SetGlobalBuffer((__gm__ float_t*)(dst + dst_stride));

        pipe.InitBuffer(src_queue, BUFFER_NUM, (sizeof(float_t) * num_elem + 32 - 1) / 32 * 32);
        pipe.InitBuffer(dst_queue, BUFFER_NUM, (sizeof(float_t) * num_elem + 32 - 1) / 32 * 32);
    }

    __aicore__ inline void copy_in() {
        LocalTensor<float_t> src_local = src_queue.AllocTensor<float_t>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_elem * sizeof(float_t);
        DataCopyPadExtParams<float_t> padParams;
        DataCopyPad(src_local, src_gm, dataCopyParams, padParams);
        
        src_queue.EnQue(src_local);
    }

    __aicore__ inline void copy_out() {
        LocalTensor<float_t> dst_local = dst_queue.DeQue<float_t>();
        
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_elem * sizeof(float_t);
        DataCopyPad(dst_gm, dst_local, dataCopyParams);

        dst_queue.FreeTensor(dst_local);
    }

    __aicore__ inline void dup() {
        copy_in();
        
        LocalTensor<float_t> src_local = src_queue.DeQue<float_t>();
        LocalTensor<float_t> dst_local = dst_queue.AllocTensor<float_t>();
        int32_t BLOCK_NUM = 32 / sizeof(float_t);
        DataCopy(dst_local, src_local, (num_elem + BLOCK_NUM - 1) 
                                            / BLOCK_NUM * BLOCK_NUM);
        dst_queue.EnQue<float_t>(dst_local);

        //
        src_queue.FreeTensor(src_local);
        copy_out();
    }

   private:
  
    TPipe pipe;
    GlobalTensor<float_t> src_gm;
    GlobalTensor<float_t> dst_gm;

    int32_t num_rows;
    int32_t num_elem;
    int32_t idx_0;
    int32_t idx_1;
    int32_t idx_2;
    int32_t src_stride;
    int32_t dst_stride;
    
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

extern "C" __global__ __aicore__ void ascendc_dup_by_rows(GM_ADDR src_gm,
                                                          GM_ADDR dst_gm,
                                                          GM_ADDR src_ne_gm,
                                                          GM_ADDR src_nb_gm,
                                                          GM_ADDR dst_ne_gm,
                                                          GM_ADDR dst_nb_gm) {

    int64_t src_ne_ub[4];
    size_t src_nb_ub[4];
    int64_t dst_ne_ub[4];
    size_t dst_nb_ub[4];

    copy_to_ub(src_ne_gm, src_ne_ub, 32);
    copy_to_ub(src_nb_gm, src_nb_ub, 32);
    copy_to_ub(dst_ne_gm, dst_ne_ub, 32);
    copy_to_ub(dst_nb_gm, dst_nb_ub, 32);

    DupByRows op;
    op.init(src_gm, dst_gm, src_ne_ub, src_nb_ub, dst_ne_ub, dst_nb_ub);
    op.dup(); 
}