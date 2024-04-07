#ifndef CANN_ACL_TENSOR_H
#define CANN_ACL_TENSOR_H

#include <aclnn/aclnn_base.h>

#include "common.h"

// Broadcast
aclDataType type_mapping(ggml_type type);

aclTensor* create_acl_tensor(const ggml_tensor* tensor,
                             int64_t* bcast_ne = nullptr,
                             size_t* bcast_nb = nullptr, int64_t bcast_dims = 0,
                             aclFormat format = ACL_FORMAT_ND);

aclTensor* create_acl_tensor(void* data_ptr, aclDataType dtype,
                             size_t type_size, int64_t* ne, size_t* nb,
                             int64_t dims, aclFormat format = ACL_FORMAT_ND);

bool need_bcast(const ggml_tensor* t0, const ggml_tensor* t1);

int64_t get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
                        int64_t* bcast_ne_src0, int64_t* bcast_ne_src1,
                        size_t* bcast_nb_src0, size_t* bcast_nb_src1);

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                       \
    int64_t bcast_ne_##src0[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_ne_##src1[GGML_MAX_DIMS * 2];                       \
    size_t bcast_nb_##src0[GGML_MAX_DIMS * 2];                        \
    size_t bcast_nb_##src1[GGML_MAX_DIMS * 2];                        \
    int64_t bcast_dims =                                              \
        get_bcast_shape(src0, src1, bcast_ne_##src0, bcast_ne_##src1, \
                        bcast_nb_##src0, bcast_nb_##src1);

#define BCAST_PARAM(src) bcast_ne_##src, bcast_nb_##src, bcast_dims

#endif  // CANN_ACL_TENSOR_H