#include "aclnn_ops.h"

#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_group_norm.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_repeat.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_reduce_sum.h>

#include <cmath>
#include <cstring>
#include <vector>

void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    size_t nbytes = ggml_nbytes(dst);
    aclrtStream main_stream = ctx.stream();
    // Set dst to a zero tensor.
    ACL_CHECK(aclrtMemsetAsync(dst->data, nbytes, 0, nbytes, main_stream));

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    int64_t repeatsArray[] = {dst->ne[3] / src->ne[3], dst->ne[2] / src->ne[2],
                              dst->ne[1] / src->ne[1], dst->ne[0] / src->ne[0]};

    aclIntArray* repeats = aclCreateIntArray(repeatsArray, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnRepeatGetWorkspaceSize(acl_src, repeats, acl_dst,
                                          &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnRepeat(workspaceAddr, workspaceSize, executor, stream));
    ACL_CHECK(aclDestroyIntArray(repeats));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_add(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    aclTensor* acl_src0;
    aclTensor* acl_src1;
    aclTensor* acl_dst;

    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = create_acl_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = create_acl_tensor(src1, BCAST_PARAM(src1));
        acl_dst = create_acl_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = create_acl_tensor(src0);
        acl_src1 = create_acl_tensor(src1);
        acl_dst = create_acl_tensor(dst);
    }

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    aclScalar* acl_negative_slope =
        aclCreateScalar(&negative_slope, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnLeakyReluGetWorkspaceSize(
        acl_src, acl_negative_slope, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_negative_slope));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclTensor* acl_src0 = create_acl_tensor(src0);
    aclTensor* acl_src1 = create_acl_tensor(src1);
    aclTensor* acl_dst = create_acl_tensor(dst);

    std::vector<aclTensor*> tmp{acl_src0, acl_src1};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // dim1 == ne2, dims in llama.cpp is reversed.
    ACL_CHECK(aclnnCatGetWorkspaceSize(tensorList, 1, acl_dst, &workspaceSize,
                                       &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnCat(workspaceAddr, workspaceSize, executor, main_stream));

    aclDestroyTensorList(tensorList);
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));

    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnArange(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_sqr(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
}

void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    aclScalar* acl_min = aclCreateScalar(&min, aclDataType::ACL_FLOAT);
    aclScalar* acl_max = aclCreateScalar(&max, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnClampGetWorkspaceSize(acl_src, acl_min, acl_max, acl_dst,
                                         &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnClamp(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_min));
    ACL_CHECK(aclDestroyScalar(acl_max));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_scale(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    aclScalar* scale = aclCreateScalar(&v, aclDataType::ACL_FLOAT);
    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMulsGetWorkspaceSize(acl_src, scale, acl_dst, &workspaceSize,
                                        &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnMuls(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(scale));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);
    void* buffer = ctx.alloc_buffer(
        ggml_nbytes(dst) / ggml_type_size(dst->type) * sizeof(int64_t));
    aclTensor* tmp_tensor =
        create_acl_tensor(buffer, ACL_INT64, ggml_type_size(dst->type), dst->ne,
                          dst->nb, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnArgsortGetWorkspaceSize(
        acl_src, -1, (order == GGML_SORT_ORDER_DESC ? true : false), tmp_tensor,
        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnArgsort(workspaceAddr, workspaceSize, executor, main_stream));

    workspaceSize = 0;
    ACL_CHECK(aclnnCastGetWorkspaceSize(tmp_tensor, type_mapping(dst->type),
                                        acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    ACL_CHECK(aclnnCast(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(tmp_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    std::vector<int64_t> normData = {dst->ne[0]};
    aclIntArray* norm = aclCreateIntArray(normData.data(), normData.size());
    ACL_CHECK(aclnnLayerNormGetWorkspaceSize(acl_src, norm, nullptr, nullptr,
                                             eps, acl_dst, nullptr, nullptr,
                                             &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnLayerNorm(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyIntArray(norm));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_group_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    const float eps = 1e-6f;  // TODO: make this a parameter
    int n_groups = dst->op_params[0];

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    int64_t N = src->ne[3];
    int64_t C = src->ne[2];
    int64_t HxW = src->ne[1] * src->ne[0];

    size_t type_size = ggml_type_size(src->type);
    int64_t ne[] = {n_groups, N};
    size_t nb[] = {type_size, type_size * n_groups};
    size_t n_bytes = N * n_groups;
    void* buffer = ctx.alloc_buffer(n_bytes * 2);
    aclTensor* acl_mean_out =
        create_acl_tensor(buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    aclTensor* acl_rstd_out = create_acl_tensor(
        (char*)buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    ACL_CHECK(aclnnGroupNormGetWorkspaceSize(
        acl_src, nullptr, nullptr, N, C, HxW, n_groups, eps, acl_dst,
        acl_mean_out, acl_rstd_out, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnGroupNorm(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_mean_out));
    ACL_CHECK(aclDestroyTensor(acl_rstd_out));
}

// TODO: need alibi.
void ggml_cann_softmax(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[0];

    aclTensor* acl_src0 = create_acl_tensor(src0);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float scale = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);
    aclScalar* acl_max_bias =
        aclCreateScalar(&max_bias, aclDataType::ACL_FLOAT);

    size_t n_bytes = ggml_nbytes(src0);
    void* buffer = ctx.alloc_buffer(n_bytes);
    aclTensor* temp_tensor =
        create_acl_tensor(buffer, ACL_FLOAT, ggml_type_size(src0->type),
                          src0->ne, src0->nb, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnMulsGetWorkspaceSize(acl_src0, acl_scale, temp_tensor, &workspaceSize,
                              &executor);
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    aclnnMuls(workspaceAddr, workspaceSize, executor, stream);

    ACL_CHECK(aclnnSoftmaxGetWorkspaceSize(temp_tensor, 3, acl_dst,
                                           &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    ACL_CHECK(aclnnSoftmax(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_acc(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    size_t nb1 = ((int32_t*)dst->op_params)[0];
    size_t nb2 = ((int32_t*)dst->op_params)[1];
    size_t nb3 = ((int32_t*)dst->op_params)[2];
    size_t offset = ((int32_t*)dst->op_params)[3];
    bool inplace = (bool)((int32_t*)dst->op_params)[4];

    size_t param_nb[] = {ggml_element_size(src0), nb1, nb2, nb3};

    aclTensor* acl_dst = create_acl_tensor(
        dst, src1->ne, param_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
    aclTensor* acl_src1 = create_acl_tensor(src1);

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclrtStream stream = ctx.stream();

    if (!inplace) {
        size_t cpy_size = ggml_nbytes(dst);
        ACL_CHECK(aclrtMemcpyAsync(dst->data, cpy_size, src0->data, cpy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        aclTensor* acl_src0 = create_acl_tensor(
            src0, src1->ne, src0->nb, GGML_MAX_DIMS, ACL_FORMAT_ND, offset);
        ACL_CHECK(aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst,
                                           &workspaceSize, &executor));
        if (workspaceSize > 0) {
            workspaceAddr = ctx.alloc_buffer(workspaceSize);
        }
        ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, stream));
        ACL_CHECK(aclDestroyTensor(acl_src0));
    } else {
        ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src1, alpha,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            workspaceAddr = ctx.alloc_buffer(workspaceSize);
        }
        ACL_CHECK(
            aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, stream));
    }

    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_sum_rows(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);

    GGML_ASSERT(dst->ne[0] == 1);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    int64_t reduce_dims_host[] = {3};
    aclIntArray* reduce_dims = aclCreateIntArray(reduce_dims_host, 1);

    ACL_CHECK(aclnnReduceSumGetWorkspaceSize(acl_src, reduce_dims, true,
                                             type_mapping(src->type), acl_dst,
                                             &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnReduceSum(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}