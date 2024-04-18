#include "aclnn_ops.h"

#include <aclnnop/aclnn_avgpool2d.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_constant_pad_nd.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_group_norm.h>
#include <aclnnop/aclnn_layer_norm.h>
#include <aclnnop/aclnn_max_pool.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_repeat.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_tril.h>
#include <aclnnop/aclnn_triu.h>
#include <aclnnop/aclnn_upsample_nearest_2d.h>
#include <float.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/ascendc_kernels.h"

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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_negative_slope));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void aclnn_concat(ggml_backend_cann_context& ctx, aclTensor *acl_src0,
                  aclTensor *acl_src1, aclTensor *acl_dst, int64_t concat_dim,
                  ggml_tensor* bind_tensor) {

    aclTensor* tensors[] = {acl_src0, acl_src1};
    aclTensorList* tensorList = aclCreateTensorList(tensors, 2);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // dims in llama.cpp is reversed.
    ACL_CHECK(aclnnCatGetWorkspaceSize(tensorList, concat_dim, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnCat(workspaceAddr, workspaceSize, executor, main_stream));

    //ACL_CHECK(aclDestroyTensor(acl_src0));
    //ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensorList(tensorList));
    ACL_CHECK(aclDestroyTensor(acl_dst));

}

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclTensor* acl_src0 = create_acl_tensor(src0);
    aclTensor* acl_src1 = create_acl_tensor(src1);
    aclTensor* acl_dst = create_acl_tensor(dst);

    int64_t concat_dim = 1;

    aclnn_concat(ctx, acl_src0, acl_src1, acl_dst, concat_dim, dst);

    // release acl_src0, acl_src1 in aclnn_concat
    // ACL_CHECK(aclDestroyTensor(acl_src0));
    // ACL_CHECK(aclDestroyTensor(acl_src1));
    // ->
    // ACL_CHECK(aclDestroyTensorList(tensorList));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void aclnn_arange(ggml_backend_cann_context& ctx, aclTensor *acl_dst,
                  float start, float stop, float step, int64_t n_elements,
                  ggml_tensor* bind_tensor) {

    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(n_elements == steps);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnArange(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
}

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_dst = create_acl_tensor(dst);

    int64_t n_elements = ggml_nelements(dst);
    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));

    aclnn_arange(ctx, acl_dst, start, stop, step, n_elements, dst);
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
    void* buffer = ctx.alloc_buffer(dst, ggml_nelements(dst) * sizeof(int64_t));
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnArgsort(workspaceAddr, workspaceSize, executor, main_stream));

    workspaceSize = 0;
    ACL_CHECK(aclnnCastGetWorkspaceSize(tmp_tensor, type_mapping(dst->type),
                                        acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
    void* buffer = ctx.alloc_buffer(dst, n_bytes * 2);
    aclTensor* acl_mean_out =
        create_acl_tensor(buffer, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);
    aclTensor* acl_rstd_out = create_acl_tensor(
        (char*)buffer + n_bytes, ACL_FLOAT, type_size, ne, nb, ACL_FORMAT_ND);

    ACL_CHECK(aclnnGroupNormGetWorkspaceSize(
        acl_src, nullptr, nullptr, N, C, HxW, n_groups, eps, acl_dst,
        acl_mean_out, acl_rstd_out, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
    void* buffer = ctx.alloc_buffer(dst, n_bytes);
    aclTensor* temp_tensor =
        create_acl_tensor(buffer, ACL_FLOAT, ggml_type_size(src0->type),
                          src0->ne, src0->nb, GGML_MAX_DIMS);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclnnMulsGetWorkspaceSize(acl_src0, acl_scale, temp_tensor, &workspaceSize,
                              &executor);
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    aclnnMuls(workspaceAddr, workspaceSize, executor, stream);

    ACL_CHECK(aclnnSoftmaxGetWorkspaceSize(temp_tensor, 3, acl_dst,
                                           &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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
            workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
        }
        ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, stream));
        ACL_CHECK(aclDestroyTensor(acl_src0));
    } else {
        ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src1, alpha,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
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

    int64_t reduce_dims_host[] = {3};
    aclIntArray* reduce_dims = aclCreateIntArray(reduce_dims_host, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnReduceSumGetWorkspaceSize(acl_src, reduce_dims, true,
                                             type_mapping(src->type), acl_dst,
                                             &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnReduceSum(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void ggml_cann_upsample_nearest2d(ggml_backend_cann_context& ctx,
                                  ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src =
        create_acl_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        create_acl_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    const int scale_factor = dst->op_params[0];
    std::vector<int64_t> output_size{dst->ne[1], dst->ne[0]};
    auto output_size_array = aclCreateIntArray(output_size.data(), 2);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnUpsampleNearest2dGetWorkspaceSize(
        acl_src, output_size_array, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(
        aclnnUpsampleNearest2d(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyIntArray(output_size_array));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void aclnn_pad(ggml_backend_cann_context& ctx, ggml_tensor* dst,
               aclTensor* acl_src, aclTensor* acl_dst, int64_t* paddings,
               float value = 0.0f) {
    aclIntArray* acl_pad = aclCreateIntArray(paddings, GGML_MAX_DIMS * 2);
    aclScalar* acl_value = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnConstantPadNdGetWorkspaceSize(
        acl_src, acl_pad, acl_value, acl_dst, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(
        aclnnConstantPadNd(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyIntArray(acl_pad));
    ACL_CHECK(aclDestroyScalar(acl_value));
}

void ggml_cann_pad(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    int64_t paddings[] = {
        0, dst->ne[0] - src->ne[0], 0, dst->ne[1] - src->ne[1],
        0, dst->ne[2] - src->ne[2], 0, dst->ne[3] - src->ne[3]};
    aclnn_pad(ctx, dst, acl_src, acl_dst, paddings);

    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_src));
}

void ggml_cann_pool2d(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_cann_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_cann_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ASSERT(false);
            break;
    }
}

void ggml_cann_avg_pool2d(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        create_acl_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        create_acl_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);

    // params
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    std::vector<int64_t> padding_avg_dims = {p1, p0};  // h, w
    std::vector<int64_t> padding_max_dims = {p1, p0, 0, 0};

    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_avg = aclCreateIntArray(padding_avg_dims.data(), 2);

    bool ceil_mode = false;  //
    bool count_include_pad = true;
    int64_t divisor_override = 0;
    int8_t cube_math_type = 0;

    // execute op api
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnAvgPool2dGetWorkspaceSize(
        acl_src, kernel_size, strides, paddings_avg, ceil_mode,
        count_include_pad, divisor_override, cube_math_type, acl_dst,
        &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }
    ACL_CHECK(aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, stream));

    // release
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(strides));
    ACL_CHECK(aclDestroyIntArray(paddings_avg));
}

void ggml_cann_max_pool2d(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src =
        create_acl_tensor(src, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    aclTensor* acl_dst =
        create_acl_tensor(dst, nullptr, nullptr, 0, ACL_FORMAT_NCHW);
    // params
    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    int64_t temp_ne[] = {src->ne[0] + p0 * 2, src->ne[1] + p1 * 2, src->ne[2],
                         src->ne[3]};
    size_t temp_nb[GGML_MAX_DIMS];

    temp_nb[0] = ggml_element_size(src);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        temp_nb[i] = temp_nb[i - 1] * temp_ne[i - 1];
    }

    void* buffer =
        ctx.alloc_buffer(dst, ggml_nbytes(src) + p0 * 2 + p1 * 2 * src->nb[1]);
    aclTensor* tmp_tensor =
        create_acl_tensor(buffer, ACL_FLOAT, ggml_element_size(src), temp_ne,
                          temp_nb, GGML_MAX_DIMS, ACL_FORMAT_NCHW);

    // pad
    int64_t paddings[] = {p0, p0, p1, p1, 0, 0, 0, 0};
    float value = -FLT_MAX;
    aclnn_pad(ctx, dst, acl_src, tmp_tensor, paddings, value);

    // max_pool
    std::vector<int64_t> kernel_dims = {k1, k0};
    std::vector<int64_t> stride_dims = {s1, s0};
    std::vector<int64_t> padding_max_dims = {0, 0, 0, 0};
    std::vector<int64_t> dilation_size = {1, 1};
    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);
    auto* paddings_max = aclCreateIntArray(padding_max_dims.data(), 4);
    auto* dilations = aclCreateIntArray(dilation_size.data(), 2);

    bool ceil_mode = false;
    int64_t auto_pads = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnMaxPoolGetWorkspaceSize(
        tmp_tensor, kernel_size, strides, auto_pads, paddings_max, dilations,
        ceil_mode, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(aclnnMaxPool(workspaceAddr, workspaceSize, executor, stream));

    // release
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(tmp_tensor));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(strides));
    ACL_CHECK(aclDestroyIntArray(paddings_max));
    ACL_CHECK(aclDestroyIntArray(dilations));
}

void ggml_cann_dup(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceCopyGetWorkspaceSize(acl_dst, acl_src, &workspaceSize,
                                               &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    aclrtStream stream = ctx.stream();
    ACL_CHECK(aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor* x,
                                         const aclTensor* gamma, double epsilon,
                                         const aclTensor* yOut,
                                         const aclTensor* rstdOout,
                                         uint64_t* workspaceSize,
                                         aclOpExecutor** executor);
aclnnStatus aclnnRmsNorm(void* workspace, uint64_t workspaceSize,
                         aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

aclTensor* aclnn_zero(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                      int64_t* ne, int64_t dims, aclDataType type,
                      size_t type_size) {
    int64_t elements = 1;
    for (int i = 0; i < dims; i++) {
        elements *= ne[i];
    }
    size_t n_bytes = elements * type_size;

    size_t nb[GGML_MAX_DIMS];
    nb[0] = type_size;
    for (int i = 1; i < dims; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    void* buffer = ctx.alloc_buffer(dst, n_bytes);
    ACL_CHECK(aclrtMemsetAsync(buffer, n_bytes, 0, n_bytes, ctx.stream()));
    aclTensor* zero = create_acl_tensor(buffer, type, type_size, ne, nb, dims);
    return zero;
}

aclTensor* aclnn_ones(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                      int64_t* ne, int64_t dims, aclDataType type,
                      size_t type_size, float value = 1.0f) {
    aclTensor* acl_tensor = aclnn_zero(ctx, dst, ne, dims, type, type_size);
    float alpha_host = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_host, aclDataType::ACL_FLOAT);
    aclScalar* other = aclCreateScalar(&value, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceAddsGetWorkspaceSize(acl_tensor, other, alpha,
                                               &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }
    ACL_CHECK(
        aclnnInplaceAdds(workspaceAddr, workspaceSize, executor, ctx.stream()));

    return acl_tensor;
}

void ggml_cann_rms_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps > 0.0f);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclTensor* acl_gamma = aclnn_ones(
        ctx, dst, src->ne, 1, type_mapping(src->type), ggml_element_size(src));

    int64_t rstd_ne[] = {1, src->ne[1], src->ne[2], src->ne[3]};
    aclTensor* acl_rstd =
        aclnn_zero(ctx, dst, rstd_ne, GGML_MAX_DIMS, type_mapping(src->type),
                   ggml_element_size(src));

    ACL_CHECK(aclnnRmsNormGetWorkspaceSize(
        acl_src, acl_gamma, eps, acl_dst, acl_rstd, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(
        aclnnRmsNorm(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyTensor(acl_gamma));
    ACL_CHECK(aclDestroyTensor(acl_rstd));
}

// TODO: performace is low.
void ggml_cann_diag_mask(ggml_backend_cann_context& ctx, ggml_tensor* dst,
                         float value) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    const int n_past = ((int32_t*)dst->op_params)[0];

    aclTensor* mask_tensor =
        aclnn_ones(ctx, dst, src->ne, GGML_MAX_DIMS, type_mapping(src->type),
                   ggml_element_size(src), value);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceTriuGetWorkspaceSize(mask_tensor, n_past + 1,
                                               &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(
        aclnnInplaceTriu(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclnnTrilGetWorkspaceSize(acl_src, n_past + 1, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(aclnnTril(workspaceAddr, workspaceSize, executor, ctx.stream()));

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, mask_tensor, alpha,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }
    ACL_CHECK(
        aclnnInplaceAdd(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyTensor(mask_tensor));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}

void aclnn_cast(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                aclTensor* acl_dst, aclDataType cast_data_type,
                ggml_tensor* bind_tensor) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnCastGetWorkspaceSize(acl_src, cast_data_type, acl_dst,
                                        &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnCast(workspaceAddr, workspaceSize, executor, stream));
}

void aclnn_permute(ggml_backend_cann_context& ctx, aclTensor* acl_src,
                   aclTensor* acl_dst, int64_t* new_dim, uint64_t dims,
                   ggml_tensor* bind_tensor) {
    aclIntArray* acl_dims = aclCreateIntArray(new_dim, dims);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnPermuteGetWorkspaceSize(acl_src, acl_dims, acl_dst,
                                           &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(
        aclnnPermute(workspaceAddr, workspaceSize, executor, ctx.stream()));

    ACL_CHECK(aclDestroyIntArray(acl_dims));
}

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor* self,
                                        const aclIntArray* kernelSize,
                                        const aclIntArray* dilation,
                                        const aclIntArray* padding,
                                        const aclIntArray* stride,
                                        aclTensor* out, uint64_t* workspaceSize,
                                        aclOpExecutor** executor);
aclnnStatus aclnnIm2col(void* workspace, uint64_t workspaceSize,
                        aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
void ggml_cann_im2col(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];  // kernel
    ggml_tensor* src1 = dst->src[1];  // input

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];
    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int64_t N = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N,C,H,W] -> [N, IC * KH * KW, OW * OH]
    aclTensor* acl_src1 = create_acl_tensor(src1);
    int64_t tmp_im2col_ne[] = {OW * OH, IC * KH * KW, N};
    size_t tmp_im2col_nb[GGML_MAX_DIMS - 1];

    tmp_im2col_nb[0] = ggml_type_size(src1->type);
    for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
        tmp_im2col_nb[i] = tmp_im2col_nb[i - 1] * tmp_im2col_ne[i - 1];
    }

    // Calculate im2col.
    // If dst is f16, tmp_buffer is f32, we need alloc src.typesize *
    // dst.elemcount.
    void* tmp_im2col_buffer =
        ctx.alloc_buffer(dst, ggml_nelements(dst) * ggml_element_size(src1));
    aclTensor* tmp_im2col_tensor = create_acl_tensor(
        tmp_im2col_buffer, type_mapping(src1->type), ggml_type_size(src1->type),
        tmp_im2col_ne, tmp_im2col_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);

    std::vector<int64_t> kernel_dims = {KH, KW};
    std::vector<int64_t> dilation_size = {d1, d0};
    std::vector<int64_t> padding_dims = {p1, p0};
    std::vector<int64_t> stride_dims = {s1, s0};
    auto* kernel_size = aclCreateIntArray(kernel_dims.data(), 2);
    auto* dilations = aclCreateIntArray(dilation_size.data(), 2);
    auto* paddings = aclCreateIntArray(padding_dims.data(), 2);
    auto* strides = aclCreateIntArray(stride_dims.data(), 2);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    aclrtStream stream = ctx.stream();

    ACL_CHECK(aclnnIm2colGetWorkspaceSize(acl_src1, kernel_size, dilations,
                                          paddings, strides, tmp_im2col_tensor,
                                          &workspaceSize, &executor));

    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(dst, workspaceSize);
    }

    ACL_CHECK(aclnnIm2col(workspaceAddr, workspaceSize, executor, stream));

    // Cast if dst is f16.
    aclTensor* tmp_cast_tensor = nullptr;
    if (src1->type != dst->type) {
        void* tmp_cast_buffer = ctx.alloc_buffer(dst, ggml_nbytes(dst));
        size_t temp_cast_nb[GGML_MAX_DIMS - 1];
        temp_cast_nb[0] = ggml_type_size(dst->type);
        for (int i = 1; i < GGML_MAX_DIMS - 1; i++) {
            temp_cast_nb[i] = temp_cast_nb[i - 1] * tmp_im2col_ne[i - 1];
        }

        tmp_cast_tensor = create_acl_tensor(
            tmp_cast_buffer, type_mapping(dst->type), ggml_type_size(dst->type),
            tmp_im2col_ne, temp_cast_nb, GGML_MAX_DIMS - 1, ACL_FORMAT_ND);
        aclnn_cast(ctx, tmp_im2col_tensor, tmp_cast_tensor,
                   type_mapping(dst->type), dst);
    }

    // Permute: [N, IC * KH * KW, OW * OH] -> [N, OW * OH, IC * KH * KW]
    int64_t dst_ne[] = {dst->ne[0], dst->ne[1] * dst->ne[2], dst->ne[3]};
    size_t dst_nb[] = {dst->nb[0], dst->nb[1], dst->nb[3]};
    aclTensor* acl_dst =
        create_acl_tensor(dst, dst_ne, dst_nb, GGML_MAX_DIMS - 1);

    int64_t permute_dim[] = {0, 2, 1};
    if (src1->type != dst->type) {
        aclnn_permute(ctx, tmp_cast_tensor, acl_dst, permute_dim, 3, dst);
    } else {
        aclnn_permute(ctx, tmp_im2col_tensor, acl_dst, permute_dim, 3, dst);
    }

    // release
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(tmp_im2col_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_cast_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
    ACL_CHECK(aclDestroyIntArray(kernel_size));
    ACL_CHECK(aclDestroyIntArray(dilations));
    ACL_CHECK(aclDestroyIntArray(paddings));
    ACL_CHECK(aclDestroyIntArray(strides));
}

void aclnn_exp(ggml_backend_cann_context& ctx, aclTensor *acl_src,
                       ggml_tensor* bind_tensor) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceExpGetWorkspaceSize(acl_src, &workspaceSize,
                                              &executor));
    if(workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnInplaceExp(workspaceAddr, workspaceSize, executor,
                              ctx.stream()));

}

void aclnn_muls(ggml_backend_cann_context& ctx, aclTensor *acl_src, float scale,
                ggml_tensor* bind_tensor) {

    aclScalar* acl_scale = aclCreateScalar(&scale, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceMulsGetWorkspaceSize(acl_src, acl_scale,
                                               &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnInplaceMuls(workspaceAddr, workspaceSize, executor,
                               ctx.stream()));

    ACL_CHECK(aclDestroyScalar(acl_scale));
}

void aclnn_inplace_mul(ggml_backend_cann_context& ctx, aclTensor *acl_src,
               aclTensor *acl_other, ggml_tensor* bind_tensor) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnInplaceMulGetWorkspaceSize(acl_src, acl_other,
                                              &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnInplaceMul(workspaceAddr, workspaceSize, executor,
                              ctx.stream()));
}

void aclnn_noinplcace_mul(ggml_backend_cann_context& ctx, aclTensor *acl_src,
               aclTensor *acl_other, aclTensor *acl_dst, ggml_tensor* bind_tensor) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnMulGetWorkspaceSize(acl_src, acl_other, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnMul(workspaceAddr, workspaceSize, executor,
                              ctx.stream()));
}

void aclnn_cos(ggml_backend_cann_context& ctx, aclTensor *acl_src,
               aclTensor *acl_dst, ggml_tensor* bind_tensor) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnCosGetWorkspaceSize(acl_src, acl_dst, &workspaceSize,
                                       &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnCos(workspaceAddr, workspaceSize, executor,
                       ctx.stream()));
}

void aclnn_sin(ggml_backend_cann_context& ctx, aclTensor *acl_src,
               aclTensor *acl_dst, ggml_tensor* bind_tensor) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnSinGetWorkspaceSize(acl_src, acl_dst, &workspaceSize,
                                       &executor));
    if (workspaceSize > 0) {
        workspaceAddr = ctx.alloc_buffer(bind_tensor, workspaceSize);
    }

    ACL_CHECK(aclnnSin(workspaceAddr, workspaceSize, executor,
                       ctx.stream()));
}

void ggml_cann_timestep_embedding(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];
    int half = dim / 2;

    aclTensor* acl_src = create_acl_tensor(src);

    // arange: [0, ..., half)
    float start = 0;
    float stop = half;
    float step = 1;
    int64_t n_elements_arange = half;
    int64_t tmp_arange_ne[] = {half};
    size_t tmp_arange_nb[] = {sizeof(dst->type)};

    void* tmp_arange_buffer = ctx.alloc_buffer(dst, half * sizeof(dst->type));
    aclTensor* tmp_arange_tensor = create_acl_tensor(tmp_arange_buffer,
                                                     type_mapping(dst->type),
                                                     ggml_type_size(dst->type),
                                                     tmp_arange_ne,
                                                     tmp_arange_nb,
                                                     GGML_MAX_DIMS-3,
                                                     ACL_FORMAT_ND);

    aclnn_arange(ctx, tmp_arange_tensor, start, stop, step, n_elements_arange,
                 dst);

    // freq
    float freq_param = -logf(max_period) / half;
    aclnn_muls(ctx, tmp_arange_tensor, freq_param, dst);
    aclnn_exp(ctx, tmp_arange_tensor, dst);

    // permute: src [0,1,2,3]->[0,1,3,2]
    int64_t tmp_permute_ne[] = {src->ne[1], src->ne[0], src->ne[2], src->ne[3]};
    size_t tmp_permute_nb[GGML_MAX_DIMS];
    tmp_permute_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_permute_nb[i] = tmp_permute_nb[i-1] * tmp_permute_ne[i-1];
    }

    void* tmp_permute_buffer = ctx.alloc_buffer(dst, ggml_nbytes(src)*320);
    aclTensor* tmp_permute_tenosr = create_acl_tensor(tmp_permute_buffer,
                                                      type_mapping(src->type),
                                                      ggml_type_size(src->type),
                                                      tmp_permute_ne,
                                                      tmp_permute_nb,
                                                      GGML_MAX_DIMS,
                                                      ACL_FORMAT_ND);
    int64_t permute_dim[] = {0, 1, 3, 2};
    int64_t num_dims = 4;
    aclnn_permute(ctx, acl_src, tmp_permute_tenosr, permute_dim, num_dims, dst);

    // timestep * freq
    int64_t tmp_mul_ne[] = {src->ne[1] * half, src->ne[0], src->ne[2],
                            src->ne[3]};
    size_t tmp_mul_nb[GGML_MAX_DIMS];
    tmp_mul_nb[0] = ggml_type_size(src->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tmp_mul_nb[i] = tmp_mul_nb[i-1] * tmp_mul_ne[i-1];
    }

    int mul_nelements = src->ne[1] * half * src->ne[0] * src->ne[2] *
                        src->ne[3];

    void* tmp_mul_buffer = ctx.alloc_buffer(dst, mul_nelements *
                                            ggml_type_size(src->type));
    aclTensor* tmp_mul_tensor = create_acl_tensor(tmp_mul_buffer,
                                                  type_mapping(src->type),
                                                  ggml_type_size(src->type),
                                                  tmp_mul_ne,
                                                  tmp_mul_nb,
                                                  GGML_MAX_DIMS,
                                                  ACL_FORMAT_ND);
    aclnn_noinplcace_mul(ctx, tmp_permute_tenosr, tmp_arange_tensor,
                         tmp_mul_tensor, dst);

    // cos
    void* tmp_cos_buffer = ctx.alloc_buffer(dst, mul_nelements *
                                            ggml_type_size(src->type));
    aclTensor* tmp_cos_tensor = create_acl_tensor(tmp_cos_buffer,
                                                  type_mapping(dst->type),
                                                  ggml_type_size(dst->type),
                                                  tmp_mul_ne,
                                                  tmp_mul_nb,
                                                  GGML_MAX_DIMS,
                                                  ACL_FORMAT_ND);

    aclnn_cos(ctx, tmp_mul_tensor, tmp_cos_tensor, dst);

    // sin
    void* tmp_sin_buffer = ctx.alloc_buffer(dst, mul_nelements *
                                            ggml_type_size(src->type));
    aclTensor* tmp_sin_tensor = create_acl_tensor(tmp_sin_buffer,
                                                  type_mapping(dst->type),
                                                  ggml_type_size(dst->type),
                                                  tmp_mul_ne,
                                                  tmp_mul_nb,
                                                  GGML_MAX_DIMS,
                                                  ACL_FORMAT_ND);

    aclnn_sin(ctx, tmp_mul_tensor, tmp_sin_tensor, dst);

    // concat
    int64_t concat_dim = 3;
    aclTensor* acl_dst = create_acl_tensor(dst);
    aclnn_concat(ctx, tmp_cos_tensor, tmp_sin_tensor, acl_dst, concat_dim, dst);

    // release
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(tmp_arange_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_permute_tenosr));
    ACL_CHECK(aclDestroyTensor(tmp_mul_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_cos_tensor));
    ACL_CHECK(aclDestroyTensor(tmp_sin_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));
}