#include "aclnn_ops.h"

#include <aclnnop/aclnn_batch_norm.h>
#include <aclnnop/aclnn_cast.h>

#include <cmath>
#include <cstring>
#include <vector>

// TODO: repeat is implemented through add to apply bcast. Optimize it.
void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    size_t nbytes = ggml_nbytes(dst);
    aclrtStream main_stream = ctx.stream();
    // Set dst to a zero tensor.
    ACL_CHECK(aclrtMemsetAsync(dst->data, nbytes, 0, nbytes, main_stream));

    aclTensor* acl_src;
    aclTensor* acl_dst;

    // Short cut for same shape.
    if (ggml_are_same_shape(src, dst)) {
        ACL_CHECK(aclrtMemcpyAsync(dst->data, nbytes, src->data, nbytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, main_stream));
    } else {
        if (need_bcast(dst, src)) {
            BCAST_SHAPE(dst, src);
            acl_dst = create_acl_tensor(dst, BCAST_PARAM(dst));
            acl_src = create_acl_tensor(src, BCAST_PARAM(src));
        } else {
            acl_dst = create_acl_tensor(dst);
            acl_src = create_acl_tensor(src);
        }

        // Add src0 to dst.
        aclScalar* alpha = nullptr;
        int alphaValue = 1;
        alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_INT32);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        void* workspaceAddr = nullptr;

        ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src, alpha,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                                  ACL_MEM_MALLOC_HUGE_FIRST));
        }

        ACL_CHECK(aclnnInplaceAdd(workspaceAddr, workspaceSize, executor,
                                  main_stream));

        ACL_CHECK(aclDestroyScalar(alpha));
        ACL_CHECK(aclDestroyTensor(acl_src));
        ACL_CHECK(aclDestroyTensor(acl_dst));

        if (workspaceSize > 0) {
            ACL_CHECK(aclrtFree(workspaceAddr));
        }
    }
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
    // TODO, workspace should free after sync. Add alloc memory to
    // backend_buffer.
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
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
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_negative_slope));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
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

    ACL_CHECK(aclnnCatGetWorkspaceSize(tensorList, 2, acl_dst, &workspaceSize,
                                       &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnCat(workspaceAddr, workspaceSize, executor, main_stream));

    aclDestroyTensorList(tensorList);
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
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
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnArange(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
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
    if (workspaceSize > 0)
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnClamp(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_min));
    ACL_CHECK(aclDestroyScalar(acl_max));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
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
    if (workspaceSize > 0)
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnMuls(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(scale));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);
    void* buffer = nullptr;
    ACL_CHECK(aclrtMalloc(
        &buffer, ggml_nbytes(dst) / ggml_type_size(dst->type) * sizeof(int64_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
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
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnArgsort(workspaceAddr, workspaceSize, executor, main_stream));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
        workspaceSize = 0;
    }

    ACL_CHECK(aclnnCastGetWorkspaceSize(tmp_tensor, type_mapping(dst->type),
                                        acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnCast(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(tmp_tensor));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    // TODO: optimize argsort kernel or free tmp buffers after stream sync.
    ACL_CHECK(aclrtSynchronizeStream(main_stream));
    ACL_CHECK(aclrtFree(buffer));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

void ggml_cann_norm(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    float *weight_host, *bias_host;
    int64_t channel = dst->ne[2];

    weight_host = new float[channel];
    bias_host = new float[channel];

    for (int i = 0; i < channel; i++) {
        weight_host[i] = 1;
        bias_host[i] = 0;
    }

    aclrtStream stream = ctx.stream();

    // Input tensors.
    void *buffer, *acl_weight, *acl_bias, *acl_mean, *acl_invstd;
    ACL_CHECK(aclrtMalloc(&buffer, 4 * channel * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST));
    acl_weight = buffer;
    acl_bias = acl_weight + sizeof(float) * channel;
    acl_mean = acl_bias + sizeof(float) * channel;
    acl_invstd = acl_mean + sizeof(float) * channel;

    // Set input params.
    ACL_CHECK(aclrtMemcpyAsync(acl_weight, channel, weight_host, channel,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    ACL_CHECK(aclrtMemcpyAsync(acl_bias, channel, bias_host, channel,
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    delete[] weight_host;
    delete[] bias_host;

    // Create input tensors.
    int64_t input_tensor_shape[] = {channel};
    size_t input_tensor_stride[] = {1};
    aclTensor* weight =
        create_acl_tensor(acl_weight, ACL_FLOAT, sizeof(float),
                          input_tensor_shape, input_tensor_stride, 1);
    aclTensor* bias =
        create_acl_tensor(acl_bias, ACL_FLOAT, sizeof(float),
                          input_tensor_shape, input_tensor_stride, 1);
    aclTensor* mean =
        create_acl_tensor(acl_mean, ACL_FLOAT, sizeof(float),
                          input_tensor_shape, input_tensor_stride, 1);
    aclTensor* invstd =
        create_acl_tensor(acl_invstd, ACL_FLOAT, sizeof(float),
                          input_tensor_shape, input_tensor_stride, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnBatchNormGetWorkspaceSize(
        acl_src, weight, bias, nullptr, nullptr, false, 0, eps, acl_dst, mean,
        invstd, &workspaceSize, &executor));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ACL_CHECK(aclnnBatchNorm(workspaceAddr, workspaceSize, executor, stream));

    ACL_CHECK(aclDestroyTensor(weight));
    ACL_CHECK(aclDestroyTensor(bias));
    ACL_CHECK(aclDestroyTensor(mean));
    ACL_CHECK(aclDestroyTensor(invstd));

    // TODO: optimize argsort kernel or free tmp buffers after stream sync.
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtFree(buffer));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}