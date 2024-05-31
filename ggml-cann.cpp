#include "ggml-cann.h"

#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend-impl.h"
#include "ggml-cann/acl_ops.h"
#include "ggml-cann/aclnn_ops.h"
#include "ggml-cann/common.h"

[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    fprintf(stderr, "CANN error: %s\n", msg);
    fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    fprintf(stderr, "  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ASSERT(!"CANN error");
}

void ggml_cann_set_device(const int32_t device) {
    // TODO: uncomment these lines after empty context has fixed.
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //   return;
    // }
    ACL_CHECK(aclrtSetDevice(device));
}

int32_t ggml_cann_get_device() {
    int32_t id;
    ACL_CHECK(aclrtGetDevice(&id));
    return id;
}

static ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrtGetDeviceCount((uint32_t*)&info.device_count);

    if (err != ACL_SUCCESS) {
        fprintf(stderr, "%s: failed to initialize " GGML_CANN_NAME ": %s\n",
                __func__, aclGetRecentErrMsg());
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);

    // TODO: add more device info later.
    return info;
}

const ggml_cann_device_info& ggml_cann_info() {
    static ggml_cann_device_info info = ggml_cann_init();
    return info;
}

// cann buffer
struct ggml_backend_cann_buffer_context {
    int32_t device;
    void* dev_ptr = nullptr;
    std::string name;
    std::vector<void*> dev_extra_ptrs;

    ggml_backend_cann_buffer_context(int32_t device, void* dev_ptr)
        : device(device),
          dev_ptr(dev_ptr),
          name(GGML_CANN_NAME + std::to_string(device)) {}

    void* get_extra_ptr(size_t size) {
        void* buffer;
        ACL_CHECK(aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST));
        dev_extra_ptrs.push_back(buffer);
        return buffer;
    }

    ~ggml_backend_cann_buffer_context() {
        ACL_CHECK(aclrtFree(dev_ptr));
        for (auto dev_extra_ptr : dev_extra_ptrs) {
            ACL_CHECK(aclrtFree(dev_extra_ptr));
        }
    }
};

GGML_CALL static const char* ggml_backend_cann_buffer_get_name(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cann(
    ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cann_buffer_get_name;
}

GGML_CALL static void ggml_backend_cann_buffer_free_buffer(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    delete ctx;
}

GGML_CALL static void* ggml_backend_cann_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cann_transform_q4_0(ggml_tensor* tensor,
                                                       const void* src,
                                                       void* dst) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);

    size_t n_bytes = ggml_nbytes(tensor);
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t* quant_offset = (uint8_t*)dst;
    uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        block_q4_0* group = (block_q4_0*)((char*)src + i * sizeof(block_q4_0));
        *scale_offset = group->d;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0x0F);
            (*quant_offset) |= ((group->qs[j + 1] << 4));
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] >> 4);
            (*quant_offset) |= (group->qs[j + 1] & 0xF0);
            quant_offset++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    for (quant_offset = (uint8_t*)dst;
         quant_offset < (uint8_t*)dst + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
}

GGML_CALL static void ggml_backend_cann_transform_back_q4_0(
    const ggml_tensor* tensor, void* src, void* dst) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);

    size_t n_bytes = ggml_nbytes(tensor);
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t* quant_offset = (uint8_t*)src;
    uint16_t* scale_offset = (uint16_t*)((char*)src + quant_bytes);

    for (;quant_offset < (uint8_t*)src + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
    quant_offset = (uint8_t*)src;

    for (int i = 0; i < groups; i++) {
        block_q4_0* group = (block_q4_0*)((char*)dst + i * sizeof(block_q4_0));
        group->d = *scale_offset;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] = ((*quant_offset) & 0x0F);
            group->qs[j + 1] = ((*quant_offset) >> 4);
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] |= ((*quant_offset) << 4);
            group->qs[j + 1] |= ((*quant_offset) & 0xF0);
            quant_offset++;
        }
    }
}

GGML_CALL static void ggml_backend_cann_transform_q8_0(ggml_tensor* tensor,
                                                       const void* src,
                                                       void* dst) {
    size_t n_bytes = ggml_nbytes(tensor);
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);

    uint8_t* quant_offset = (uint8_t*)dst;
    uint16_t* scale_offset = (uint16_t*)((char*)dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        block_q8_0* group = (block_q8_0*)((char*)src + i * sizeof(block_q8_0));
        *scale_offset = group->d;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(quant_offset, group->qs, group_quant_size);
        quant_offset += group_quant_size;
    }
}

GGML_CALL static void ggml_backend_cann_transform_back_q8_0(
    const ggml_tensor* tensor, const void* src, void* dst) {
    size_t n_bytes = ggml_nbytes(tensor);
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK8_0;
    size_t quant_bytes = n_elems * sizeof(uint8_t);

    uint8_t* quant_offset = (uint8_t*)src;
    uint16_t* scale_offset = (uint16_t*)((char*)src + quant_bytes);

    for (int i = 0; i < groups; i++) {
        block_q8_0* group = (block_q8_0*)((char*)dst + i * sizeof(block_q8_0));
        group->d = *scale_offset;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(group->qs, quant_offset, group_quant_size);
        quant_offset += group_quant_size;
    }
}

GGML_CALL static void ggml_backend_cann_transform(ggml_tensor* tensor,
                                                  const void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

GGML_CALL static void ggml_backend_cann_transform_back(
    const ggml_tensor* tensor, void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

GGML_CALL static bool need_transform(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static void set_tensor_extra(ggml_backend_buffer_t buffer,
                             ggml_tensor* tensor) {
    // if tensor is need transform, make sure all meta data are copied to
    // npu.
    // TODO: All tensors should copy meta data to npu, but extra is used to
    // record memory usage. Only used for perf test.
    size_t tensor_meta_size = sizeof(ggml_tensor);
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    tensor->extra = ctx->get_extra_ptr(tensor_meta_size);
    ACL_CHECK(aclrtMemcpy(tensor->extra, tensor_meta_size, tensor,
                          tensor_meta_size, ACL_MEMCPY_HOST_TO_DEVICE));
}

GGML_CALL static void ggml_backend_cann_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        set_tensor_extra(buffer, tensor);
        return;
    }

    tensor->backend = GGML_BACKEND_TYPE_GPU;

    // TODO: can backend doesn't support quantized yet. Just leave the code
    // here.
    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size =
            ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((char*)tensor->data + original_size,
                                  memset_size, 0, memset_size));
        }
    }
    set_tensor_extra(buffer, tensor);
}

// TODO: need handle tensor which pas paddings.
GGML_CALL static void ggml_backend_cann_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data,
    size_t offset, size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    // TODO: refer to cann(#6017), it use thread's default stream.
    // For acl, synchronous functions use this default stream.
    // Why aclrtSynchronizeDevice?

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(tensor->data, size, (char*)data + offset, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        void* transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, (char*)data + offset,
                                    transform_buffer);

#ifndef NDEBUG
        void* check_buffer = malloc(size);
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         check_buffer);
        GGML_ASSERT(memcmp((char*)data + offset, check_buffer, size) == 0);
        free(check_buffer);
#endif
        ACL_CHECK(aclrtMemcpy(tensor->data, size, transform_buffer, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        free(transform_buffer);
    }
}

GGML_CALL static void ggml_backend_cann_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy((char*)data + offset, size, tensor->data, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        void* transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpy(transform_buffer, size, tensor->data, size,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         (char*)data + offset);
        free(transform_buffer);
    }
}

GGML_CALL static bool ggml_backend_cann_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    if (ggml_backend_buffer_is_cann(src->buffer)) {
        GGML_ASSERT(src->backend == GGML_BACKEND_TYPE_GPU);
        GGML_ASSERT(dst->backend == GGML_BACKEND_TYPE_GPU);
        ggml_backend_cann_buffer_context* src_ctx =
            (ggml_backend_cann_buffer_context*)src->buffer->context;
        ggml_backend_cann_buffer_context* dst_ctx =
            (ggml_backend_cann_buffer_context*)buffer->context;

        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        if (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                  (const char*)src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            return true;
        } else {
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device,
                                               dst_ctx->device));
            if (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                      (const char*)src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                return true;
            }
        }
    }
    return false;
}

GGML_CALL static void ggml_backend_cann_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

static ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .get_name        = */ ggml_backend_cann_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

// cann buffer type
struct ggml_backend_cann_buffer_type_context {
    int32_t device;
    std::string name;
};

GGML_CALL static const char* ggml_backend_cann_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    ggml_backend_cann_buffer_type_context* ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t
ggml_backend_cann_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    ggml_cann_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1);

    void* dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        fprintf(
            stderr,
            "%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device,
            aclGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_cann_buffer_context* ctx =
        new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface,
                                    ctx, size);
}

GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);

    // TODO: not support quantized yet.
    // TODO: consider un-continue tensor.
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(
                tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cann_buffer_type_supports_backend(
    ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    if (!ggml_backend_is_cann(backend)) {
        return false;
    }

    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return buft_ctx->device == cann_ctx->device;
}

static ggml_backend_buffer_type_i ggml_backend_cann_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cann_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cann_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cann_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cann_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cann_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cann_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cann_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    static bool ggml_backend_cann_buffer_type_initialized = false;

    if (!ggml_backend_cann_buffer_type_initialized) {
        for (int32_t i = 0; i < GGML_CANN_MAX_DEVICES; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .context  = */
                new ggml_backend_cann_buffer_type_context{
                    i, GGML_CANN_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cann_buffer_type_initialized = true;
    }

    return &ggml_backend_cann_buffer_types[device];
}

static bool ggml_cann_compute_forward(ggml_backend_cann_context& ctx,
                                      struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cann_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_mul_div<aclnnDivGetWorkspaceSize, aclnnDiv>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cann_activation<aclnnSiluGetWorkspaceSize, aclnnSilu>(
                        ctx, dst);
                    break;
                // TODO: Use faster gelu??
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cann_activation<aclnnTanhGetWorkspaceSize, aclnnTanh>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cann_activation<aclnnReluGetWorkspaceSize, aclnnRelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cann_activation<aclnnHardsigmoidGetWorkspaceSize,
                                         aclnnHardsigmoid>(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cann_activation<aclnnHardswishGetWorkspaceSize,
                                         aclnnHardswish>(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cann_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // Do nothing with these ops.
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            break;
        case GGML_OP_ALIBI:
            ggml_cann_alibi(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
GGML_CALL static const char* ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return cann_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    cann_ctx->free_device_buffers();
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));
    delete cann_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t
ggml_backend_cann_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return ggml_backend_cann_buffer_type(cann_ctx->device);
}

GGML_CALL static void ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                                         ggml_tensor* tensor,
                                                         const void* data,
                                                         size_t offset,
                                                         size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ggml_backend_buffer_t buf =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync(tensor->data, size, (char*)data + offset,
                                   size, ACL_MEMCPY_HOST_TO_DEVICE,
                                   cann_ctx->stream()));
    } else {
        void* transform_buffer = malloc(size);
        ggml_backend_cann_transform(tensor, (char*)data + offset,
                                    transform_buffer);

#ifndef NDEBUG
        void* check_buffer = malloc(size);
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         check_buffer);
        GGML_ASSERT(memcmp((char*)data + offset, check_buffer, size));
        free(check_buffer);
#endif
        ACL_CHECK(aclrtMemcpyAsync(tensor->data, size, transform_buffer, size,
                                   ACL_MEMCPY_HOST_TO_DEVICE,
                                   cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        free(transform_buffer);
    }
}

GGML_CALL static void ggml_backend_cann_get_tensor_async(
    ggml_backend_t backend, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;
    ggml_backend_buffer_t buf =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) &&
                "unsupported buffer type");

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpyAsync((char*)data + offset, size, tensor->data,
                                   size, ACL_MEMCPY_DEVICE_TO_HOST,
                                   cann_ctx->stream()));
    } else {
        void* transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpyAsync(transform_buffer, size, tensor->data, size,
                                   ACL_MEMCPY_DEVICE_TO_HOST,
                                   cann_ctx->stream()));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
        ggml_backend_cann_transform_back(tensor, transform_buffer,
                                         (char*)data + offset);
        free(transform_buffer);
    }
}

GGML_CALL static bool ggml_backend_cann_cpy_tensor_async(
    ggml_backend_t backend_src, ggml_backend_t backend_dst,
    const ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) ||
                ggml_backend_is_cann(backend_dst));

    if (!ggml_backend_buffer_is_cann(src->buffer) ||
        !ggml_backend_buffer_is_cann(dst->buffer)) {
        return false;
    }

    GGML_ASSERT(src->backend == GGML_BACKEND_TYPE_GPU);
    GGML_ASSERT(dst->backend == GGML_BACKEND_TYPE_GPU);

    ggml_backend_buffer_t buf_src =
        src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst =
        dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cann_context* cann_ctx_src =
        (ggml_backend_cann_context*)backend_src->context;
    ggml_backend_cann_context* cann_ctx_dst =
        (ggml_backend_cann_context*)backend_dst->context;

    size_t copy_size = ggml_nbytes(dst);
    if (backend_src != backend_dst) {
        ggml_backend_cann_buffer_context* buf_ctx_src =
            (ggml_backend_cann_buffer_context*)buf_src->context;
        ggml_backend_cann_buffer_context* buf_ctx_dst =
            (ggml_backend_cann_buffer_context*)buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device,
                                           cann_ctx_dst->device));
        if (!canAccessPeer) {
            return false;
        }

        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));

        // record event on src stream
        if (!cann_ctx_src->copy_event) {
            ACL_CHECK(aclrtCreateEvent(&cann_ctx_src->copy_event));
        }

        ACL_CHECK(
            aclrtRecordEvent(cann_ctx_src->copy_event, cann_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx_dst->stream(),
                                       cann_ctx_src->copy_event));
    } else {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    return true;
}

GGML_CALL static void ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));

    // Free temp buffers binding to stream.
    cann_ctx->free_stream_buffers(0);
}

GGML_CALL static enum ggml_status ggml_backend_cann_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        // if tensor is reused, free temp buffers first.
        cann_ctx->free_tensor_buffers(node);
        bool ok = ggml_cann_compute_forward(*cann_ctx, node);

        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cann_supports_op(ggml_backend_t backend,
                                                    const ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT: {
            switch (op->src[0]->type) {
                // case GGML_TYPE_Q4_0:
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_MUL_MAT_ID:
            return false;
        // embedding
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;
        case GGML_OP_CPY: {
            switch (op->type) {
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cann_offload_op(ggml_backend_t backend,
                                                   const ggml_tensor* op) {
    const int min_batch_size = 32;
    GGML_UNUSED(backend);

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
}

static ggml_backend_event_t ggml_backend_cann_event_new(
    ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    return new ggml_backend_event{
        /* .backend = */ backend,
        /* .context = */ event,
    };
}

static void ggml_backend_cann_event_free(ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent)event->context));

    delete event;
}

static void ggml_backend_cann_event_record(ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)event->backend->context;

    ACL_CHECK(aclrtRecordEvent((aclrtEvent)event->context, cann_ctx->stream()));
}

static void ggml_backend_cann_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    if (ggml_backend_is_cann(event->backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(),
                                       (aclrtEvent)event->context));
    } else {
        GGML_ASSERT(false);
    }
}

static void ggml_backend_cann_event_synchronize(ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));
}

static ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .get_default_buffer_type = */ ggml_backend_cann_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_event_new,
    /* .event_free              = */ ggml_backend_cann_event_free,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .event_synchronize       = */ ggml_backend_cann_event_synchronize,
};

static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_cann_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_cann_get_device_count()) {
        fprintf(stderr, "%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context* ctx = new ggml_backend_cann_context(device);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cann_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_cann_guid(),
                         /* .interface = */ ggml_backend_cann_interface,
                         /* .context   = */ ctx};

    return cann_backend;
}

GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

GGML_CALL int32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_info().device_count;
}

GGML_CALL void ggml_backend_cann_get_device_description(
    int32_t device, char* description, size_t description_size) {
    ggml_cann_set_device(device);
    const char* soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

GGML_CALL void ggml_backend_cann_get_device_memory(int32_t device, size_t* free,
                                                   size_t* total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cann_init(const char* params,
                                                           void* user_data) {
    ggml_backend_t cann_backend =
        ggml_backend_cann_init((int)(intptr_t)user_data);
    return cann_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cann_reg_devices();

GGML_CALL int ggml_backend_cann_reg_devices() {
    aclInit(nullptr);
    uint32_t device_count = ggml_backend_cann_get_device_count();
    // initialization
    for (uint32_t i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CANN_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cann_init,
                              ggml_backend_cann_buffer_type(i),
                              (void*)(intptr_t)i);
    }
    return device_count;
}

void ggml_cann_backend_init(void) { ACL_CHECK(aclInit(nullptr)); }

void ggml_cann_backend_free(void) { ACL_CHECK(aclFinalize()); }
