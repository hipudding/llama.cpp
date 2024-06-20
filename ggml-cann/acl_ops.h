#ifndef CANN_ACL_OPS
#define CANN_ACL_OPS

#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>

#include <string>
#include <vector>

#include "acl_tensor.h"
#include "common.h"

struct OpCaller {
    std::string op_name;
    std::vector<aclTensorDesc*> input_descs;
    std::vector<aclDataBuffer*> input_buffers;
    std::vector<aclTensorDesc*> output_descs;
    std::vector<aclDataBuffer*> output_buffers;
    aclopAttr* attrs;
    std::vector<void*> ptrs;

    OpCaller();

    virtual ~OpCaller();

    OpCaller& name(std::string _op_name);

    OpCaller& input_no_contiguous(ggml_tensor* tensor, const char* name);

    OpCaller& input(ggml_tensor* tensor, const char* name);

    OpCaller& output(ggml_tensor* tensor, const char* name);

    OpCaller& attr(int64_t value, const char* name);

    OpCaller& attr(bool value, const char* name);

    OpCaller& attr(float value, const char* name);

    template <typename T>
    OpCaller& input(ggml_backend_cann_context& ctx, ggml_tensor* dst, T* values,
                    aclDataType dtype, size_t dims, int64_t* dim,
                    const char* name, aclrtStream stream = nullptr) {
        size_t n_elem = 1;
        for (size_t i = 0; i < dims; i++) {
            n_elem *= dim[i];
        }

        size_t n_bytes = n_elem * sizeof(T);
        void* device_ptr = ctx.alloc_buffer(dst, n_bytes);
        if (stream == nullptr) {
            ACL_CHECK(aclrtMemcpy(device_ptr, n_bytes, values, n_bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            ACL_CHECK(aclrtMemcpyAsync(device_ptr, n_bytes, values, n_bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE, stream));
        }

        aclTensorDesc* tensor_desc =
            aclCreateTensorDesc(dtype, dims, dim, ACL_FORMAT_ND);
        aclSetTensorDescName(tensor_desc, name);
        input_descs.push_back(tensor_desc);
        aclDataBuffer* data_buffer = aclCreateDataBuffer(device_ptr, n_bytes);
        input_buffers.push_back(data_buffer);

        return *this;
    }

    OpCaller& run(aclrtStream stream = nullptr);
};

#endif  // CANN_ACL_OPS
