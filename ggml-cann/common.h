#ifndef CANN_COMMON_H
#define CANN_COMMON_H

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../ggml-cann.h"
#include "../ggml.h"

#define MATRIX_ROW_PADDING 512
#define GGML_CANN_MAX_STREAMS 8

[[noreturn]] void ggml_cann_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg);

// Error handling macro
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, aclGetRecentErrMsg)

struct ggml_cann_device_info {
    int32_t device_count;

    // TODO: add more device info later.
    // struct cann_device_info {
    //     int     cc;                 // compute capability
    //     size_t  smpb;               // max. shared memory per block
    //     bool    vmm;                // virtual memory support
    //     size_t  vmm_granularity;    // granularity of virtual memory
    //     size_t  total_vram;
    // };

    // cann_device_info devices[GGML_CANN_MAX_DEVICES] = {};
};

const ggml_cann_device_info& ggml_cann_info();

void ggml_cann_set_device(int32_t device);
int32_t ggml_cann_get_device();

struct ggml_backend_cann_context {
    int32_t device;
    std::string name;
    aclrtEvent copy_event = nullptr;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = {{nullptr}};

    // bind temp buffers to stream. Free after sync.
    std::vector<void*> buffers[GGML_CANN_MAX_STREAMS];

    explicit ggml_backend_cann_context(int device)
        : device(device), name(GGML_CANN_NAME + std::to_string(device)) {}

    ~ggml_backend_cann_context() {
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
                // Buffers should have been freed.
                GGML_ASSERT(buffers[i].size() == 0);
            }
        }
    }

    void* alloc_buffer(size_t size, int stream) {
        void* buffer;
        ACL_CHECK(aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_HUGE_FIRST));
        bind_buffer(buffer, stream);
        return buffer;
    }

    void* alloc_buffer(size_t size) {
        return alloc_buffer(size, 0);
    }

    void free_buffers() {
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; i++) {
            for (void* buffer : buffers[i]) {
                ACL_CHECK(aclrtFree(buffer));
            }
            buffers[i].clear();
        }
    }

    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    void bind_buffer(void* buf, int stream) { buffers[stream].push_back(buf); }

    aclrtStream stream() { return stream(0); }
};



#endif //CANN_COMMON_H