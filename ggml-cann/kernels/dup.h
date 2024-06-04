#ifndef DUP_H
#define DUP_H

#pragma pack(push, 8)
typedef struct {
    int64_t src_ne[4];
    int64_t src_nb[4];
    int64_t dst_ne[4];
    int64_t dst_nb[4];

} dup_param;
#pragma pack(pop)

#endif //DUP_H