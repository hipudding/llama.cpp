#ifndef GET_ROW_Q8_0_H
#define GET_ROW_Q8_0_H

#pragma pack(push, 8)
typedef struct {
    int64_t input_ne[4];
    int64_t indices_ne[4];
} get_row_param;
#pragma pack(pop)

#endif //GET_ROW_Q8_0_H