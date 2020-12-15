/******************************************************************
*\file utils.h
*\brief 一些工具函数
*\date 2020-12-13 
*\author runrunrun1994
*
*******************************************************************/

#define CHECK(call) do{\
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    {\
        printf("ERROR: %s:%d,",__FILE__, __LINE__); \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}while(0)