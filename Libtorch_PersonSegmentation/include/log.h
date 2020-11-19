/*******************************************
* \file log.h
* \brief 日志
*
* \author runrunrun1994
* \date 2020-11-15 
*******************************************/

#include <cstdio>

#define LOG_ERROR(fmt, args...) do{                         \
    printf("\033[31m ERROR! \033[0m]");                     \
    printf("[%s : %d : %s] ", __FILE__, __LINE__, __func__);\
    printf(fmt, ##args);                                    \
    printf("\n");                                           \
}while(0);

#define LOG_INFO(fmt, args...) do{                         \
    printf(fmt, ##args);                                   \
    printf("\n");                                          \
}while(0);

