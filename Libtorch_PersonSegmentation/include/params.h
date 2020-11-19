/*******************************************
* \file params.h
* \brief 模型参数
*
* \author runrunrun1994
* \date 2020-11-15 
*******************************************/


#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <vector>
#include <string>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

typedef struct _tModelParam
{
    std::string modelPath;                    ///< 模型路径

    std::vector<float> meanValue;             ///< 均值
    std::vector<float> stdValue;              ///< 方差
    
    torch::DeviceType deviceType;             ///< 运行设备类型
    int gpuId;                                ///< GPU编号
} TModelParam, *PTModelParam;

#endif