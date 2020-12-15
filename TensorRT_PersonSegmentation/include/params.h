/********************************************************
* \file params.h
* \brief 模型参数结构体
* \date 2020-12-11 
* \author runrunrun1994
*
*********************************************************/


#ifndef __PARAMS_H__
#define __PARAMS_H__
#include <string>
#include <vector>

namespace ModelParams
{
    typedef struct Params
    {
        int batchSize{1};                        ///< batch size
        int gpuId{0};                            ///< gpu id
        
    }Params;    //params

    struct OnnxParams : public Params
    {
        std::string onnxFilePath;                ///< onnx文件路径
        std::string trtFilePath;                 ///< trt文件路径
        std::vector<float> mean;                   ///< 均值
        std::vector<float> variance;             ///< 方差
    };  //OnnxParams

} //ModelParams

#endif