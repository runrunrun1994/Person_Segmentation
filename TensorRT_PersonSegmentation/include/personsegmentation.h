/*****************************************************
* \file personsegmention.h
* \brief 算法封装类
* \date 2020-12-11 
* \author runrunrun1994
*
*******************************************************/

#ifndef __PERSONSEGMENTION_H__
#define __PERSONSEGMENTION_H__

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "common.h"
#include "params.h"
#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace Model{
    const std::string gProjectName = "TensorRT.PersonSegmentation";

    class PersonSegmentation{
        template <typename T>
        using UniquePtr = std::unique_ptr<T, InferDeleter>;

    public:
        PersonSegmentation(const ModelParams::OnnxParams& params);
        ~PersonSegmentation();
        int build();
        int run(const cv::Mat& image, float* output);

    private:
        ModelParams::OnnxParams mParams;
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
        cudaStream_t mCudaStream;
        cudaEvent_t mCudaEvent;
        nvinfer1::IExecutionContext* mContext;

        float* mHostInput;
        std::vector<void*> mDeviceBuffer;

        int mallocHostAndDevice();
        void freeHostAndDevice();
        int prepocess(const cv::Mat& image);
        int normalize(const cv::Mat& rgbImage);

        bool parseFromOnnx();
        bool parseFromTrt();
        bool constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
                            UniquePtr<nvinfer1::INetworkDefinition>& network,
                            UniquePtr<nvinfer1::IBuilderConfig>& config,
                            UniquePtr<nvonnxparser::IParser>& parser);
                            
    }; ///< end of PersonSegmentation

}      ///< end of namespace Model

#endif