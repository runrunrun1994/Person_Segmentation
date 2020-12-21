#include "personsegmentation.h"
#include "common.h"
#include "params.h"
#include "logger.h"
#include "errorcode.h"
#include "utils.h"

#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

namespace Model{
    PersonSegmentation::PersonSegmentation(const ModelParams::OnnxParams& params){
        mParams.batchSize     = params.batchSize;
        mParams.gpuId         = params.gpuId;
        mParams.onnxFilePath  = params.onnxFilePath;
        mParams.trtFilePath   = params.trtFilePath;
        mParams.mean          = params.mean;
        mParams.variance      = params.variance;

        mEngine = nullptr;
        mHostInput = nullptr;
        mDeviceBuffer.resize(2, nullptr);
    }

    int PersonSegmentation::build(){
        bool parseFlag = false;

        if (access(mParams.trtFilePath.c_str(), 0) == 0){
            parseFlag = parseFromTrt();

            if (!parseFlag){
                gLogError << "Parsed The trt file failed!\n";

                return PERSON_FILEERROR;
            }
        }
        else{
            //从onnx文件中解析
            parseFlag = parseFromOnnx();

            if (!parseFlag){
                gLogError << "Parsed The Onnx file Failed!\n";

                return PERSON_FILEERROR;
            }
        }

        int mallocFlag = mallocHostAndDevice();

        if (mallocFlag != PERSON_SUCCESSED){
            return PERSON_MALLOCERROR;
        }

        CHECK(cudaEventCreate(&mCudaEvent));
        CHECK(cudaStreamCreate(&mCudaStream));

        mContext = mEngine->createExecutionContext();


        return PERSON_SUCCESSED;
    }

    int PersonSegmentation::run(const cv::Mat& image, float* output){
        if (mEngine == nullptr || mContext == nullptr){
            gLogError << "The Engine or Context is invalid\n";

            return PERSON_VARIABLEINVALID;
        }
        prepocess(image);

        CHECK(cudaMemcpyAsync(mDeviceBuffer.at(0), mHostInput, mParams.batchSize*480*480*3*sizeof(float),
                            cudaMemcpyHostToDevice, mCudaStream));
        mContext->executeV2(mDeviceBuffer.data());

        CHECK(cudaMemcpyAsync(output, mDeviceBuffer.at(mEngine->getBindingIndex("output")),
                                mParams.batchSize*480*480*2*sizeof(float),
                                cudaMemcpyDeviceToHost, mCudaStream));
        CHECK(cudaStreamSynchronize(mCudaStream));
    }

    bool PersonSegmentation::parseFromOnnx(){
        auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));

        if (!builder){
            return false;
        }

        auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1U));

        if (!network){
            return false;
        }

        auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
       
        if (!config){
            return false;
        }

        auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network,
                                                                            gLogger.getTRTLogger()));

        if (!parser){
            return false;
        }

        constructNetwork(builder, network, config, parser);

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config),
                                                                                    InferDeleter());

        if (!mEngine){
            return false;
        }

        nvinfer1::IHostMemory* engine_serialize = mEngine->serialize();
        std::ofstream out(mParams.trtFilePath.c_str(), std::ios::binary);
        out.write((const char*)engine_serialize->data(), engine_serialize->size());
        out.close();

        return true;

    }

    bool PersonSegmentation::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
                            UniquePtr<nvinfer1::INetworkDefinition>& network,
                            UniquePtr<nvinfer1::IBuilderConfig>& config,
                            UniquePtr<nvonnxparser::IParser>& parser)
    {
        auto parsed = parser->parseFromFile(mParams.onnxFilePath.c_str(),
                                    static_cast<int>(gLogger.getReportableSeverity()));

        if (!parsed){
            return false;
        }

        config->setAvgTimingIterations(1);
        config->setMinTimingIterations(1);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setMaxWorkspaceSize(2_GiB);
        builder->setMaxBatchSize(mParams.batchSize);

        return true;
    }

    bool PersonSegmentation::parseFromTrt(){
        gLogInfo << "Deserialize for local: " << mParams.trtFilePath << std::endl;
        nvinfer1::IRuntime* iruntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

        std::ifstream intrt(mParams.trtFilePath, std::ios::binary);
        intrt.seekg(0, std::ios::end);
        size_t length = intrt.tellg();

        if (length < 0){
            gLogError << "The file : " << mParams.trtFilePath << " is empty!\n";
            return false;
        }

        intrt.seekg(0, std::ios::beg);
        std::vector<char> data(length);
        intrt.read(data.data(), length);

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(iruntime->deserializeCudaEngine(data.data(), data.size()),
                                                        InferDeleter());

        if (!mEngine){
            gLogError << "Deserialize Faild \n";
            return false;
        }

        return true;
    }

    int PersonSegmentation::mallocHostAndDevice(){
        CHECK(cudaMallocHost((void**)&mHostInput, mParams.batchSize*480*480*3*sizeof(float)));
        memset(mHostInput, 0.0, mParams.batchSize*480*480*3*sizeof(float));
        if (mHostInput == nullptr){
            gLogError <<"The Host Memery malloc failed!\n";

            return PERSON_MALLOCERROR;
        }

        if (mEngine == nullptr){
            gLogError << "The Engine is nullptr\n";
            return PERSON_ENGINEERROR; 
        }

        CHECK(cudaMalloc((void**)&mDeviceBuffer.at(mEngine->getBindingIndex("input")), mParams.batchSize*480*480*3*sizeof(float)));
        CHECK(cudaMalloc((void**)&mDeviceBuffer.at(mEngine->getBindingIndex("output")),mParams.batchSize*480*480*2*sizeof(float)));

        if ((mDeviceBuffer.at(0) == nullptr) || (mDeviceBuffer.at(1) == nullptr)){
            gLogError << "The Device Memery malloc failed!\n";

            return PERSON_MALLOCERROR;
        }

        return PERSON_SUCCESSED;
    }

    void PersonSegmentation::freeHostAndDevice(){
        if (mHostInput){
            cudaFreeHost(mHostInput);
            mHostInput = nullptr;
        }

        for (size_t i = 0; i < 2; i++){
            if (mDeviceBuffer.at(i)){
                cudaFree(mDeviceBuffer.at(i));
                mDeviceBuffer.at(i) = nullptr;
            }
        }
    }

    int PersonSegmentation::normalize(const cv::Mat& RGBImage){
        const int H = RGBImage.rows;
        const int W = RGBImage.cols;

        if (RGBImage.empty()){
            gLogError << "The RGB image is invalid\n";

            return PERSON_VARIABLEINVALID;
        }


        if (mParams.mean.size() < 3 || mParams.variance.size() < 3){
            gLogError <<"The variable invalid!\n";

            return PERSON_VARIABLEINVALID; 
        }

        for (int row = 0; row <H; ++row){
            unsigned char* pRGB = RGBImage.data + row * RGBImage.step;

            for (int col = 0; col < W; ++col){
               mHostInput[row*480 +col]            = (float)((pRGB[0] / 255.0 - mParams.mean[0])/mParams.variance[0]);
               mHostInput[row*480 + col+480*480]   = (float)((pRGB[1] / 255.0 - mParams.mean[1])/mParams.variance[1]);
               mHostInput[row*480 + col+2*480*480] = (float)((pRGB[2] / 255.0 - mParams.mean[2])/mParams.variance[2]);

               pRGB += 3;
            }
        }
    }

    int PersonSegmentation::prepocess(const cv::Mat& image){
        cv::Mat RGBImage;
        cv::cvtColor(image, RGBImage,cv::COLOR_BGR2RGB);
        normalize(RGBImage);
    }

    PersonSegmentation::~PersonSegmentation(){
        
    }
}