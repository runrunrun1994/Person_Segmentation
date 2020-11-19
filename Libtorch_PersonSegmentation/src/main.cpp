#include "log.h"
#include "params.h"
#include "PersonSegmentation.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        LOG_ERROR("Too few input parameters");
        LOG_INFO("./person model_path image_path save_path");
        exit(-1);
    }
    TModelParam modelParam;
    modelParam.modelPath = argv[1];
    modelParam.meanValue = {0.45734706, 0.43338275, 0.40058118};
    modelParam.stdValue  = {0.23965294, 0.23532275, 0.2398498};
    modelParam.deviceType = torch::DeviceType::CUDA;
    modelParam.gpuId = 0;

    PersonSegmentation  ptPersonSeg;
    bool is_load = ptPersonSeg.InitModel(&modelParam);
    if (!is_load)
    {
        LOG_ERROR("The model loading failed!");
    }
    cv::Mat image = cv::imread(argv[2]);

    if (image.empty())
    {
        LOG_ERROR("The image read failed!");
    }
    
    torch::Tensor output = ptPersonSeg.Forward(image);

    if (output.numel() < image.rows*image.cols)
    {
        LOG_ERROR("The output is invalid!");
    }

    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1);
    memcpy(mask.data, output.data_ptr(), output.numel()*sizeof(torch::kU8));
    cv::Mat res;
    image.copyTo(res, mask);

    cv::imwrite(argv[3], res);

    return 0;
}