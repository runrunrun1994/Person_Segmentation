#include "PersonSegmentation.h"
#include "gpu.h"
#include <iostream>

PersonSegmentation::PersonSegmentation(const char* param, const char* bin)
{
    personSeg.load_param(param);
    personSeg.load_model(bin);
}

bool PersonSegmentation::process(cv::Mat& image, ncnn::Mat& input)
{
    if (image.empty())
        return false;
    int img_w = image.cols;
    int img_h = image.rows;
    cv::Mat float_Mat;

    //image.convertTo(float_Mat, CV_32F, 1.0/255, 0);
    input = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);
    input.substract_mean_normalize(mean, std);
    const float* pInput = input.channel(0);

    return true;
}

bool PersonSegmentation::run(cv::Mat& img, ncnn::Mat& res)
{
    ncnn::Mat input;
    bool is_process = process(img, input);
    const float* pInput = input.channel(0);
    if (!is_process)
    {
        std::cout << "The image is invlid!\n";

        return false;
    }
    ncnn::Extractor ex = personSeg.create_extractor();
    ex.input("input", input);

    ex.extract("output", res);

    return true;
}

PersonSegmentation::~PersonSegmentation()
{
    personSeg.clear();
}