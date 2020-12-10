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
    std::cout << pInput[447] << std::endl;

    return true;
}

bool PersonSegmentation::run(cv::Mat& img, ncnn::Mat& res)
{
    ncnn::Mat input;
    bool is_process = process(img, input);
    const float* pInput = input.channel(0);
    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < input.w; x++)
        {
            std::cout << pInput[x] << " ";
        }

        pInput += input.w;
        std::cout << std::endl;
    }
    std::cout << pInput[447] << std::endl;
    if (!is_process)
    {
        std::cout << "The image is invlid!\n";

        return false;
    }
    ncnn::Extractor ex = personSeg.create_extractor();
    ex.input("input", input);
    ncnn::Mat Clip_10;
    ex.extract("1003", Clip_10);
    const float* pClip = Clip_10.channel(2);

    for (int y = 0; y < Clip_10.h; y++)
    {
        for (int x = 0; x < Clip_10.w; x++)
            std::cout << pClip[x] << " ";
        pClip += Clip_10.w;
        std::cout << std::endl;
    }
    std::cout << "Clip w: " << Clip_10.w << std::endl;
    std::cout << "Clip h: " << Clip_10.h << std::endl;
    std::cout << "Clip c: " << Clip_10.c << std::endl;
    ex.extract("output", res);
    std::cout << "Here" << std::endl;

    return true;
}

PersonSegmentation::~PersonSegmentation()
{
    personSeg.clear();
}