/************************************************
*\file PersonSegmentation.h
*\date 2020-11-23 
*\author runrunrun1994
*
*************************************************/
#ifndef _PERSIN_SEGMENTATION_H_
#define _PERSIN_SEGMENTATION_H_

#include "net.h"
#include "mat.h"
#include <opencv2/opencv.hpp>

class PersonSegmentation{

public:
    PersonSegmentation(const char* param, const char* bin);
    ~PersonSegmentation();
    bool run(cv::Mat& img, ncnn::Mat& res);

private:
    ncnn::Net personSeg;
    const float mean[3] = {0.45734706f * 255.f, 0.43338275f * 255.f, 0.40058118f*255.f};
    const float std[3] = {1/0.23965294/255.f, 1/0.23532275/255.f, 1/0.2398498/255.f};
    bool process(cv::Mat& image, ncnn::Mat& input);
};

#endif