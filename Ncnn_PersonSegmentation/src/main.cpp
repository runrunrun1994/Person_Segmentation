#include <opencv2/opencv.hpp>

#include "PersonSegmentation.h"

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cout << "Too few input parameters \n";
        std::cout << "./person model_param model_bin  image_path save_path\n";
        exit(-1);
    }
    const char* param = argv[1];
    const char* bin = argv[2];
    bool hasGPU = false;
    PersonSegmentation *personSeg = new PersonSegmentation(param, bin);

    const char* img_path = argv[3];
    cv::Mat image = cv::imread(img_path);
    cv::resize(image, image, cv::Size(480, 480));
    ncnn::Mat res;
    personSeg->run(image, res);

    cv::Mat mask(image.rows, image.cols, CV_8UC1);
    uchar* pMask = mask.data;
    const float* class0mask = res.channel(0);
    const float* class1mask = res.channel(1);
    for (int i = 0; i < image.cols*image.rows; i++)
    {
        pMask[i] = class0mask[i] > class1mask[i] ? 0:1;
    }

    cv::Mat person;
    image.copyTo(person, mask);

    cv::imwrite(argv[4], person);

    return 0;
}