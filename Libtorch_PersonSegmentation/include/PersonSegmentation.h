/******************************************
*\author runrunrun1994
*\date   2020-11-15 
*******************************************/

#ifndef __PERSON_SEGMENTATION_H__
#define __PERSON_SEGMENTATION_H__

#include "params.h"

class PersonSegmentation{
public:

    /// \brief 初始化模型
    /// \param pparams 模型参数
    /// \return 返回模型是否初始化成功标志位
    bool InitModel(PTModelParam pparams);

    /// \brief 运行模型
    /// \param img 输入图像
    /// \return 返回分割结果
    torch::Tensor Forward(cv::Mat& img);

    /// 释放模型
    bool UnInitModel();

    ~PersonSegmentation();

private:
    TModelParam mtParam;
    std::shared_ptr<torch::jit::script::Module> mpPersonSegModel;

    /// \brief 图像预处理函数
    /// \param src 原始输入图像
    /// \param res 结果图像
    void PreProcess(const cv::Mat& src, torch::Tensor& input);
};

#endif