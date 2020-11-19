/*******************************************
* \file utils.h
* \brief 一些工具函数
*
* \author runrunrun1994
* \date 2020-11-15 
*******************************************/
#include <string>
#include <unistd.h>
#include <torch/torch.h>

#include "log.h"

bool IsExists(std::string file_path)
{
    if (access(file_path.c_str(), 0) == 0)
        return true;
    else
        return false;
}

bool HaveGPUs()
{
    if (torch::cuda::is_available())
    {
        
#ifdef DEBUG
        LOG_INFO("Detect the GPU!");
#endif

        return true;
    }
    else
        return false;
}

void Normalize(torch::Tensor& input, std::vector<float> meanValue, std::vector<float> stdValue)
{
    input = input.to(torch::kFloat);
    input = input / 255.0;

    if (((meanValue.size()) != 3 && (stdValue.size() != 3)))
    {
        LOG_ERROR("The meanValue or stdValue is invalid");
        return ;
    }

    input[0][0] = (input[0][0] - meanValue[0]) / stdValue[0];
    input[0][1] = (input[0][1] - meanValue[1]) / stdValue[1];
    input[0][2] = (input[0][2] - meanValue[2]) / stdValue[2];
}