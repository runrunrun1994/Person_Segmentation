/*******************************************
* \file utils.h
* \brief 一些工具函数
*
* \author runrunrun1994
* \date 2020-11-15 
*******************************************/
#include <string>

bool IsExists(std::string file_path);

bool HaveGPUs();

void Normalize(torch::Tensor& input,std::vector<float> meanValue, std::vector<float> stdValue);