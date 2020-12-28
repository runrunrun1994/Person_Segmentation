### 1. 文件目录说明
- Libtorch_PersonSegmentation Libtorch实现代码
- Ncnn_PersonSegmentation     ncnn实现代码
- TensorRT_PersonSegmentation TensorRT实现代码

### 2.实验结果
|Backbone|Pixel Accuracy|Mean_IoU|Background IoU |Person IoU|
|---|---|---|---|---|
|Mobilenetv2_1.0|**0.971**|**0.943**|**0.953**|**0.933**|
|Mobilenetv2_0.5|0.966|0.933|0.944|0.922|
|Mobilenetv2_0.35|0.961|0.922|0.935|0.909|
|Mobilenetv2_0.25|0.955|0.911|0.926|0.896|

|Backbone|Params|FLOPS|
|---|---|---|
|Mobilenetv2_1.0|5.794M|23.260G|
|Mobilenetv2_0.5|3.282M|20.597G|
|Mobilenetv2_0.35|2.697M|19.993G|
|Mobilenetv2_0.25|2.344M|19.632G|
- 数据分布、及组成**待补充**
- 速度对比**待补充**

|Original|Result|
|---|---|
|![org1](https://github.com/runrunrun1994/Image/blob/main/PersonSegmeantation/PC/pexels-photo-880474.jpg)|![res1](https://github.com/runrunrun1994/Image/blob/main/PersonSegmeantation/PC/pexels-photo-880474.png)|
|![org2](https://github.com/runrunrun1994/Image/blob/main/PersonSegmeantation/PC/timg.jpg)|![res2](https://github.com/runrunrun1994/Image/blob/main/PersonSegmeantation/PC/timg.png)|  

### 3.引用
[1] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)  
[2] [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)  
[3] [https://github.com/Tencent/ncnn](https://github.com/Tencent/ncnn)  
[4] [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)  
[5] [https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)  
**其他引用待补充**