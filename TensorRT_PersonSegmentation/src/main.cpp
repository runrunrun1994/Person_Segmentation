#include "personsegmentation.h"
#include "logger.h"
#include <sys/time.h>

int main(int argc, char** argv)
{

    if (argc < 4){
        gLogError << "The params is few!" << std::endl;
        gLogInfo << "./person onnx_file_path trt_file_path image_path save_path\n";

        exit(-1);
    }
    std::string onnxFilePath = argv[1];
    std::string trtFilePath  = argv[2];
    std::string imagePath    = argv[3];
    std::string savePath    = argv[4];


    ModelParams::OnnxParams params;
    params.batchSize = 1;
    params.onnxFilePath = onnxFilePath;
    params.gpuId = 0;
    params.trtFilePath = trtFilePath;
    params.mean = {0.45734706f, 0.43338275f, 0.40058118f};
    params.variance = {0.23965294f, 0.23532275f, 0.2398498f};

    Model::PersonSegmentation personSeg(params);
    personSeg.build();

    cv::Mat image= cv::imread(imagePath);
    cv::resize(image, image, cvSize(480, 480));
    float* output = (float*)malloc(480*480*2*sizeof(float));
    struct timeval tp;
    struct timeval tp1;
    int start;
    int end;

    gettimeofday(&tp, NULL);
    start = tp.tv_sec*1000 + tp.tv_usec/1000;

    for (int i = 0; i < 1000; i++){
        personSeg.run(image, output);
    }

    gettimeofday(&tp1, NULL);
    end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    std::cout << (end -start) / 1000 << std::endl;

    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (int row = 0; row< 480; ++row){
        unsigned char* pMask = mask.ptr<uchar>(row);
        for (int col = 0; col < 480; ++col){
            if (output[row*480 + col] > output[row*480 + col + 480*480]){
                pMask[col] = 0;
            }
            else
                pMask[col] = 1;
        }
    }

    cv::Mat res;
    image.copyTo(res, mask);

    cv::imwrite(savePath, res);
    cv::waitKey(1);

    return 0;
}