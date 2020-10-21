#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "cuda_jpeg_decode.h"

int main(int argc, char **argv)
{
    CudaJpegDecode jpg_decoder;

    jpg_decoder.DeviceInit(32, std::thread::hardware_concurrency(), NVJPEG_OUTPUT_BGR);
    
    std::ifstream ifile("whl_feature.jpg", std::ios::in | std::ios::binary | std::ios::ate);
    int length = ifile.tellg();
    ifile.seekg(0, std::ios::beg);
    std::vector<uchar> buffer(length, 0);
    ifile.read((char*)buffer.data(), length);
    ifile.close();

    cv::Mat image;
    jpg_decoder.Decode(buffer, image, true);
    cv::imshow("win", image);
    cv::waitKey(0);

    return 0;
}