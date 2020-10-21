/*
 * @Author: your name
 * @Date: 2020-10-20 08:41:10
 * @LastEditTime: 2020-10-21 09:06:56
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /tensorrt/CudaJpeg/main.cpp
 */
#include <thread>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../cuda_jpeg_decode.h"

std::vector<uchar> GetContents(const std::string &file_name)
{
    std::ifstream ifile(file_name, std::ios::in | std::ios::binary | std::ios::ate);
    int length = ifile.tellg();
    ifile.seekg(0, std::ios::beg);
    std::vector<uchar> buffer(length, 0);
    ifile.read((char*)buffer.data(), length);
    ifile.close();

    return buffer;
}

void TestOneImage()
{
    CudaJpegDecode jpg_decoder;

    jpg_decoder.DeviceInit(32, std::thread::hardware_concurrency(), NVJPEG_OUTPUT_BGR);

    cv::Mat image;
    jpg_decoder.Decode(GetContents("whl_feature.jpg"), image, false);
    cv::imshow("win", image);
    cv::waitKey(0);
}

void TestBatchedImages()
{
    std::vector<std::vector<uchar>> images;
    images.push_back(GetContents("whl_feature.jpg"));
    images.push_back(GetContents("hezhongjie.jpg"));

    std::vector<cv::Mat> outs;
    CudaJpegDecode jpg_decoder;
    jpg_decoder.DeviceInit(2, std::thread::hardware_concurrency(), NVJPEG_OUTPUT_BGR);
    jpg_decoder.Decode(images, outs);

    std::cout << "Press Enter scan next" << std::endl;
    cv::imshow("win", outs.at(0));
    cv::waitKey(0);
    cv::imshow("win", outs.at(1));
    cv::waitKey(0);
}

int main(int argc, char **argv)
{
    TestBatchedImages();
    return 0;
}