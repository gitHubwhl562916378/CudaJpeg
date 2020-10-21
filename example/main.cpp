/*
 * @Author: your name
 * @Date: 2020-10-20 08:41:10
 * @LastEditTime: 2020-10-21 11:06:41
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

    cv::Mat image; //or cv::cuda::GpuMat
    std::vector<uchar> content = GetContents("whl_feature.jpg");
    jpg_decoder.Decode(content.data(), content.size(), image, false);
    cv::imshow("win", image);
    cv::waitKey(0);
}

void TestBatchedImages()
{
    std::vector<uchar*> images;
    std::vector<size_t> lengths;
    std::vector<uchar> img1 = GetContents("whl_feature.jpg");
    std::vector<uchar> img2 = GetContents("hezhongjie.jpg");
    images.push_back(img1.data());
    images.push_back(img2.data());
    lengths.push_back(img1.size());
    lengths.push_back(img2.size());

    std::vector<cv::Mat> outs; //or std::vector<cv::cuda::GpuMat>
    CudaJpegDecode jpg_decoder;
    jpg_decoder.DeviceInit(2, std::thread::hardware_concurrency(), NVJPEG_OUTPUT_BGR);
    jpg_decoder.Decode(images, lengths, outs);

    std::cout << "Press Enter scan next" << std::endl;
    cv::imshow("win", outs.at(0));
    cv::waitKey(0);
    cv::imshow("win", outs.at(1));
    cv::waitKey(0);
}

int main(int argc, char **argv)
{
    TestOneImage();
    return 0;
}