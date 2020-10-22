<!--
 * @Author: your name
 * @Date: 2020-10-21 06:17:14
 * @LastEditTime: 2020-10-22 10:10:58
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /tensorrt/CudaJpeg/README.md
-->
# CUDA_JPEG
=========================

使用cuda解压缩jpg图片，　输出cv::Mat或者cv::cuda::GpuMat

# Requires
>+ cuda 10.0以上
>+ opencv master分支带编译cuda版本, opencv_no_cuda分支常规版本

# Build
1. mkdir build
2. cd build
3. cmake .. && make

# Report
* 解压缩到cpu的话，建议使用cv::imdecode。解压缩到gpu使用CudaJpegDecode
* 使用master分支，也就是opencv带cuda版本时; CudaJpegDecode接口性能比不带cuda的好