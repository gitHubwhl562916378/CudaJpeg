<!--
 * @Author: your name
 * @Date: 2020-10-21 06:17:14
 * @LastEditTime: 2020-10-21 07:40:45
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