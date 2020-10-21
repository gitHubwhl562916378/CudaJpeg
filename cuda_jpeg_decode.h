/*
 * @Author: your name
 * @Date: 2020-10-20 03:40:00
 * @LastEditTime: 2020-10-21 06:28:08
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /tensorrt/CudaJpeg/cuda_jpeg_decode.h
 */
#include <vector>
#include <nvjpeg.h>
#include <opencv2/opencv.hpp>

class CudaJpegDecode
{
public:
    explicit CudaJpegDecode();
    ~CudaJpegDecode();
    /**
     * @description: 初始化硬件及nvjpeg
     * @param {batch_size} 批处理时，批次大小；不足一批的从list开头继续补充
     * @param {max_cpu_threads} cpu的最大线程数
     * @param {out_fmt} 解出的格式
     * @param {device} 显卡编号
     * @return {bool} 成功true, 失败false
     */    
    bool DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, const int device = -1);
    
    /**
     * @description: 解压缩一张图片
     * @param {image} 输入图片的内容
     * @param {pipelined} true使用解耦的接口形式
     * @return {dst} 解压缩的图片，　cv::Mat或者cv::cuda::Mat
     * @return {bool} 成功true, 失败false
     */    
    bool Decode(const std::vector<uchar> &image, cv::OutputArray dst, bool pipelined = false);

    /**
     * @description: 解压缩批量图片
     * @param {images} 所有图片的数据，个数需要小于等于DeviceInit中的batch_size 
     * @return {dst} 解出的图上，可以是std::vector<cv::Mat>或者std::vector<cv::cuda::GpuMat>
     * @return {bool} 解压缩成功true, 失败false
     */    
    bool Decode(const std::vector<std::vector<uchar>> &images, cv::OutputArray &dst);

private:
    static int host_malloc(void **p, size_t s, unsigned int f);
    static int host_free(void *p);
    static int dev_malloc(void **p, size_t s);
    static int dev_free(void *p);
    int ConvertSMVer2Cores(int major, int minor);
    int GpuGetMaxGflopsDeviceId();
    bool DecodePipelined(const std::vector<char> &image, cv::OutputArray dst);

    nvjpegDevAllocator_t dev_allocator_;
    nvjpegPinnedAllocator_t pinned_allocator_;
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegState_t nvjepg_state_;
    nvjpegOutputFormat_t out_fmt_;
    int batch_size_ = 0;
    
    // used with decoupled API
    nvjpegJpegDecoder_t nvjpeg_decoder_;
    nvjpegJpegState_t nvjpeg_decoupled_state_;
    nvjpegBufferPinned_t pinned_buffers_;
    nvjpegBufferDevice_t device_buffer_;
    nvjpegJpegStream_t jpeg_streams_;
    nvjpegDecodeParams_t nvjpeg_decode_params_;
};