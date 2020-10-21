/*
 * @Author: your name
 * @Date: 2020-10-20 03:40:00
 * @LastEditTime: 2020-10-20 09:43:47
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
    bool DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, const int device = -1);
    void SetWarmup(const int warmup);
    bool Decode(const std::vector<uchar> &image, cv::OutputArray dst, bool pipelined = false);
    bool Decode(const std::vector<std::vector<uchar>> &images, std::vector<cv::OutputArray> &out);

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
    
    // used with decoupled API
    nvjpegJpegDecoder_t nvjpeg_decoder_;
    nvjpegJpegState_t nvjpeg_decoupled_state_;
    nvjpegBufferPinned_t pinned_buffers_;
    nvjpegBufferDevice_t device_buffer_;
    nvjpegJpegStream_t jpeg_streams_;
    nvjpegDecodeParams_t nvjpeg_decode_params_;
};