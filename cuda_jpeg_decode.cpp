/*
 * @Author: your name
 * @Date: 2020-10-20 03:40:09
 * @LastEditTime: 2020-10-21 02:22:00
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /tensorrt/CudaJpeg/cuda_jpeg_decode.cpp
 */
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "cuda_jpeg_decode.h"
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName((cudaError_t)result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

CudaJpegDecode::CudaJpegDecode()
{
}

bool CudaJpegDecode::DeviceInit(const int batch_size, const int max_cpu_threads, const nvjpegOutputFormat_t out_fmt, const int device)
{
    int device_id = -1;

    if (device == -1)
    {
        device_id = GpuGetMaxGflopsDeviceId();
    }
    else
    {
        device_id = device;
    }

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, device_id));
    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
           device_id, props.name, props.multiProcessorCount,
           props.maxThreadsPerMultiProcessor, props.major, props.minor,
           props.ECCEnabled ? "on" : "off");

    dev_allocator_ = {&CudaJpegDecode::dev_malloc, &CudaJpegDecode::dev_free};
    pinned_allocator_ = {&CudaJpegDecode::host_malloc, &CudaJpegDecode::host_free};
    int flags = 0;
    checkCudaErrors(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_, &pinned_allocator_, flags, &nvjpeg_handle_));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjepg_state_));
    out_fmt_ = out_fmt;
    checkCudaErrors(nvjpegDecodeBatchedInitialize(nvjpeg_handle_, nvjepg_state_, batch_size, max_cpu_threads, out_fmt));

    //for pipelined
    checkCudaErrors(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder_));
    checkCudaErrors(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decoupled_state_));

    checkCudaErrors(nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_));
    checkCudaErrors(nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &device_buffer_));

    checkCudaErrors(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_));

    checkCudaErrors(nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));
}

void CudaJpegDecode::SetWarmup(const int warmup)
{
}

bool CudaJpegDecode::Decode(const std::vector<uchar> &image, cv::OutputArray dst, bool pipelined)
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvjpegImage_t iout;
    for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
    {
        iout.channel[i] = nullptr;
        iout.pitch[i] = 0;
    }

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;

    checkCudaErrors(nvjpegGetImageInfo(nvjpeg_handle_, image.data(), image.size(), &channels, &subsampling, widths, heights));
    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (out_fmt_ == NVJPEG_OUTPUT_RGBI || out_fmt_ == NVJPEG_OUTPUT_BGRI)
    {
        channels = 1;
        mul = 3;
    }
    // in the case of rgb create 3 buffers with sizes of original image
    else if (out_fmt_ == NVJPEG_OUTPUT_RGB ||
             out_fmt_ == NVJPEG_OUTPUT_BGR)
    {
        channels = 3;
        widths[1] = widths[2] = widths[0];
        heights[1] = heights[2] = heights[0];
    }

    // prepare output buffer
    cv::cuda::GpuMat c1(heights[0], widths[0], CV_8UC1), c2(heights[0], widths[0], CV_8UC1), c3(heights[0], widths[0], CV_8UC1);
    iout.channel[0] = (unsigned char*)c1.cudaPtr();
    iout.pitch[0] = c1.step;
    iout.channel[1] = (unsigned char*)c2.cudaPtr();
    iout.pitch[1] = c2.step;
    iout.channel[2] = (unsigned char*)c3.cudaPtr();
    iout.pitch[2] = c3.step;

    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    checkCudaErrors(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

    if(!pipelined)
    {
        checkCudaErrors(cudaEventRecord(startEvent, stream));
        checkCudaErrors(nvjpegDecode(nvjpeg_handle_, nvjepg_state_, image.data(), image.size(), out_fmt_, &iout, stream));
        checkCudaErrors(cudaEventRecord(stopEvent, stream));
    }else{
        checkCudaErrors(cudaEventRecord(startEvent, stream));
        checkCudaErrors(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state_, device_buffer_));
        int buffer_index = 0;
        checkCudaErrors(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, out_fmt_));
        checkCudaErrors(nvjpegJpegStreamParse(nvjpeg_handle_, image.data(), image.size(), 0, 0, jpeg_streams_));
        checkCudaErrors(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state_, pinned_buffers_));
        checkCudaErrors(nvjpegDecodeJpegHost(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, nvjpeg_decode_params_, jpeg_streams_));
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, jpeg_streams_, stream));
        checkCudaErrors(nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_, &iout, stream));
        checkCudaErrors(cudaEventRecord(stopEvent, stream));
    }
    checkCudaErrors(cudaEventSynchronize(stopEvent));

    std::vector<cv::cuda::GpuMat> channel_mats;
    channel_mats.push_back(c1);
    channel_mats.push_back(c2);
    channel_mats.push_back(c3);
    
    cv::cuda::GpuMat result(heights[0], widths[0], CV_8UC3);
    cv::cuda::merge(channel_mats, result);
    if(dst.isGpuMat())
    {
        dst.getGpuMatRef() = result;
    }else if(dst.isMat())
    {
        result.download(dst.getMatRef());
    }else{
        throw std::invalid_argument("unsupport format of cv::OutputArray");
    }

    return true;
}

bool CudaJpegDecode::Decode(const std::vector<std::vector<uchar>> &images, std::vector<cv::OutputArray> &out)
{
}

int CudaJpegDecode::host_malloc(void **p, size_t s, unsigned int f)
{
    return (int)cudaHostAlloc(p, s, f);
}

int CudaJpegDecode::host_free(void *p)
{
    return (int)cudaFreeHost(p);
}

int CudaJpegDecode::dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int CudaJpegDecode::dev_free(void *p)
{
    return (int)dev_free(p);
}

int CudaJpegDecode::ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

int CudaJpegDecode::GpuGetMaxGflopsDeviceId()
{
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        int computeMode = -1, major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited)
        {
            if (major == 9999 && minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc =
                    ConvertSMVer2Cores(major, minor);
            }
            int multiProcessorCount = 0, clockRate = 0;
            checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
            cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess)
            {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if (result == cudaErrorInvalidValue)
                {
                    clockRate = 1;
                }
                else
                {
                    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
                            static_cast<unsigned int>(result), cudaGetErrorName(result));
                    exit(EXIT_FAILURE);
                }
            }
            uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf > max_compute_perf)
            {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        }
        else
        {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() CUDA error:"
                " all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}

bool CudaJpegDecode::DecodePipelined(const std::vector<char> &image, cv::OutputArray dst)
{
}