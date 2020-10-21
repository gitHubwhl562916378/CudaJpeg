cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(CudaJpeg
        LANGUAGES CXX
        VERSION "1.0"
        DESCRIPTION "transofrm color by cuda"
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

find_package(CUDA REQUIRED)
include_directories(
    ${CUDA_INCLUDE_DIRS}
)
message(satus " cuda root dir ${CUDA_TOOLKIT_ROOT_DIR}/include")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_compile_options(-std=c++11 -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss)
set(CMAKE_BUILD_TYPE "DEBUG")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

file(GLOB SRCS "*.cpp" "../*.h" "../*.cpp")
add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} nvjpeg)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})