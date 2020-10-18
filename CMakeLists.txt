cmake_minimum_required(VERSION 3.18)
project(NVRTC_Table LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

include_directories(.)

find_package(CUDAToolkit REQUIRED)

add_executable(NVRTC_Table
        DeviceTable.h
        Source.cpp
        Table.cpp
        Table.h)

target_link_libraries(NVRTC_Table CUDA::cudart CUDA::nvrtc)