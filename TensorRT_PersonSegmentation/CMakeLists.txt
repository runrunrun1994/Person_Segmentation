project(PersonSegmention)

cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

#CUDA
find_package(CUDA REQUIRED)

#Opencv
set(OPENCV_LIBS "")
if(OpenCV_FOUND)
    message(STATUS "OpenCV3 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    message(STATUS "OpenCV2 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV libs: (${OPENCV_LIBS})")
else()
    message(WARNING "Not find OpenCV. project maybe invalid ")
endif()

#TensorRT
set(TENSORRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/TensorRT)
message(STATUS ${TENSORRT_ROOT}) 
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
         HINTS ${TENSORRT_ROOT}
         PATH_SUFFIXES include/)

message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
            HINTS ${TENSORRT_ROOT} 
            PATH_SUFFIXES lib)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
            HINTS ${TENSORRT_ROOT}
            PATH_SUFFIXES lib)

find_library(TENSORRT_LIBRARY_PARSER nvparsers
            HINTS ${TENSORRT_ROOT}
            PATH_SUFFIXES lib)

find_library(TENSORRT_LIBRARY_ONNXOARSER nvonnxparser
            HINTS ${TENSORRT_ROOT}
            PATH_SUFFIXES lib)

find_library(TENSORRT_LIBRARY_MYELIN myelin
            HINTS ${TENSORRT_ROOT}
            PATH_SUFFIXES lib)

set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${TENSORRT_LIBRARY_ONNXOARSER} ${TENSORRT_LIBRARY_PARSER} )

message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
link_directories(/home/dell/workspace/CODE/C++/Person_Segmentation/TensorRT_PersonSegmentation/third_party/TensorRT/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TENSORRT_ROOT}/include)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(person ${SRC_LIST})
target_link_libraries(person ${TENSORRT_LIBRARY})
target_link_libraries(person ${CUDA_LIBRARY})
target_link_libraries(person ${CUDA_CUDART_LIBRARY})
target_link_libraries(person ${OPENCV_LIBS})
set_property(TARGET person PROPERTY CXX_STANDARD 11)