project(PERSON)
cmake_minimum_required(VERSION 3.5)

add_definitions(-DDEBUG)
add_definitions(-g)

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

#openMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#ncnn
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/include/ncnn)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(person ${SRC_LIST})
target_link_libraries(person ncnn ${OPENCV_LIBS})
set_property(TARGET person PROPERTY CXX_STANDARD 11)