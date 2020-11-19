project(PERSON)
cmake_minimum_required(VERSION 3.5)

add_definitions(-DDEBUG)
add_definitions(-g)


#torch
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)
find_package(Torch REQUIRED)
message(STATUS "TORCH FOUND!")
message(STATUS "Torch include: " ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch library: " ${TORCH_LIBRARIES})

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(person ${SRC_LIST})
target_link_libraries(person ${TORCH_LIBRARIES})
target_link_libraries(person ${OPENCV_LIBS})
set_property(TARGET person PROPERTY CXX_STANDARD 14)