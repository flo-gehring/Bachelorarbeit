cmake_minimum_required(VERSION 3.12)
project(Panorama2Cubemap C CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_BINARY_DIR}")
link_directories(/home/flo/Workspace/darknet)



add_executable(Panorama2Cubemap main.cpp opencv_detect.cpp
        cubetransform.cpp cubetransform.h detect.cpp detect.h tracking.cpp opencv_tracking.cpp opencv_tracking.h PanoramaTracking.cpp PanoramaTracking.h PanoramaTrackingImplementations.cpp PanoramaTrackingImplementations.h TrackingHelpers.h testMaskRCNN.cpp testMaskRCNN.h)

target_link_libraries(Panorama2Cubemap -lm -lpthread -ldarknet ${OpenCV_LIBS})
