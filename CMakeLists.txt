cmake_minimum_required(VERSION 3.12)
project(Panorama2Cubemap C CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(/usr/include/libxml2)

include_directories("${PROJECT_BINARY_DIR}")
#include_directories(-L/usr/local/lib -lxml2 -lz -l)
link_directories(/home/flo/Workspace/darknet)
link_directories(/usr/local/cuda/lib64)



add_executable(Panorama2Cubemap main.cpp Detectors/opencv_detect.cpp
        cubetransform.cpp cubetransform.h Detectors/detect.cpp Detectors/detect.h RegionTracker/tracking.cpp OtherTracking/opencv_tracking.cpp OtherTracking/opencv_tracking.h OtherTracking/PanoramaTracking.cpp OtherTracking/PanoramaTracking.h OtherTracking/PanoramaTrackingImplementations.cpp OtherTracking/PanoramaTrackingImplementations.h RegionTracker/TrackingHelpers.h Detectors/testMaskRCNN.cpp Detectors/testMaskRCNN.h RegionTracker/TrackingHelpers.cpp)

target_link_libraries(Panorama2Cubemap -lm -lpthread -ldarknet ${OpenCV_LIBS} -L/usr/local/lib -lxml2 -lz -lm)
