cmake_minimum_required(VERSION 3.12)
project(Panorama2Cubemap C CXX)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/home/flo/Workspace/darknet)

add_executable(Panorama2Cubemap main.cpp)

target_link_libraries(Panorama2Cubemap -lm -lpthread -ldarknet ${OpenCV_LIBS})