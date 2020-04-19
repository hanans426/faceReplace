cmake_minimum_required(VERSION 3.14)
project(faceReplace)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( ${PROJECT_SOURCE_DIR}/include/)

add_executable(${PROJECT_NAME}
        src/facialLandmarkDetection.cpp
        src/drawLandmarks.hpp
        src/main.cpp)

target_link_libraries( ${PROJECT_NAME}
        PRIVATE
        ${OpenCV_LIBS}
        )




