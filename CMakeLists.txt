cmake_minimum_required(VERSION 2.8)
project(project)
find_package(OpenCV REQUIRED)

#include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${PTHREADS_INCLUDE_DIR})
#include_directories(${CMAKE_SOURCE_DIR}/library)

add_executable(project main.cpp)

target_link_libraries(project opencvblobslib)
target_link_libraries(project ${OpenCV_LIBS})

#target_link_libraries(project ${CMAKE_THREAD_LIBS_INIT})


#target_link_libraries(project ${OpenCV_LIBS})
