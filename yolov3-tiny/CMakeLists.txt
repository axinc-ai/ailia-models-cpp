cmake_minimum_required(VERSION 3.1)

set (PROJECT_NAME yolov3-tiny)
set (SRC_FILES ${PROJECT_NAME}.cpp)
set (SRC_FILES ${SRC_FILES} ../util/utils.cpp)
set (SRC_FILES ${SRC_FILES} ../util/detector_utils.cpp)
set (SRC_FILES ${SRC_FILES} ../util/webcamera_utils.cpp)
set (INCLUDE_PATH $ENV{AILIA_LIBRARY_PATH}/include ../util)
if(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/windows/x64)
elseif(APPLE)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/mac)
else(WIN32)
    set (LIBRARY_PATH $ENV{AILIA_LIBRARY_PATH}/linux)
endif(WIN32)
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

project(${PROJECT_NAME} CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${INCLUDE_PATH})
link_directories(${OpenCV_LIBRARY_DIRS} ${LIBRARY_PATH})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PROJECT_NAME} ailia ${OpenCV_LIBRARIES})
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
