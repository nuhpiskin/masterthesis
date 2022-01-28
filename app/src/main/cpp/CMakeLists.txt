# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)
project(crackfinder)

set( CMAKE_CXX_FLAGS "-std=c++17 -Wall" )
set( CMAKE_CXX_FLAGS_DEBUG "-g -std=c++17 -Wall " )
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210525-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library( # Sets the name of the library.
        crackfinder

        # Sets the library as a shared library.
        SHARED

        crackdetection.cpp
        crackfinder.cpp
        ndkcamera.cpp
        )
find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)
find_library(jnigraphics-lib jnigraphics)
target_link_libraries( # Specifies the target library.
        crackfinder
        ncnn
        # Links the target library to the log library
        # included in the NDK.
        ${OpenCV_LIBS}
        ${jnigraphics-lib}
        ${log-lib}
        camera2ndk
        mediandk)
