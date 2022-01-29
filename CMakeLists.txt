project(cppaiprojects)

cmake_minimum_required(VERSION 3.10)
set( CMAKE_BUILD_TYPE Debug )

if( WIN32 )
	set( CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++17 /D_UNICODE /DUNICODE" )
	set( CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1 /std:c++17 /D_UNICODE /DUNICODE" )
	message( "Win settings chosen..." )
elseif( ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" )
	set( CMAKE_CXX_FLAGS "-std=c++17 -Wall" )
	set( CMAKE_CXX_FLAGS_DEBUG "-g -std=c++17 -Wall" )
	message( "Mac settings chosen..." )
elseif( UNIX )
	set( CMAKE_CXX_FLAGS "-std=c++17 -Wall" )
	set( CMAKE_CXX_FLAGS_DEBUG "-g -std=c++17 -Wall " )
	message( "Linux settings chosen..." )
endif()

# Inform CMake where the header files are
include_directories( include )

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/libs/opencv-mobile-4.5.3-ubuntu-1804/lib/cmake/opencv4)
find_package(OpenCV REQUIRED core imgproc highgui video)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/libs/ncnn-20210525-ubuntu-2004/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# Automatically add all *.cpp and *.h files to the project
file ( GLOB SOURCES "./src/*.cpp" "./include/*.h" )
add_executable( ${PROJECT_NAME} ${SOURCES} )

target_link_libraries(cppaiprojects ncnn ${OpenCV_LIBS})
