# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)

add_definitions(-DUSE_CUDA)

add_executable(image_classifier "main.cc" "jpeg_handle.cc" "jpeg_handle.h" "jpeg_mem.cc" "jpeg_mem.h"
        image_loader.cc image_loader.h CachedInterpolation.h Callback.h local_filesystem.h local_filesystem.cc parallel_task_callback.h)
target_include_directories(image_classifier PRIVATE ${GLIB_INCLUDE_DIRS})
target_link_libraries(image_classifier PRIVATE onnxruntime ${GLIB_LIBRARIES} ${JPEG_LIBRARIES})

add_executable(resize_image "resize_image_cmd.cc" "jpeg_handle.cc" "jpeg_handle.h" "jpeg_mem.cc" "jpeg_mem.h"
        image_loader.cc image_loader.h CachedInterpolation.h Callback.h local_filesystem.h local_filesystem.cc)
target_link_libraries(resize_image PRIVATE ${JPEG_LIBRARIES} ${PNG_LIBRARIES})

#TODO: remove this part(need to talk to Ryan and find a solution for this)
if(UNIX)
 include_directories("/usr/local/include/onnxruntime")
endif()


add_executable(fns_candy_style_transfer "fns_candy_style_transfer.c")
target_include_directories(fns_candy_style_transfer PRIVATE "${TEST_SRC_DIR}/util/include" ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES})
set_target_properties(fns_candy_style_transfer PROPERTIES FOLDER "ONNXRuntimeTest")