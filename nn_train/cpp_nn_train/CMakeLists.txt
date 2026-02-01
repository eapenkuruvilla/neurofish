cmake_minimum_required(VERSION 3.14)
project(batch_loaders VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find zstd library
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Find threads
find_package(Threads REQUIRED)

# NNUE Batch Loader
add_library(nnue_batch_loader SHARED
    nnue_batch_loader.cpp
)
target_include_directories(nnue_batch_loader PRIVATE ${ZSTD_INCLUDE_DIRS})
target_link_libraries(nnue_batch_loader PRIVATE 
    ${ZSTD_LIBRARIES}
    Threads::Threads
)
set_target_properties(nnue_batch_loader PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "nnue_batch_loader"
)

# DNN Batch Loader
add_library(dnn_batch_loader SHARED
    dnn_batch_loader.cpp
)
target_include_directories(dnn_batch_loader PRIVATE ${ZSTD_INCLUDE_DIRS})
target_link_libraries(dnn_batch_loader PRIVATE 
    ${ZSTD_LIBRARIES}
    Threads::Threads
)
set_target_properties(dnn_batch_loader PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "dnn_batch_loader"
)

# Installation
install(TARGETS nnue_batch_loader dnn_batch_loader
    LIBRARY DESTINATION lib
)
install(FILES nnue_batch_loader.h dnn_batch_loader.h
    DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Batch Loaders Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "ZSTD include: ${ZSTD_INCLUDE_DIRS}")
message(STATUS "ZSTD libraries: ${ZSTD_LIBRARIES}")
message(STATUS "")
