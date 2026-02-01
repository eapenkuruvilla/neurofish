cmake_minimum_required(VERSION 3.14)
project(chess_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Include chess-library header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/chess-library/include)

# Create the Python module
pybind11_add_module(chess_cpp
    chess_wrapper.cpp
    bindings.cpp
)

# Set output name
set_target_properties(chess_cpp PROPERTIES
    OUTPUT_NAME "chess_cpp"
    PREFIX ""
)

# Link any required libraries (chess-library is header-only)
target_compile_options(chess_cpp PRIVATE -O3 -march=native)

# Installation
install(TARGETS chess_cpp LIBRARY DESTINATION .)
