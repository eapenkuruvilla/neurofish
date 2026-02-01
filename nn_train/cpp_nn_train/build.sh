#!/bin/bash
# build.sh - Build the NNUE and DNN batch loaders
#
# Usage:
#   ./build.sh           # Build both loaders
#   ./build.sh clean     # Clean build directory
#   ./build.sh nnue      # Build only NNUE loader
#   ./build.sh dnn       # Build only DNN loader

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[*]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for g++
    if ! command -v g++ &> /dev/null; then
        print_error "g++ not found. Please install build-essential."
        exit 1
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found. Please install cmake."
        exit 1
    fi
    
    # Check for pkg-config
    if ! command -v pkg-config &> /dev/null; then
        print_error "pkg-config not found. Please install pkg-config."
        exit 1
    fi
    
    # Check for zstd
    if ! pkg-config --exists libzstd; then
        print_error "libzstd not found. Please install libzstd-dev."
        exit 1
    fi
    
    print_status "All dependencies found."
}

# Build using CMake
build_cmake() {
    print_status "Building with CMake..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    
    print_status "Build complete!"
    print_status "Libraries built:"
    ls -la *.so 2>/dev/null || true
}

# Build NNUE loader only (without CMake)
build_nnue_manual() {
    print_status "Building NNUE batch loader..."
    
    mkdir -p "${BUILD_DIR}"
    
    g++ -O3 -march=native -shared -fPIC \
        -o "${BUILD_DIR}/libnnue_batch_loader.so" \
        "${SCRIPT_DIR}/nnue_batch_loader.cpp" \
        -lzstd -lpthread -std=c++17
    
    print_status "NNUE batch loader built: ${BUILD_DIR}/libnnue_batch_loader.so"
}

# Build DNN loader only (without CMake)
build_dnn_manual() {
    print_status "Building DNN batch loader..."
    
    mkdir -p "${BUILD_DIR}"
    
    g++ -O3 -march=native -shared -fPIC \
        -o "${BUILD_DIR}/libdnn_batch_loader.so" \
        "${SCRIPT_DIR}/dnn_batch_loader.cpp" \
        -lzstd -lpthread -std=c++17
    
    print_status "DNN batch loader built: ${BUILD_DIR}/libdnn_batch_loader.so"
}

# Clean build directory
clean() {
    print_status "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
    print_status "Clean complete."
}

# Main
case "${1:-all}" in
    clean)
        clean
        ;;
    nnue)
        check_dependencies
        build_nnue_manual
        ;;
    dnn)
        check_dependencies
        build_dnn_manual
        ;;
    all|*)
        check_dependencies
        build_cmake
        ;;
esac

#echo ""
#print_status "Done!"
echo ""
#echo "To use the loaders, ensure the .so files are in your Python path:"
#echo "  export LD_LIBRARY_PATH=\"${BUILD_DIR}:\$LD_LIBRARY_PATH\""
#echo ""
#echo "Or copy them to the same directory as your Python scripts:"
#echo "  cp ${BUILD_DIR}/lib*.so /path/to/your/project/"
echo "Copying ${BUILD_DIR}/lib*.so to the libs directory.."
cp ${BUILD_DIR}/lib*.so ../../../libs/
print_status "Done!"
echo ""
