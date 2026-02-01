#!/bin/bash
# build.sh - Build script for chess_cpp Python extension
#
# This script downloads dependencies and builds the C++ chess library.
# Usage: ./build.sh [--clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "${GREEN}==>${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

echo_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Clean build if requested
if [ "$1" == "--clean" ]; then
    echo_step "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info chess_cpp*.so chess-library/
    echo "Clean complete."
    exit 0
fi

# Check dependencies
echo_step "Checking dependencies..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo_error "Python 3 is required but not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# Check for pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo_error "pip is required but not found"
    exit 1
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo_error "CMake is required but not found"
    echo "Install with: sudo apt-get install cmake (Ubuntu) or brew install cmake (macOS)"
    exit 1
fi
echo "  CMake: $(cmake --version | head -n1)"

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    CXX_COMPILER="g++"
    echo "  C++ Compiler: $(g++ --version | head -n1)"
elif command -v clang++ &> /dev/null; then
    CXX_COMPILER="clang++"
    echo "  C++ Compiler: $(clang++ --version | head -n1)"
else
    echo_error "C++ compiler (g++ or clang++) is required but not found"
    exit 1
fi

# Install pybind11 if needed
echo_step "Checking pybind11..."
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "  Installing pybind11..."
    pip3 install pybind11
fi
echo "  pybind11: $(python3 -c 'import pybind11; print(pybind11.__version__)')"

# Download chess-library if not present
echo_step "Checking chess-library..."
if [ ! -d "chess-library" ]; then
    echo "  Downloading Disservin/chess-library..."
    git clone --depth 1 https://github.com/Disservin/chess-library.git
else
    echo "  chess-library already present"
fi

# Create build directory
echo_step "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo_step "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Build
echo_step "Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy the built module
echo_step "Installing module..."
cp chess_cpp*.so ../
cd ..

# Verify the build
echo_step "Verifying build..."
if python3 -c "import chess_cpp; print(f'chess_cpp module loaded successfully')"; then
    echo -e "${GREEN}Build successful!${NC}"
    echo ""
    echo "The chess_cpp module has been built and is ready to use."
    #echo "To use it in your chess engine:"
    echo "Moving chess_cpp*.so to your libs directory"
    mv chess_cpp*.so ../libs/
    # echo "  2. Copy cached_board.py to your engine directory (replacing the old one)"
    #echo "  2. Run your engine - it will automatically use the C++ backend"
else
    echo_error "Build verification failed"
    exit 1
fi
