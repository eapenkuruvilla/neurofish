# Chess Engine C++ Backend Integration

This package provides a high-performance C++ chess library backend for your Python chess engine, replacing the pure-Python `python-chess` library for performance-critical operations.

## Performance Gains

Expected speedup: **3-5x improvement in NPS (nodes per second)**

The C++ backend accelerates:
- Legal move generation
- Move making/unmaking
- Check detection
- Game state queries (checkmate, stalemate, etc.)
- Bitboard operations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Chess Engine                     │
│                      (engine.py)                         │
├─────────────────────────────────────────────────────────┤
│                     CachedBoard                          │
│               (Python caching layer)                     │
│  • Zobrist hash caching (Polyglot-compatible)           │
│  • Material evaluation caching                           │
│  • Move info pre-computation                             │
├───────────────────────┬─────────────────────────────────┤
│    chess_cpp (C++)    │   python-chess (fallback)       │
│  • Move generation    │  • Used when C++ unavailable    │
│  • Position queries   │  • Automatic fallback           │
│  • Game state         │                                  │
├───────────────────────┴─────────────────────────────────┤
│              Disservin/chess-library                     │
│            (Header-only C++ library)                     │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- **Python 3.8+**
- **C++17 compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14+**
- **pybind11** (installed automatically)
- **Git** (for downloading chess-library)

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev
pip install pybind11
```

**macOS:**
```bash
xcode-select --install
brew install cmake
pip install pybind11
```

**Windows:**
```powershell
# Install Visual Studio Build Tools with C++ workload
# Install CMake from https://cmake.org/download/
pip install pybind11
```

## Quick Start

### Option 1: Using the Build Script (Recommended)

```bash
cd cpp_board
./build.sh
```

The script will:
1. Check all dependencies
2. Download Disservin's chess-library
3. Build the C++ extension
4. Verify the installation

### Option 2: Manual Build

```bash
# 1. Clone chess-library
git clone --depth 1 https://github.com/Disservin/chess-library.git

# 2. Create build directory
mkdir build && cd build

# 3. Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. Build
cmake --build . --config Release -j4

# 5. Copy the module
cp chess_cpp*.so ../
```

### Option 3: Using pip

```bash
pip install .
```

## Integration

### Step 1: Copy Files

Copy these files to your chess engine directory:
- `chess_cpp*.so` (the compiled extension)
- `cached_board.py` (the updated Python wrapper)

### Step 2: Verify Import

```python
# Test the installation
import chess_cpp
print("C++ backend loaded successfully!")

# The CachedBoard will automatically detect and use the C++ backend
from cached_board import CachedBoard, HAS_CPP_BACKEND
print(f"Using C++ backend: {HAS_CPP_BACKEND}")
```

### Step 3: Use as Before

Your existing code requires **no changes**! The `CachedBoard` class automatically uses the C++ backend when available:

```python
from cached_board import CachedBoard

board = CachedBoard()  # Uses C++ backend automatically
board.push(chess.Move.from_uci("e2e4"))
moves = board.get_legal_moves_list()  # 3-5x faster!
```

## API Compatibility

The `CachedBoard` class maintains full API compatibility with `python-chess`:

| Method | Description | Cached |
|--------|-------------|--------|
| `push(move)` | Make a move | - |
| `pop()` | Unmake a move | - |
| `get_legal_moves_list()` | Get all legal moves | ✓ |
| `is_check()` | Check if in check | ✓ |
| `is_checkmate()` | Check for checkmate | ✓ |
| `is_game_over()` | Check if game is over | ✓ |
| `zobrist_hash()` | Get Zobrist hash | ✓ |
| `material_evaluation()` | Material + PST eval | ✓ |
| `precompute_move_info()` | Batch move analysis | ✓ |
| `is_capture_cached(move)` | Check if capture | ✓ |
| `gives_check_cached(move)` | Check if gives check | ✓ |

## Benchmarking

Run a quick benchmark to verify the speedup:

```python
import time
from cached_board import CachedBoard, HAS_CPP_BACKEND

def benchmark(iterations=10000):
    board = CachedBoard()
    
    start = time.perf_counter()
    for _ in range(iterations):
        moves = board.get_legal_moves_list()
        if moves:
            board.push(moves[0])
            board.pop()
    elapsed = time.perf_counter() - start
    
    print(f"Backend: {'C++' if HAS_CPP_BACKEND else 'Python'}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Ops/sec: {iterations/elapsed:.0f}")

benchmark()
```

## Troubleshooting

### "C++ backend not available" warning

The engine will work with python-chess as a fallback. To enable C++:

1. **Missing compiler:**
   ```bash
   # Ubuntu
   sudo apt-get install build-essential
   # macOS
   xcode-select --install
   ```

2. **Missing CMake:**
   ```bash
   # Ubuntu
   sudo apt-get install cmake
   # macOS
   brew install cmake
   ```

3. **Wrong Python version:**
   - Rebuild with the same Python that runs your engine
   - Check with `python3 --version`

### Import errors

```bash
# Check the module was built
ls -la chess_cpp*.so

# Check Python can find it
python3 -c "import chess_cpp; print('OK')"
```

### Hash mismatch warnings

The C++ backend uses a different internal Zobrist hash. For compatibility with Polyglot opening books and transposition tables, we maintain a parallel `python-chess` board for hash computation. This is automatic and transparent.

## Files Overview

```
chess_cpp_integration/
├── chess_wrapper.hpp      # C++ wrapper header
├── chess_wrapper.cpp      # C++ wrapper implementation
├── bindings.cpp           # pybind11 Python bindings
├── CMakeLists.txt         # CMake build configuration
├── setup.py               # Python package setup
├── build.sh               # Build script (Linux/macOS)
├── cached_board.py        # Updated CachedBoard with C++ support
├── README.md              # This file
└── chess-library/         # (downloaded) Disservin's library
    └── include/
        └── chess.hpp      # Header-only chess library
```

## Advanced: Custom Build Options

### Debug Build

```bash
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

### Specific Python Version

```bash
cmake .. -DPYTHON_EXECUTABLE=/path/to/python3.11
```

### Cross-compilation

For building on one system to run on another (e.g., building on x86 for ARM):

```bash
cmake .. -DCMAKE_CXX_FLAGS="-march=armv8-a"
```

## Performance Tips

1. **Compile with optimizations:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
   ```

2. **Use the caching layer:** Don't bypass `CachedBoard` - its caching provides significant speedup for repeated queries.

3. **Batch move operations:** Use `precompute_move_info()` to analyze all moves at once instead of individual queries.

4. **Minimize Python-C++ boundary crossings:** Group related operations when possible.

## License

- This integration code: MIT License
- chess-library (Disservin): MIT License
- pybind11: BSD License

## Credits

- [Disservin/chess-library](https://github.com/Disservin/chess-library) - Fast, header-only C++ chess library
- [pybind11](https://github.com/pybind/pybind11) - Seamless C++/Python interoperability
- [python-chess](https://github.com/niklasf/python-chess) - The original Python chess library
