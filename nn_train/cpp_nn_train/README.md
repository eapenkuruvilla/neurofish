# C++ Batch Loaders for Neural Network Training

High-performance multi-threaded data loading pipelines for both **NNUE** and **DNN** training that can saturate high-end GPUs even with limited CPU cores.

## Performance

| Loader Type | CPU Cores Needed | GPU Utilization |
|-------------|------------------|-----------------|
| Python DataLoader | 10-16 | ~80-95% |
| **C++ Batch Loader** | **4-6** | **~80-95%** |

The C++ loaders achieve the same GPU utilization with **2-3x fewer CPU cores** because:
- Native C++ is 10-50x faster than Python for data parsing
- Multi-threaded decompression and batch preparation
- Zero-copy transfer to PyTorch via numpy arrays
- Pre-sorted indices for coalesced sparse tensors

## Supported Network Types

| Network | Input Features | Batch Format | Library |
|---------|---------------|--------------|---------|
| NNUE | 40960 (HalfKP) | white_sparse, black_sparse, stm, targets | `libbatch_loader.so` |
| DNN | 768 | features_sparse, targets | `libdnn_batch_loader.so` |

## Prerequisites

- CMake 3.14+
- C++17 compiler (g++ 7+ or clang++ 5+)
- libzstd development files
- pthreads

### Ubuntu/Debian
```bash
sudo apt-get install cmake build-essential libzstd-dev pkg-config
```

### Fedora/RHEL
```bash
sudo dnf install cmake gcc-c++ libzstd-devel pkgconfig
```

### macOS
```bash
brew install cmake zstd pkg-config
```

## Building

### Build Both Loaders (Recommended)
```bash
cd cpp_batch_loader
chmod +x build.sh
./build.sh
```

### Build Specific Loader
```bash
./build.sh nnue    # Build only NNUE loader
./build.sh dnn     # Build only DNN loader
./build.sh clean   # Clean build directory
```

### Manual Build with CMake
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

This creates:
- `libbatch_loader.so` - NNUE batch loader
- `libdnn_batch_loader.so` - DNN batch loader

## Installation

Copy the libraries to your project directory:
```bash
cp cpp_batch_loader/build/libbatch_loader.so .
cp cpp_batch_loader/build/libdnn_batch_loader.so .
```

Or add to library path:
```bash
export LD_LIBRARY_PATH=/path/to/cpp_batch_loader/build:$LD_LIBRARY_PATH
```

Also copy the Python wrappers:
```bash
cp cpp_batch_loader/nnue_batch_loader.py .
cp cpp_batch_loader/dnn_batch_loader.py .
```

## Usage

### With nn_train.py (Automatic)

The training script automatically uses the appropriate C++ loader if available:

```python
# In nn_train.py, these are the defaults:
CPP_LOADER = True   # Enable C++ loader for both NNUE and DNN
SPARSE_COO = True   # Fallback if C++ loader not available
```

Then run training as usual:
```bash
# NNUE training
python nn_train.py --nn-type NNUE --data-dir data/nnue --num-workers 4

# DNN training
python nn_train.py --nn-type DNN --data-dir data/dnn --num-workers 4
```

If the C++ library is found, you'll see:
```
Training mode: C++ batch loader (high performance)
```

If not found, it falls back to Python with a note:
```
Training mode: Sparse COO tensors (dense gradients)
Note: C++ loader not available, falling back to Python loader
      Build with: cd cpp_batch_loader && ./build.sh
```

### Standalone NNUE Usage

```python
from nn_train.nnue_batch_loader import CppBatchLoader
import torch

loader = CppBatchLoader(
    shard_paths=["data/nnue/shard_001.bin.zst", "data/nnue/shard_002.bin.zst"],
    batch_size=16384,
    num_workers=4,  # C++ threads (not Python processes)
    num_features=40960,  # HalfKP feature count
    shuffle=True,
    device=torch.device("cuda")
)

for white_sparse, black_sparse, stm, targets in loader:
    outputs = model(white_sparse, black_sparse, stm)
    loss = criterion(outputs, targets)
    # ... training step
```

### Standalone DNN Usage

```python
from nn_train.dnn_batch_loader import DNNCppBatchLoader
import torch

loader = DNNCppBatchLoader(
    shard_paths=["data/dnn/shard_001.bin.zst", "data/dnn/shard_002.bin.zst"],
    batch_size=16384,
    num_workers=4,  # C++ threads (not Python processes)
    num_features=768,  # DNN feature count
    shuffle=True,
    device=torch.device("cuda")
)

for features_sparse, targets in loader:
    outputs = model(features_sparse)
    loss = criterion(outputs, targets)
    # ... training step
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ Shared Library                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │         │
│  │ Read Shard  │  │ Read Shard  │  │ Read Shard  │         │
│  │ Decompress  │  │ Decompress  │  │ Decompress  │         │
│  │ Parse Batch │  │ Parse Batch │  │ Parse Batch │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │  Thread-safe Queue    │                      │
│              │  (Ready Batches)      │                      │
│              └───────────┬───────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           ▼
                    Python (ctypes)
                    torch.sparse_coo_tensor()
                    GPU Training
```

## Binary Shard Formats

### NNUE Format
```
Normal:     [score:int16][stm:uint8][num_white:uint8][white:uint16[]]
            [num_black:uint8][black:uint16[]]
Diagnostic: [0xFF][score:int16][stm:uint8][num_white:uint8][white:uint16[]]
            [num_black:uint8][black:uint16[]][fen_length:uint8][fen_bytes]
```

### DNN Format
```
Normal:     [score:int16][num_features:uint8][features:uint16[]]
Diagnostic: [0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]]
            [fen_length:uint8][fen_bytes]
```

## Configuration Recommendations

### Half A100 (40GB) + 6 vCPUs
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 4 --batch-size 16384

python nn_train.py --nn-type DNN --data-dir data/dnn \
    --num-workers 4 --batch-size 16384
```

### Full A100 (80GB) + 12 vCPUs  
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 8 --batch-size 32768

python nn_train.py --nn-type DNN --data-dir data/dnn \
    --num-workers 8 --batch-size 32768
```

### RTX 4090 + 8 vCPUs
```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue \
    --num-workers 6 --batch-size 16384

python nn_train.py --nn-type DNN --data-dir data/dnn \
    --num-workers 6 --batch-size 16384
```

## Files

| File | Description |
|------|-------------|
| `batch_loader.h` | C header for NNUE loader API |
| `batch_loader.cpp` | C++ NNUE implementation |
| `batch_loader.py` | Python wrapper for NNUE loader |
| `dnn_batch_loader.h` | C header for DNN loader API |
| `dnn_batch_loader.cpp` | C++ DNN implementation |
| `dnn_batch_loader.py` | Python wrapper for DNN loader |
| `CMakeLists.txt` | CMake build configuration |
| `build.sh` | Convenience build script |

## Troubleshooting

### Library not found
```
RuntimeError: Could not find libbatch_loader.so
RuntimeError: Could not find libdnn_batch_loader.so
```
**Solution:** Copy the `.so` files to your project directory, or set:
```bash
export LD_LIBRARY_PATH=/path/to/cpp_batch_loader/build:$LD_LIBRARY_PATH
```

### Decompression errors
```
Decompression failed: ...
```
**Solution:** Ensure shard files are valid zstd-compressed files created by `prepare_data.py`.

### Invalid feature index errors
```
Invalid feature index: 50000 (max: 767)
```
**Solution:** Ensure you're using the correct loader for your data:
- NNUE data → `batch_loader.py` / `libbatch_loader.so` (40960 features)
- DNN data → `dnn_batch_loader.py` / `libdnn_batch_loader.so` (768 features)

### Still low GPU utilization
If GPU utilization is still low with the C++ loader:
1. Try increasing `--num-workers` (up to number of CPU cores - 2)
2. Check if disk I/O is the bottleneck (use SSD)
3. Monitor with `htop` to see if CPU cores are saturated

### Compilation errors
If CMake fails to find zstd:
```bash
# Ubuntu/Debian
sudo apt-get install libzstd-dev pkg-config

# Then rebuild
cd build && cmake .. && make -j$(nproc)
```

## Network Architecture Comparison

| Aspect | NNUE | DNN |
|--------|------|-----|
| Input Size | 40960 (HalfKP) | 768 |
| Features per Position | ~30 white + ~30 black | ~30 total |
| Perspective | White & Black separate | Single |
| Side-to-Move in Output | Yes | No |
| Hidden Layers | 256 → 32 → 32 → 1 | 1024 → 256 → 32 → 1 |
| First Layer | Shared transformer | Standard linear |

## Training Modes

Both loaders support sparse COO tensors which provide:
- **Memory Efficiency**: ~2MB per batch vs 1.3GB for dense one-hot vectors
- **Dense Gradients**: Good convergence with standard Adam optimizer
- **Fast Training**: Pre-sorted indices for coalesced tensor creation