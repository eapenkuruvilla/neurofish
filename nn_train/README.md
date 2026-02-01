# nn_train

Neural network training module for chess position evaluation. Supports training both **NNUE** (Efficiently Updatable Neural Network) and **DNN** (Deep Neural Network) architectures from Lichess game data.

## Overview

This module provides a complete pipeline for:

1. **Data Preparation** — Extract position features from PGN files into efficient binary shards
2. **High-Performance Loading** — C++ multi-threaded batch loaders with Python bindings
3. **Model Training** — PyTorch training with sparse tensor support and multiple optimization strategies

## Directory Structure

```
nn_train/
├── nn_train.py              # Main training script
├── prepare_data.py          # PGN → binary shard conversion
├── shard_io.py              # Shared I/O utilities for binary shards
├── nnue_batch_loader.py     # Python wrapper for C++ NNUE loader
├── dnn_batch_loader.py      # Python wrapper for C++ DNN loader
├── __init__.py
└── cpp_nn_train/            # C++ batch loader implementations
    ├── nnue_batch_loader.cpp
    ├── nnue_batch_loader.h
    ├── dnn_batch_loader.cpp
    ├── dnn_batch_loader.h
    ├── CMakeLists.txt
    └── build.sh
```

## Installation

### Dependencies

```bash
pip install torch numpy zstandard python-chess
```

### Building C++ Batch Loaders (Recommended)

The C++ loaders provide 2-3x better performance than pure Python:

```bash
cd cpp_nn_train
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Usage

### 1. Prepare Training Data

Convert Lichess PGN files to binary shards:

```bash
python prepare_data.py \
    --pgn-zst-file lichess_2023_01.pgn.zst \
    --output-dir data/ \
    --output-file-prefix jan2023
```

This creates:
- `data/dnn/jan2023_0001.bin.zst, ...` — DNN training shards
- `data/nnue/jan2023_0001.bin.zst, ...` — NNUE training shards
- `data/jan2023_progress.json` — Progress file for resume capability

Options:
- `--positions-per-shard` — Positions per shard file (default: 1,000,000)
- `--resume` — Resume interrupted processing

### 2. Train Models

**Train NNUE:**

```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue
```

**Train DNN:**

```bash
python nn_train.py --nn-type DNN --data-dir data/dnn
```

**Resume from checkpoint:**

```bash
python nn_train.py --nn-type NNUE --data-dir data/nnue --resume model/nnue.pt
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--nn-type` | Required | `NNUE` or `DNN` |
| `--data-dir` | Required | Directory containing shard files |
| `--checkpoint` | `model/{nn_type}.pt` | Checkpoint save path |
| `--resume` | None | Resume from checkpoint |
| `--epochs` | 600 | Number of training epochs |
| `--batch-size` | 16384 | Batch size |
| `--lr` | 8.75e-4 | Learning rate |
| `--positions-per-epoch` | 100M | Positions per epoch |
| `--val-size` | 1M | Validation set size |
| `--early-stopping` | 10 | Early stopping patience |
| `--num-workers` | 4 | DataLoader worker processes |
| `--prefetch-factor` | 4 | Batches to prefetch per worker |

## Architecture

### Network Architectures

**NNUE (40,960 → 256 → 32 → 32 → 1)**
- HalfKP feature encoding (king-piece relative positions)
- Separate white/black feature transformers
- Side-to-move aware accumulator concatenation

**DNN (768 → hidden layers → 1)**
- Standard piece-square encoding (12 planes × 64 squares)
- Configurable hidden layer architecture

### Training Modes

Training mode is selected automatically based on availability (highest to lowest priority):

| Mode | Description | Performance |
|------|-------------|-------------|
| **CPP_LOADER** | C++ multi-threaded batch loader | Best (requires compiled `.so`) |
| **SPARSE_COO** | Sparse COO tensors with dense gradients | Good (recommended fallback) |
| **EMBEDDING_BAG** | EmbeddingBag with sparse gradients | Fair (may have convergence issues) |
| **DENSE** | Explicit one-hot vectors | Slow (high memory usage) |

## Binary Shard Format

Shards use zstandard compression for efficient storage.

**DNN Record:**
```
Normal:     [score:int16][num_features:uint8][features:uint16[]]
Diagnostic: [0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]][fen_len:uint8][fen]
```

**NNUE Record:**
```
Normal:     [score:int16][stm:uint8][num_white:uint8][white:uint16[]][num_black:uint8][black:uint16[]]
Diagnostic: [0xFF][score:int16][stm:uint8][num_white:uint8][white:uint16[]][num_black:uint8][black:uint16[]][fen_len:uint8][fen]
```

Diagnostic records (every 1000 positions) include FEN strings for validation.

## Feature Encoding

**DNN Features (768 total):**
- 12 piece planes × 64 squares
- `feature_idx = piece_type * 64 + square`
- Piece order: P=0, N=1, B=2, R=3, Q=4, K=5

**NNUE Features (40,960 total):**
- HalfKP encoding: king square × piece square × piece type/color
- `feature_idx = king_sq * 640 + piece_sq * 10 + (type_idx + color_idx * 5)`

## Configuration

Key training parameters (in `nn_train.py`):

```python
BATCH_SIZE = 16384
LEARNING_RATE = 8.75e-4
POSITIONS_PER_EPOCH = 100_000_000
VALIDATION_SIZE = 1_000_000
EPOCHS = 600
EARLY_STOPPING_PATIENCE = 10
```

## Output

Trained models are saved in inference-compatible format:

```
model/
├── nnue.pt    # NNUE checkpoint
└── dnn.pt     # DNN checkpoint
```

Use with `nn_inference.py` for evaluation.

## Performance Tips

1. **Build C++ loaders** — Provides 2-3x speedup over Python loaders
2. **Use GPU** — Automatically detected, falls back to CPU
3. **Tune workers** — Increase `--num-workers` and `--prefetch-factor` for faster GPUs
4. **Pin memory** — Enabled automatically when using CUDA

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- zstandard
- python-chess
- C++17 compiler (for C++ loaders)
- libzstd-dev (for C++ loaders)

## Related Modules

- `nn_inference` — Model inference and feature extraction
- `config` — Shared configuration (MAX_SCORE, TANH_SCALE)
