# Utility Scripts for NeuroFish Chess Engine

Helper scripts for model analysis, Stockfish evaluation, and development workflow automation.

## Utilities Overview

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `quantize_analysis.py` | Analyze INT8/INT16 quantization potential | NumPy, PyTorch (optional) |
| `sf_static_eval.py` | Pre-compute Stockfish NNUE evaluations | Stockfish, shard_io |
| `sf_static_eval.sh` | Interactive Stockfish position evaluator | Stockfish |
| `wait.sh` | Wait for data preparation to complete | None |

## Prerequisites

- Python 3.8+
- NumPy
- Stockfish (for evaluation scripts)
- PyTorch (optional, for loading .pt models)
- zstandard (optional, for reading shard files)

### Installing Dependencies

```bash
# Core dependencies
pip install numpy

# Optional: For full quantize_analysis.py functionality
pip install torch zstandard

# Stockfish
sudo apt-get install stockfish  # Ubuntu/Debian
brew install stockfish          # macOS
```

## Quantization Analysis (`quantize_analysis.py`)

Analyzes the potential for INT8/INT16 quantization of NNUE neural networks to improve inference speed.

### Features

- Weight distribution analysis per layer
- Quantization error comparison (INT8 vs INT16 vs FP32)
- Output accuracy degradation testing
- Speed benchmarks for quantized inference
- Hardware-specific speedup estimates (AVX-512, VNNI, AVX2)

### Architecture Analyzed

```
NNUE Network:
  ft: 40960 → 256  (Feature Transformer)
  l1: 512 → 32     (Concatenated white+black accumulators)
  l2: 32 → 32
  l3: 32 → 1
```

### Usage

```bash
# Synthetic weights (no dependencies required)
python utils/quantize_analysis.py --synthetic

# Real model weights
python utils/quantize_analysis.py --model model/nnue.pt --data-dir data/nnue

# NumPy weights file
python utils/quantize_analysis.py --weights nnue_weights.npz --data-dir data/nnue

# Skip accuracy or benchmark tests
python utils/quantize_analysis.py --model model/nnue.pt --skip-accuracy
python utils/quantize_analysis.py --model model/nnue.pt --skip-benchmark

# Save weights for later analysis
python utils/quantize_analysis.py --model model/nnue.pt --save-weights nnue_weights.npz
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to PyTorch model (.pt) | None |
| `--weights` | Path to NumPy weights (.npz) | None |
| `--synthetic` | Use synthetic weights | False |
| `--data-dir` | NNUE shard directory for testing | None |
| `--positions` | Number of test positions | 10000 |
| `--bench-iterations` | Benchmark iterations | 10000 |
| `--skip-accuracy` | Skip accuracy testing | False |
| `--skip-benchmark` | Skip speed benchmarks | False |
| `--save-weights` | Save weights to .npz file | None |

### Output

```
================================================================================
WEIGHT DISTRIBUTION ANALYSIS
================================================================================
Layer           Shape           Min        Max        Mean       Std        Sparsity
--------------------------------------------------------------------------------
ft.weight       (256, 40960)   -0.0823    0.0812    -0.0001    0.0200     0.00%
l1.weight       (32, 512)      -0.1245    0.1198     0.0002    0.0443     0.00%
...

================================================================================
QUANTIZATION ERROR ANALYSIS
================================================================================
Layer           INT8 MSE       INT8 Max      INT16 MSE      INT16 Max
--------------------------------------------------------------------------------
l1.weight       2.34e-07       0.000892      3.58e-11       0.000014
...

================================================================================
RECOMMENDATIONS
================================================================================
1. ACCURACY ASSESSMENT:
   [OK] INT16 L1-only: EXCELLENT (<1 cp avg error)
   [OK] INT8 L1-only: GOOD (3.2 cp avg error)
...
```

## Stockfish Evaluation File Builder (`sf_static_eval.py`)

Pre-computes Stockfish NNUE static evaluations for test positions, enabling fast NN-vs-Stockfish comparisons without repeated subprocess calls.

### Binary Output Format

```
Header:
    [num_records:uint32]
Each record:
    [fen_length:uint8][fen_bytes:char[]][sf_eval_cp:int16][shard_cp:int16]
```

### Usage

```bash
# Default: 10000 positions from data/dnn
python utils/sf_static_eval.py

# Custom parameters
python utils/sf_static_eval.py --num-positions 5000 --output data/sf_eval.bin

# Specify data directory and Stockfish path
python utils/sf_static_eval.py --data-dir data/dnn --stockfish /usr/local/bin/stockfish

# Set random seed for reproducibility
python utils/sf_static_eval.py --num-positions 10000 --seed 123
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num-positions`, `-n` | Positions to evaluate | 10000 |
| `--output`, `-o` | Output file path | `data/sf_nnue_static_eval.bin` |
| `--data-dir`, `-d` | Shard files directory | `data/dnn` |
| `--stockfish`, `-s` | Stockfish binary path | `$STOCKFISH_PATH` or `stockfish` |
| `--seed` | Random seed | 42 |

### Output

```
======================================================================
Stockfish Static Evaluation File Builder
======================================================================
Num positions:  10000
Output file:    data/sf_nnue_static_eval.bin
Data directory: data/dnn
Stockfish:      stockfish
Random seed:    42
======================================================================

Testing Stockfish...
✓ Stockfish working (startpos eval: +26 cp)

Collecting positions from shards...
Found 5 shard files in data/dnn
Scanning shards for diagnostic records: 100%|████| 5/5 [00:12<00:00]
Collected 12000 unique positions

Evaluating positions with Stockfish...
Stockfish evaluation: 100%|██████████████| 10000/10000 [08:32<00:00]

Successfully evaluated 10000 positions
Wrote 10000 records to data/sf_nnue_static_eval.bin
```

## Interactive Stockfish Evaluator (`sf_static_eval.sh`)

Interactive shell script for quickly evaluating positions with Stockfish's NNUE static evaluation.

### Usage

```bash
./utils/sf_static_eval.sh
```

### Example Session

```
Stockfish Position Evaluator (depth 0)
Enter FEN positions to evaluate, or 'quit'/'exit' to end.

FEN> rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
NNUE Score: +0.26

FEN> r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4
NNUE Score: +0.45

FEN> quit
Goodbye!
```

### Features

- Uses Stockfish's built-in `eval` command
- Returns NNUE evaluation in pawns (e.g., +0.26 = 26 centipawns)
- Depth 0 (static evaluation only, no search)

## Process Wait Script (`wait.sh`)

Waits for `prepare_data.py` to finish, then automatically starts model training.

### Usage

```bash
# Start data preparation in background
python prepare_data.py --input games.pgn --output data/dnn &

# Run wait script (checks every 10 minutes)
./utils/wait.sh
```

### Behavior

1. Checks every 10 minutes (600 seconds) for running `prepare_data` processes
2. Once no processes are found, runs `defunc_build_model.py`
3. Useful for long-running data preparation jobs

### Customization

Edit the script to change:
- Check interval: `sleep 600` (default: 10 minutes)
- Follow-up command: `python defunc_build_model.py`

## Typical Workflows

### Quantization Analysis Workflow

```bash
# 1. Train an NNUE model
python nn_train.py --nn-type NNUE --epochs 10

# 2. Analyze quantization potential
python utils/quantize_analysis.py --model model/nnue.pt --data-dir data/nnue

# 3. If results are good, implement quantized inference
# (See recommendations in output)
```

### Stockfish Comparison Workflow

```bash
# 1. Pre-compute Stockfish evaluations (one-time)
python utils/sf_static_eval.py --num-positions 10000

# 2. Run NN-vs-Stockfish test (uses pre-computed file)
python test/nn_tests.py --nn-type NNUE --test 4 --positions 10000
```

### Long Data Preparation Workflow

```bash
# Terminal 1: Start data preparation
python prepare_data.py --input large_dataset.pgn --output data/nnue &

# Terminal 2: Wait and auto-train
./utils/wait.sh
```

## Files Overview

```
utils/
├── quantize_analysis.py    # INT8/INT16 quantization analysis
├── sf_static_eval.py       # Pre-compute Stockfish evaluations
├── sf_static_eval.sh       # Interactive Stockfish evaluator
├── wait.sh                 # Process wait automation
└── README.md               # This file
```

## Troubleshooting

### "PyTorch not available" error

```bash
# Install PyTorch for --model support
pip install torch

# Or use --weights with pre-exported NumPy file
python utils/quantize_analysis.py --weights nnue_weights.npz
```

### "shard_io not available" error

```bash
# Install zstandard for shard reading
pip install zstandard

# Or use synthetic data
python utils/quantize_analysis.py --synthetic
```

### Stockfish not found

```bash
# Set environment variable
export STOCKFISH_PATH=/usr/local/bin/stockfish

# Or specify path directly
python utils/sf_static_eval.py --stockfish /path/to/stockfish
```

### Interactive evaluator shows no output

The `sf_static_eval.sh` script requires Stockfish to support the `eval` command. Ensure you have a recent version:

```bash
stockfish --version
# Should be Stockfish 14 or later for NNUE eval support
```

## License

These utility scripts are part of the NeuroFish chess engine project and follow the same license terms.
