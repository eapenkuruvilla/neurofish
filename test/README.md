# Test Suite for NeuroFish Chess Engine

Comprehensive test suite for validating the chess engine, neural network evaluation, and training data integrity.

## Test Categories

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `engine_test.py` | Chess engine move quality | WAC, Eigenmann test suites |
| `engine_test_batch.py` | Parameter tuning via engine tests | Batch testing multiple parameter values |
| `nn_tests.py` | Neural network correctness | Accumulator, symmetry, accuracy |
| `data_test.py` | Training data integrity | Feature extraction, Stockfish comparison |
| `stockfish.sh` | ELO measurement | Matches against Stockfish |
| `stockfish_batch.py` | Parameter tuning via ELO | Batch testing against Stockfish |
| `oldfish.sh` | Regression testing | Matches against previous versions |

## Prerequisites

- Python 3.8+
- Trained model in `model/nnue.pt` or `model/dnn.pt`
- [cutechess-cli](https://github.com/cutechess/cutechess) (for ELO tests)
- Stockfish (for comparison tests)

### Installing Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# cutechess-cli (build from source)
sudo apt-get update
sudo apt-get install build-essential cmake qtbase5-dev libqt5svg5-dev
cd ..
git clone https://github.com/cutechess/cutechess.git
cd cutechess && mkdir build && cd build
cmake .. && make
cd ../../neurofish

# Stockfish NNUE evaluations with 10K positions
python -m utils.sf_static_eval
```

## Quick Start

### Run Engine Tests (WAC Suite)

```bash
# Single-threaded
python -m test.engine_test

# Multi-processing
python -m test.engine_test --mp
```

### Run Parameter Tuning

```bash
# Tune parameter using engine tests (fast feedback)
python -m test.engine_test_batch MAX_QS_DEPTH 20 21 22 23 24

# Tune parameter using Stockfish matches (accurate ELO measurement)
python -m test.stockfish_batch FUTILITY_MARGIN 100 150 200 --games 30 --elo 2300
```

### Run Neural Network Tests

```bash
# Interactive FEN evaluation
python -m test.nn_tests --nn-type NNUE --test 0

# Run all tests
python -m test.nn_tests --nn-type NNUE --test 13

# Accumulator correctness
python -m test.nn_tests --nn-type DNN --test 2
```

### Run Data Integrity Tests

```bash
# Auto-detect and verify shards
python -m test.data_test

# Verify specific shard
python -m test.data_test --dnn-shard data/dnn/train_0001.bin.zst
python -m test.data_test --nnue-shard data/nnue/train_0001.bin.zst

# Analyze shard statistics
python -m test.data_test --analyze data/nnue/train_0001.bin.zst --nn-type NNUE
```

### Measure ELO Rating

```bash
# Play against Stockfish at specific ELO
./test/stockfish.sh 2000 30

# Play against previous engine version
./test/oldfish.sh neuro512 old1024 40/120+1 6
```

## Engine Tests (`engine_test.py`)

Tests the engine's tactical and positional move quality using standard test suites.

### Test Suites

| Suite | Positions | Focus | Time/Position |
|-------|-----------|-------|---------------|
| WAC (Win at Chess) | 300 | Tactics | 5 seconds |
| Eigenmann Rapid | 111 | Mixed | 3 seconds |

### Usage

```bash
# Run WAC suite (default)
python -m test.engine_test

# Enable multiprocessing
python -m test.engine_test --mp
```

### Output

```
time_limit=5
total=1, passed=1, success-rate=100.0%
total=2, passed=2, success-rate=100.0%
...
total=300, passed=267, success-rate=89.0%
time-avg=4.12, time-max=5.03
```

## Parameter Tuning (`engine_test_batch.py`)

Tunes engine parameters by running the WAC test suite with different parameter values. Provides quick feedback on parameter impact via success rate comparison.

### Usage

```bash
python -m test.engine_test_batch PARAM_NAME value1 value2 value3 ...

# Examples
python -m test.engine_test_batch MAX_QS_DEPTH 20 21 22 23 24
python -m test.engine_test_batch QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
python -m test.engine_test_batch MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"

# With multiprocessing
python -m test.engine_test_batch FUTILITY_MAX_DEPTH 2 3 4 --mp
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `PARAM_NAME` | Environment variable name to tune | Required |
| `values` | Space-separated values to test | Required |
| `--mp` | Enable multi-processing mode | False |

### Output

The script shows real-time progress with failures highlighted:

```
Testing MAX_QS_DEPTH=21
----------------------------------------------------------------------
✓ Using fast C++ chess backend (chess_cpp)
✓ Using Cython-accelerated NN operations
Loading NNUE model...
time_limit=5
#2 FAIL: 8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b
#23 FAIL: r3nrk1/2p2p1p/p1p1b1p1/2NpPq2/3R4/P1N1Q3/1PP2PPP/4...
Progress: 300 tests, 231 passed (77.0%)
time-avg=0.64, time-max=3.08
```

Final summary table sorted by success rate:

```
==========================================================================================
TUNING SUMMARY: MAX_QS_DEPTH
==========================================================================================

Value                            Success%     Passed    Total    TimeAvg    TimeMax
------------------------------------------------------------------------------------------
22                                  77.33        232      300       0.63       2.94
23                                  77.00        231      300       0.64       3.01
21                                  76.33        229      300       0.64       3.08
------------------------------------------------------------------------------------------

*** BEST VALUE: MAX_QS_DEPTH=22
    Success Rate: 77.33% (232/300)
    Time: avg=0.63s, max=2.94s

    Improvement over worst (21): +1.00%
```

### Note

Parameters are passed as environment variables. Ensure your `config.py` reads parameters from `os.environ` for the tuning to take effect.

## Parameter Tuning via ELO (`stockfish_batch.py`)

Tunes engine parameters by playing matches against Stockfish at a specified ELO level. Provides accurate ELO measurement but takes longer than engine tests.

### Usage

```bash
python -m test.stockfish_batch PARAM_NAME value1 value2 value3 ... [options]

# Examples
python -m test.stockfish_batch QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
python -m test.stockfish_batch MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]"
python -m test.stockfish_batch FUTILITY_MAX_DEPTH 2 3 4 --games 50 --elo 2400
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `PARAM_NAME` | Environment variable name to tune | Required |
| `values` | Space-separated values to test | Required |
| `--games`, `-g` | Number of games per value | 30 |
| `--elo`, `-e` | Stockfish ELO level | 2300 |
| `--script`, `-s` | Path to stockfish.sh | Auto-detect |

### Output

Final summary table sorted by ELO difference:

```
==========================================================================================
TUNING SUMMARY: QS_SOFT_STOP_DIVISOR
==========================================================================================

Value                            Est.Elo    EloDiff     ±Error        W-L-D     LOS%
------------------------------------------------------------------------------------------
8.0                                 2395       +94.9      125.5       17-9-4     94.2
9.0                                 2350       +50.0      110.2       14-10-6    82.1
7.0                                 2320       +20.0       98.5       12-10-8    68.5
------------------------------------------------------------------------------------------

*** BEST VALUE: QS_SOFT_STOP_DIVISOR=8.0
    Estimated Elo: 2395 (SF 2300 + +94.9 ±125.5)
    W-L-D: 17-9-4, LOS: 94.2%

    Improvement over worst (7.0): +74.9 Elo
```

## Neural Network Tests (`nn_tests.py`)

Comprehensive test suite for NNUE and DNN model validation.

### Test Types

| ID | Name | Description |
|----|------|-------------|
| 0 | Interactive-FEN | Interactive position evaluation |
| 1 | Incremental-vs-Full | Performance comparison |
| 2 | Accumulator-Correctness | Verify incremental == full evaluation |
| 3 | Eval-Accuracy | Test against training data ground truth |
| 4 | NN-vs-Stockfish | Compare against Stockfish static eval |
| 5 | Feature-Extraction | Verify feature extraction correctness |
| 6 | Symmetry | Test evaluation symmetry (mirrored positions) |
| 7 | Edge-Cases | Checkmate, stalemate, special moves |
| 8 | Reset-Consistency | Test evaluator reset functionality |
| 9 | Deep-Search-Simulation | Many push/pop cycles |
| 10 | Random-Games | Random legal move sequences |
| 11 | CP-Integrity | Centipawn score validation |
| 12 | Engine-Tests | Best-move comparison |
| 13 | All | Run all non-interactive tests |

### Usage Examples

```bash
# Interactive evaluation
python -m test.nn_tests --nn-type NNUE --test 0

# Performance benchmark
python -m test.nn_tests --nn-type DNN --test 1

# Accuracy test with 10000 positions
python -m test.nn_tests --nn-type NNUE --test 3 --positions 10000

# Deep search simulation
python -m test.nn_tests --nn-type NNUE --test 9 --depth 4 --iterations 10

# Random games test
python -m test.nn_tests --nn-type DNN --test 10 --num-games 10 --max-moves 100

# Compare against Stockfish
python -m test.nn_tests --nn-type NNUE --test 4 --positions 100 --stockfish /usr/bin/stockfish
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--nn-type` | Network type: NNUE or DNN | Required |
| `--test` | Test type number (0-13) | Required |
| `--positions` | Positions for accuracy tests | 10000 |
| `--model-path` | Path to model file | `model/{type}.pt` |
| `--depth` | Search depth for simulation | 4 |
| `--iterations` | Iterations for simulation | 10 |
| `--num-games` | Games for random test | 10 |
| `--max-moves` | Max moves per game | 100 |
| `--tolerance` | Float comparison tolerance | 1e-4 |
| `--stockfish` | Path to Stockfish binary | `stockfish` |

## Data Integrity Tests (`data_test.py`)

Verifies correctness of training data shards by comparing stored features against recomputed features from embedded FEN strings.

### Features Tested

- **DNN**: 768 sparse features (12 planes × 64 squares)
- **NNUE**: 40960 HalfKP features (white + black perspectives)
- **Side-to-move**: Correct perspective orientation
- **Scores**: Comparison against Stockfish evaluation

### Usage

```bash
# Auto-detect shards in data/ directory
python -m test.data_test

# Verify specific DNN shard
python -m test.data_test --dnn-shard data/dnn/train_0001.bin.zst

# Verify specific NNUE shard  
python -m test.data_test --nnue-shard data/nnue/train_0001.bin.zst

# Analyze shard statistics
python -m test.data_test --analyze data/nnue/train_0001.bin.zst --nn-type NNUE

# Verify more diagnostic records
python -m test.data_test --max-records 100
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dnn-shard` | Path to DNN shard | Auto-detect |
| `--nnue-shard` | Path to NNUE shard | Auto-detect |
| `--analyze` | Shard to analyze | None |
| `--nn-type` | NN type for analysis | Inferred |
| `--data-dir` | Base data directory | `data` |
| `--max-records` | Max diagnostic records | 10 |

## ELO Testing Scripts

### `stockfish.sh` - Stockfish ELO Measurement

Plays matches against Stockfish at a specified ELO level to measure engine strength.

```bash
./test/stockfish.sh <stockfish-elo> <num-games> [--debug]

# Examples
./test/stockfish.sh 1500 10           # 10 games vs Stockfish 1500
./test/stockfish.sh 2000 50           # 50 games vs Stockfish 2000
./test/stockfish.sh 2300 30 --debug   # 30 games with debug output
```

**Parameters:**
- `stockfish-elo`: Target ELO for Stockfish (UCI_LimitStrength)
- `num-games`: Number of games to play
- `--debug`: Optional flag to enable debug output

**Time Control:** 40 moves in 120 seconds + 1 second increment

### `oldfish.sh` - Regression Testing

Plays matches against a previous version of the engine for regression testing.

```bash
./test/oldfish.sh <new-engine> <old-engine> <time-control> <num-games>

# Examples
./test/oldfish.sh neuro512 old1024 40/120+1 6
./test/oldfish.sh current previous 60+0.5 20
```

**Parameters:**
- `new-engine`: Name for current engine version
- `old-engine`: Name for previous engine version  
- `time-control`: Time control string (e.g., `40/120+1`)
- `num-games`: Number of games to play

**Requirements:**
- Current engine at `../uci.sh`
- Previous engine at `../../oldfish/uci.sh`
- cutechess-cli at `../../cutechess/build/cutechess-cli`

## Test Output Locations

| Test | Output |
|------|--------|
| `stockfish.sh` | `/tmp/fileXXXXXX.pgn` |
| `oldfish.sh` | `/tmp/fileXXXXXX.pgn` |
| Engine tests | stdout |
| NN tests | stdout |
| Data tests | stdout |

## Typical Test Workflow

```bash
# 1. Verify training data
python -m test.data_test

# 2. Train model
python nn_train.py --nn-type NNUE --epochs 10

# 3. Run NN tests
python -m test.nn_tests --nn-type NNUE --test 13

# 4. Run engine tests
python -m test.engine_test

# 5. Tune parameters (quick feedback)
python -m test.engine_test_batch MAX_QS_DEPTH 20 21 22 23 24

# 6. Tune parameters (accurate ELO)
python -m test.stockfish_batch MAX_QS_DEPTH 21 22 23 --games 30 --elo 2300

# 7. Measure final ELO
./test/stockfish.sh 2300 30
```

## Troubleshooting

### "Model not found" error

```bash
# Ensure model exists
ls -la model/nnue.pt model/dnn.pt

# Specify path explicitly
python -m test.nn_tests --nn-type NNUE --test 2 --model-path /path/to/model.pt
```

### "No diagnostic records found"

Diagnostic records are written every 1000 positions during data preparation. Regenerate shards:

```bash
python prepare_data.py --nn-type DNN --input games.pgn --output data/dnn
```

### cutechess-cli not found

Update the path in `stockfish.sh`:

```bash
CUTECHESS_PATH=/your/path/to/cutechess
```

### Stockfish not responding

```bash
# Test Stockfish manually
stockfish
uci
quit

# Or specify full path
python -m test.nn_tests --nn-type NNUE --test 4 --stockfish /usr/games/stockfish
```

### Parameter tuning not taking effect

Ensure your `config.py` reads parameters from environment variables:

```python
import os

MAX_QS_DEPTH = int(os.environ.get('MAX_QS_DEPTH', 22))
```

## Files Overview

```
test/
├── engine_test.py        # Engine move quality tests (WAC, Eigenmann)
├── engine_test_batch.py  # Parameter tuning via engine tests
├── nn_tests.py           # Neural network validation suite
├── data_test.py          # Training data integrity verification
├── stockfish.sh          # ELO measurement vs Stockfish
├── stockfish_batch.py    # Parameter tuning via Stockfish matches
├── oldfish.sh            # Regression testing vs previous versions
└── README.md             # This file
```

## Performance Benchmarks

Expected test durations on typical hardware:

| Test | Duration |
|------|----------|
| WAC Suite (300 positions) | ~25 minutes |
| Engine Test Batch (5 values) | ~2 hours |
| NN All Tests | ~5 minutes |
| Data Verification | ~1 minute |
| Stockfish Match (10 games) | ~30 minutes |
| Stockfish Batch (5 values × 30 games) | ~12 hours |

## License

This test suite is part of the NeuroFish chess engine project and follows the same license terms.