# Test Suite for NeuroFish Chess Engine

Comprehensive test suite for validating the chess engine, neural network evaluation, and training data integrity.

## Test Categories

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `engine_test.py` | Chess engine move quality | WAC, Eigenmann test suites |
| `nn_tests.py` | Neural network correctness | Accumulator, symmetry, accuracy |
| `data_test.py` | Training data integrity | Feature extraction, Stockfish comparison |
| `stockfish.sh` | ELO measurement | Matches against Stockfish |
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
git clone https://github.com/cutechess/cutechess.git
cd cutechess && mkdir build && cd build
cmake .. && make
```

## Quick Start

### Run Engine Tests (WAC Suite)

```bash
# Single-threaded
python test/engine_test.py

# Multi-processing
python test/engine_test.py --mp
```

### Run Neural Network Tests

```bash
# Interactive FEN evaluation
python test/nn_tests.py --nn-type NNUE --test 0

# Run all tests
python test/nn_tests.py --nn-type NNUE --test 13

# Accumulator correctness
python test/nn_tests.py --nn-type DNN --test 2
```

### Run Data Integrity Tests

```bash
# Auto-detect and verify shards
python test/data_test.py

# Verify specific shard
python test/data_test.py --dnn-shard data/dnn/train_0001.bin.zst
python test/data_test.py --nnue-shard data/nnue/train_0001.bin.zst

# Analyze shard statistics
python test/data_test.py --analyze data/nnue/train_0001.bin.zst --nn-type NNUE
```

### Measure ELO Rating

```bash
# Play against Stockfish at specific ELO
./test/stockfish.sh NeuroFish-v1 1500 10

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
python test/engine_test.py

# Enable multiprocessing
python test/engine_test.py --mp
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
python test/nn_tests.py --nn-type NNUE --test 0

# Performance benchmark
python test/nn_tests.py --nn-type DNN --test 1

# Accuracy test with 10000 positions
python test/nn_tests.py --nn-type NNUE --test 3 --positions 10000

# Deep search simulation
python test/nn_tests.py --nn-type NNUE --test 9 --depth 4 --iterations 10

# Random games test
python test/nn_tests.py --nn-type DNN --test 10 --num-games 10 --max-moves 100

# Compare against Stockfish
python test/nn_tests.py --nn-type NNUE --test 4 --positions 100 --stockfish /usr/bin/stockfish
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
python test/data_test.py

# Verify specific DNN shard
python test/data_test.py --dnn-shard data/dnn/train_0001.bin.zst

# Verify specific NNUE shard  
python test/data_test.py --nnue-shard data/nnue/train_0001.bin.zst

# Analyze shard statistics
python test/data_test.py --analyze data/nnue/train_0001.bin.zst --nn-type NNUE

# Verify more diagnostic records
python test/data_test.py --max-records 100
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
./test/stockfish.sh <engine-tag> <stockfish-elo> <num-games>

# Examples
./test/stockfish.sh DNN512 1500 10    # 10 games vs Stockfish 1500
./test/stockfish.sh NNUE-v2 2000 50   # 50 games vs Stockfish 2000
```

**Parameters:**
- `engine-tag`: Name for your engine in PGN output
- `stockfish-elo`: Target ELO for Stockfish (UCI_LimitStrength)
- `num-games`: Number of games to play

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
- cutechess-cli at `~/Temp/cutechess/build/cutechess-cli`

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
python test/data_test.py

# 2. Train model
python nn_train.py --nn-type NNUE --epochs 10

# 3. Run NN tests
python test/nn_tests.py --nn-type NNUE --test 13

# 4. Run engine tests
python test/engine_test.py

# 5. Measure ELO
./test/stockfish.sh NNUE-trained 1800 20
```

## Troubleshooting

### "Model not found" error

```bash
# Ensure model exists
ls -la model/nnue.pt model/dnn.pt

# Specify path explicitly
python test/nn_tests.py --nn-type NNUE --test 2 --model-path /path/to/model.pt
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
python test/nn_tests.py --nn-type NNUE --test 4 --stockfish /usr/games/stockfish
```

## Files Overview

```
test/
├── engine_test.py      # Engine move quality tests (WAC, Eigenmann)
├── nn_tests.py         # Neural network validation suite
├── data_test.py        # Training data integrity verification
├── stockfish.sh        # ELO measurement vs Stockfish
├── oldfish.sh          # Regression testing vs previous versions
└── README.md           # This file
```

## Performance Benchmarks

Expected test durations on typical hardware:

| Test | Duration |
|------|----------|
| WAC Suite (300 positions) | ~25 minutes |
| NN All Tests | ~5 minutes |
| Data Verification | ~1 minute |
| Stockfish Match (10 games) | ~30 minutes |

## License

This test suite is part of the NeuroFish chess engine project and follows the same license terms.
