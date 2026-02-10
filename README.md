# NeuroFish

NeuroFish is a hybrid chess engine that combines classical alpha–beta search techniques with selective neural network evaluation to produce strong, explainable move choices within fixed time constraints. Unlike pure neural engines or purely handcrafted evaluators, NeuroFish blends the best of both worlds: a fast, well-optimized negamax search core enhanced by a NN-based positional evaluator that is invoked only when it is most informative.

The engine plays at approximately **2500 ELO** rating, making it possibly the highest-rated chess engine written in Python.

**Play against NeuroFish online:** [https://lichess.org/?user=neurofish#friend](https://lichess.org/?user=neurofish#friend)

## Project Description

NeuroFish was developed to explore the practical limits of building a competitive chess engine in Python while leveraging modern neural network evaluation techniques. The engine uses approximately **1 billion positions** downloaded from [Lichess.org](https://lichess.org) for neural network training.

### Key Features

- **Hybrid Evaluation**: Combines fast classical evaluation with selective neural network assessment
- **Dual NN Architectures**: Supports both DNN (Deep Neural Network) and NNUE (Efficiently Updatable Neural Network) evaluation
- **UCI Protocol Support**: Compatible with any UCI-compliant chess GUI
- **Opening Book Support**: Polyglot opening book integration with weighted random selection
- **Advanced Search Techniques**: Negamax with alpha-beta pruning, null-move pruning, late move reductions (LMR), futility pruning, razoring, SEE pruning, aspiration windows, and singular extensions
- **Transposition Tables**: Zobrist hashing with configurable table sizes
- **Lazy SMP Parallel Search**: Multi-threaded search using the Lazy SMP algorithm with shared transposition table
- **Pondering**: Thinks on the opponent's time for stronger play under time controls
- **Incremental NN Updates**: Efficient accumulator-based neural network updates
- **Cython Acceleration**: Performance-critical operations optimized with Cython
- **C++ Backend Option**: Optional high-performance C++ move generation via Disservin's chess-library

### Limited Use of C++ and Cython

While NeuroFish is primarily written in Python, selective use of C++ and Cython provides critical performance improvements:

- **Cython (`nn_ops_fast.pyx`)**: Neural network operations including clipped ReLU, matrix multiplications, and accumulator updates are implemented in Cython for 3-5x speedup over pure Python
- **C++ Backend (`cpp_board/`)**: Optional integration with [Disservin's chess-library](https://github.com/Disservin/chess-library) for fast move generation, reducing the overhead of python-chess
- **C++ Batch Loaders (`nn_train/cpp_nn_train/`)**: Multi-threaded data loading for neural network training

A pure Python fallback (`nn_ops_fallback.py`) is provided when Cython extensions are not compiled.

### Neural Network Architectures

NeuroFish supports two neural network architectures for position evaluation:

#### NNUE (Efficiently Updatable Neural Network)

```
Input:  40,960 features (HalfKP encoding: king square × piece square × piece type/color)
    ↓
Feature Transformer: 40,960 → 256 (separate white/black accumulators)
    ↓
Concatenation: 512 (white + black, side-to-move aware)
    ↓
Hidden Layer 1: 512 → 32 (Clipped ReLU)
    ↓
Hidden Layer 2: 32 → 32 (Clipped ReLU)
    ↓
Output: 32 → 1 (centipawn score)
```

The NNUE architecture uses HalfKP feature encoding where each feature represents the relationship between a king position and a piece position. This enables efficient incremental updates—when a move is made, only the affected features need to be updated rather than recomputing the entire evaluation.

#### DNN (Deep Neural Network)

```
Input:  768 features (12 piece planes × 64 squares)
    ↓
Hidden Layers: Configurable architecture
    ↓
Output: 1 (centipawn score)
```

The DNN uses standard piece-square encoding with 12 planes (6 piece types × 2 colors) across 64 squares.

## Directory Structure

```
neurofish/
├── book/                    # Opening book files (Polyglot format)
├── book_move.py             # Opening book lookup and move selection
├── build/                   # Build artifacts
├── build.sh                 # Cython extension build script
├── cached_board.py          # Board wrapper with caching and C++ backend support
├── chess_engine.py          # Main search engine (negamax, alpha-beta)
├── config.py                # Engine configuration and parameters
├── cpp_board/               # C++ chess library integration
│   └── README.md            # C++ backend documentation
├── data/                    # Training data directory
├── docs/                    # Documentation
│   ├── parameter_tuning_guide.md
│   └── time_depth_management_guide.md
├── environment.yml          # Conda environment specification
├── lazy_smp.py              # Lazy SMP parallel search implementation
├── libs/                    # Compiled shared libraries (.so files)
├── model/                   # Trained neural network models
├── nn_evaluator.py          # Neural network evaluation interface
├── nn_inference.py          # NN inference and feature extraction
├── nn_ops_fallback.py       # Pure Python fallback for Cython operations
├── nn_ops_fast.pyx          # Cython-optimized NN operations
├── nn_train/                # Neural network training module
│   └── README.md            # Training documentation
├── pgn/                     # PGN game files
├── requirements.txt         # pip requirements file
├── setup.py                 # Cython build configuration
├── test/                    # Test suite
│   └── README.md            # Testing documentation
├── uci.py                   # UCI protocol implementation
├── uci.sh                   # UCI engine launcher script
└── utils/                   # Utility scripts
    └── README.md            # Utilities documentation
```

### Subdirectory Documentation

- [cpp_board/README.md](cpp_board/README.md) — C++ backend integration guide
- [nn_train/README.md](nn_train/README.md) — Neural network training pipeline
- [test/README.md](test/README.md) — Test suite documentation
- [utils/README.md](utils/README.md) — Utility scripts documentation
- [docs/parameter_tuning_guide.md](docs/parameter_tuning_guide.md) — Configuration parameter tuning
- [docs/time_depth_management_guide.md](docs/time_depth_management_guide.md) — Time and depth management

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- C++ compiler (optional, for Cython extensions)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/eapenkuruvilla/neurofish.git
   cd neurofish
   ```

2. **Create the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate neurofish
   ```
   - If anaconda is not available on your machine, you may install the python packages using the command 'pip install -r requirements.txt'
   
   
3. **Build Cython extensions (recommended for performance):**

   ```bash
   ./build.sh
   ```

4. **Build C++ backend (optional, for additional performance):**

   ```bash
   cd cpp_board
   ./build.sh
   cd ..
   ```
   
5. **Build or download the neural network models:**
   
   - Follow the instructions given in [nn_train/README](nn_train/README.md) to build the NN models if you are willing spend couple of weeks.
   - Alternatively you can download a prebuilt NNUE model from [OneDrive](https://1drv.ms/u/c/4c1b0c9c6e5795c7/IQA7Y1iU7XDJR5qi1FaLGzTRAdRsNe5VygcI0qM0qQb_4Yo?e=wuF7zw).
     - Copy the downloaded nnue.pt to model/nnue directory
   - A third option is to set IS_NN_ENABLED=False in the config.py file, but be willing to sacrifice around 400 ELOs of engine's strength.


6. **Download a book file:**

   - For example, download [komodo.bin](https://github.com/gmcheems-org/free-opening-books?tab=readme-ov-file)
   - Copy the downloaded file to the book directory

## Executing the Program

### Playing Online

You can play against NeuroFish at [https://lichess.org/@/neurofish](https://lichess.org/@/neurofish)

### Running Locally with PyChess

The engine speaks the UCI (Universal Chess Interface) protocol and can be used with chess GUIs such as PyChess, Arena, Cute Chess, and other UCI-compatible interfaces.

1. **Install PyChess:**

   ```bash
   sudo apt install pychess
   ```

2. **Start PyChess:**

   ```bash
   pychess
   ```

3. **Configure PyChess to use NeuroFish:**
   - Go to **Edit → Engines → New**
   - Browse to the cloned directory and select `uci.sh`
   - Click OK to add the engine

4. **Play against the engine:**
   - Start a new game and select NeuroFish as your opponent

### Running from Command Line

```bash
# Activate the environment
conda activate neurofish

# Run the UCI interface
./uci.sh
```

## Benchmarks

### DNN vs NNUE Performance Comparison

| Test                                                   | Measurement Unit                  | DNN                    | NNUE                    |
|--------------------------------------------------------|-----------------------------------|------------------------|-------------------------|
| Time per positional evaluation                         | Milliseconds                      | 0.162                  | 0.037                   |
| Training validation error                              | MSE - tanh(CP/410)                | 0.041                  | 0.046                   |
| Against Stockfish depth 20+                            | MAE - Difference capped at 100 CP | 54                     | 55                      |
| Against Stockfish depth 20+                            | MSE - tanh(CP/410)                | 0.037                  | 0.040                   |
| Against Stockfish NNUE positional evaluation (depth 0) | MAE - Difference capped at 100 CP | 8 (DNN better than SF) | 5 (NNUE better than SF) |
| Against Stockfish NNUE positional evaluation (depth 0) | MSE - tanh(CP/410)                | 0.021                  | 0.0178                  |
| Against Classical piece-square evaluation (ELO 1883)   | MAE - ELO                         | +394                   | +607                    |

**Analysis:** NNUE outperforms DNN in playing strength primarily due to its 4x faster evaluation time, allowing deeper search within the same time constraints. The NNUE architecture's incremental update capability means that after each move, only affected features need recalculation, whereas DNN requires full forward propagation.  Interestingly, both NN architectures slightly outperform Stockfish's NNUE in raw positional accuracy (lower MAE against SF NNUE eval). This may be attributed to quantization errors in Stockfish's heavily optimized INT8 implementation.

### ELO Gains by Various Features

| Type                           | ELO Change |
|--------------------------------|------------|
| C++ Board                      | +95        |
| Cython acceleration            | +107       |
| Quantization (INT8)            | -89        |
| Quantization (INT16)           | +95        |
| Pondering                      | +90        |
| BLAS multi-core, single thread | +0         |

**Analysis:** The two largest gains come from reducing Python interpreter overhead: Cython-optimized NN operations (+107) and C++ move generation (+95) together contribute over 200 ELO, underscoring that raw execution speed is the dominant bottleneck in a Python chess engine. Faster evaluation and move generation translate directly into deeper search within the same time budget. INT8 quantization (+46) further accelerates NN inference by leveraging narrower integer arithmetic, while INT16 quantization (-47) regresses because its wider data types reduce SIMD throughput without a compensating improvement in evaluation accuracy. BLAS multi-core (+35) provides a modest gain by parallelizing the matrix operations within NN inference—this benefit is orthogonal to search-level parallelism and stacks with single-threaded optimizations.

### Lazy SMP Performance

| Threads         | ELO      |
|-----------------|----------|
| 1               | Baseline |
| 2 (with GIL)    | -84      |
| 2 (without GIL) | +109     |
| 3 (without GIL) | +218     |
| 4 (without GIL) | +109     |


**Analysis:** Two threads yield a modest +14 ELO over single-threaded search, but scaling to four threads regresses by −27 ELO below the single-threaded baseline. This is a direct consequence of Python's Global Interpreter Lock (GIL): while NumPy and the NN inference layers release the GIL during heavy computation, the search logic itself—move ordering, alpha-beta recursion, transposition table lookups—runs under the GIL and becomes a contention bottleneck as thread count increases. At two threads the diversity benefit of Lazy SMP (workers exploring slightly different search trees and populating the shared transposition table) narrowly outweighs the GIL overhead, but at four threads the contention cost dominates. The optimal configuration for NeuroFish is therefore two threads; meaningful scaling beyond this would require moving the core search loop to C/C++ or to a GIL-free runtime.

## Future Work

### The 2500 ELO Ceiling

The ~2500 ELO rating likely represents the practical ceiling for a Python-based chess engine. The primary limitations are:

1. **Interpreted Language Overhead**: Python's interpreted nature adds significant overhead compared to compiled languages. Each operation involves type checking, reference counting, and interpreter dispatch.

2. **Global Interpreter Lock (GIL)**: Python's GIL prevents true parallel execution of Python bytecode, limiting the effectiveness of multithreading and making it impossible to share data structures efficiently between threads. The Lazy SMP results above illustrate this directly—scaling beyond two threads produces negative returns.

3. **Memory Management**: Python's object model and garbage collection add latency that compounds during deep searches with millions of nodes.

4. **Function Call Overhead**: The high cost of Python function calls limits the effectiveness of highly recursive algorithms like negamax search.

### Potential Next Steps

Developing a C++ engine leveraging the architectural insights and techniques refined in this Python implementation is a natural next project. The experience gained from NeuroFish—particularly around neural network integration, incremental updates, and search/evaluation tradeoffs—would transfer directly to a faster implementation capable of significantly higher playing strength.

## Authors

**Eapen Kuruvilla** — [LinkedIn](https://www.linkedin.com/in/eapenkuruvilla/)

## Credits

### Libraries and Tools

- **[Disservin/chess-library](https://github.com/Disservin/chess-library)** — Fast, header-only C++ chess library used for the optional high-performance backend. See their repository for licensing terms.
- **[python-chess](https://github.com/niklasf/python-chess)** — Pure Python chess library providing the core board representation
- **[pybind11](https://github.com/pybind/pybind11)** — C++/Python interoperability for the C++ backend
- **[Lichess.org](https://lichess.org)** — Source of the 1 billion training positions
- **[Vultr.com](https://lichess.org)** — GPU and cloud resources for the AI training
- **[JetBrains PyCharm](https://lichess.org)** — IDE for the code development

### AI Assistance

[Anthropic Claude](https://claude.ai) was used extensively throughout development for writing code, generating test cases, debugging, and troubleshooting. The AI assistance multiplied development efficiency by at least **20 times**, enabling rapid iteration on complex algorithms and comprehensive test coverage that would have taken significantly longer to develop manually.

## License

This project is free to use, modify, and redistribute for any purpose, including commercial use. It is provided as-is, without any warranty of any kind, and the authors assume no responsibility or liability for any damages arising from its use.