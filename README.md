# NeuroFish

NeuroFish is a hybrid chess engine that combines classical alpha–beta search techniques with selective neural network evaluation to produce strong, explainable move choices within fixed time constraints. Unlike pure neural engines or purely handcrafted evaluators, NeuroFish blends the best of both worlds: a fast, well-optimized negamax search core enhanced by a NN-based positional evaluator that is invoked only when it is most informative.

The engine plays at approximately **2400 ELO** rating, making it possibly the highest-rated chess engine written in Python.

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

- pyenv and pip
- C++ compiler (for Cython extensions)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/eapenkuruvilla/neurofish.git
   cd neurofish
   ```

2. **Create python virtual environment:**

   ```bash
   sudo apt update; sudo apt install build-essential libssl-dev libffi-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libgdbm-dev libsqlite3-dev libbz2-dev -y
   pyenv install 3.14t
   pyenv local 3.14t
   python3.14t -m venv .venv
   pip install --upgrade pip
   pip install -r requirements.txt
   source .venv/bin/activate
   ``` 
   
3. **Build Cython extensions:**

   ```bash
   ./build.sh
   ```

4. **Build C++ backend:**

   ```bash
   cd cpp_board
   ./build.sh
   cd ..
   ```
   
5. **Build or download the neural network models:**
   
   - Follow the instructions given in [nn_train/README](nn_train/README.md) to build the NN models if you are willing spend a couple of weeks.
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
source .venv/bin/activate

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
| Quantization (INT8)            | +178       |
| Quantization (INT16)           | +107       |
| Pondering                      | +90        |
| BLAS multi-core, single thread | +0         |

**Analysis:** The largest gains come from reducing Python interpreter overhead and optimizing neural network inference. INT8 quantization provides the biggest single improvement (+178 ELO) by leveraging narrower integer arithmetic for faster SIMD operations. Cython-optimized NN operations (+107) and C++ move generation (+95) together contribute over 200 ELO, underscoring that raw execution speed is the dominant bottleneck in a Python chess engine. INT16 quantization (+107) provides similar gains to Cython by reducing memory bandwidth compared to FP32 while maintaining sufficient precision. Pondering (+90) allows the engine to think on the opponent's time, effectively increasing search depth. BLAS multi-core (+0) shows no measurable gain because the NN inference matrices are too small to benefit from parallelization overhead, and search-level parallelism via Lazy SMP is the more effective approach.

### Lazy SMP Performance

| Threads          | Python Version         | ELO      |
|------------------|------------------------|----------|
| 1                | 3.14                   | Baseline |
| 2  | 3.14                   | -84      |
| 2  | 3.14t (Free Threading) | +210     |
| 3  | 3.14t (Free Threading)                       | +181     |
| 4   | 3.14t (Free Threading)                       | +163     |


**Analysis:** The Lazy SMP results demonstrate the critical impact of Python's Global Interpreter Lock (GIL) on parallel search. With standard Python 3.14, two threads actually regress by −84 ELO because the GIL serializes the search logic—move ordering, alpha-beta recursion, and transposition table operations—creating contention that outweighs any benefit from parallel exploration. However, Python 3.14t with free-threading eliminates this bottleneck entirely: two threads yield a dramatic +210 ELO gain as workers can truly execute in parallel, populating the shared transposition table with diverse search results. The gains diminish with additional threads (+181 at 3 threads, +163 at 4 threads), likely due to increased transposition table contention and diminishing returns from search diversity. The optimal configuration for NeuroFish is two threads with Python 3.14t free-threading.

## Future Work

### The 2400 ELO Ceiling

The ~2400 ELO rating likely represents the practical ceiling for a Python-based chess engine. The primary limitations are:

1. **Interpreted Language Overhead**: Python's interpreted nature adds significant overhead compared to compiled languages. Each operation involves type checking, reference counting, and interpreter dispatch.

2. **Memory Management**: Python's object model and garbage collection add latency that compounds during deep searches with millions of nodes.

3. **Function Call Overhead**: The high cost of Python function calls limits the effectiveness of highly recursive algorithms like negamax search.

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
- **[Vultr.com](https://www.vultr.com)** — GPU and cloud resources for the AI training
- **[JetBrains PyCharm](https://www.jetbrains.com/pycharm/)** — IDE for the code development

### AI Assistance

[Anthropic Claude](https://claude.ai) was used extensively throughout development for writing code, generating test cases, debugging, and troubleshooting. The AI assistance multiplied development efficiency by at least **20 times**, enabling rapid iteration on complex algorithms and comprehensive test coverage that would have taken significantly longer to develop manually.

## License

This project is free to use, modify, and redistribute for any purpose, including commercial use. It is provided as-is, without any warranty of any kind, and the authors assume no responsibility or liability for any damages arising from its use.