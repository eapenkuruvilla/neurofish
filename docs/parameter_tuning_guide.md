# NeuroFish Parameter Tuning Guide

## Overview

This guide recommends which testing method to use for tuning each parameter in `config.py`:

| Test Method | Tool | Use Case |
|-------------|------|----------|
| **engine_test.py** | WAC/Eigenmann suites | **Tactical parameters** - Pure tactical positions with known best moves. Fast iteration (3-5 sec/position). Measures move-finding accuracy. |
| **stockfish.sh** | cutechess-cli vs Stockfish | **Positional/Time parameters** - Full games with time controls. Measures actual playing strength (ELO). Slower but more realistic. |

**Key Principle**: Parameters affecting *what* the engine searches → `engine_test.py`. Parameters affecting *when/how long* to search → `stockfish.sh`.

---

## Parameter Recommendations

### 🔴 SCORING & EVALUATION

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MAX_SCORE` | 10,000 | **Neither** | Infrastructure constant, don't tune |
| `TANH_SCALE` | 410 | **stockfish.sh** | Affects NN output scaling, needs full games to evaluate positional understanding |

---

### 🟠 NEURAL NETWORK EVALUATION

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `IS_NN_ENABLED` | True | **stockfish.sh** | Fundamental mode change, needs ELO comparison |
| `NN_TYPE` | "NNUE" | **stockfish.sh** | Architecture choice, requires full game evaluation |
| `L1_QUANTIZATION` | "INT8" | **stockfish.sh** | Speed vs accuracy tradeoff, needs ELO measurement. INT8 is recommended default. |
| `FULL_NN_EVAL_FREQ` | 3000 | **stockfish.sh** | Performance optimization, measure ELO impact. Increase to 50,000 after initial testing. |

**NN in Quiescence Search** (use **stockfish.sh** for all):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `QS_DEPTH_MIN_NN_EVAL` | 6 | Deeper NN use affects positional accuracy |
| `QS_DEPTH_MAX_NN_EVAL` | 999 | Controls when NN is used in QS |
| `QS_DELTA_MAX_NN_EVAL` | 100 | Threshold for triggering NN eval |
| `STAND_PAT_MAX_NN_EVAL` | 200 | Affects quiet position evaluation |

---

### 🟡 QUIESCENCE SEARCH PARAMETERS

These control the tactical search - **use engine_test.py** as primary, validate with stockfish.sh:

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MAX_QS_DEPTH` | 22 | **engine_test.py** | Directly affects tactical resolution |
| `MAX_QS_MOVES` | [12, 6, 4, 2] | **engine_test.py** | Move limits at QS depths (list format) |
| `MAX_QS_MOVES_DIVISOR` | [4, 2.0, 1.33] | **engine_test.py** | Depth threshold calculation (list format) |
| `CHECK_QS_MAX_DEPTH` | 5 | **engine_test.py** | How deep to search checks in QS |
| `DELTA_PRUNING_QS_MIN_DEPTH` | 5 | **engine_test.py** | When delta pruning kicks in |
| `DELTA_PRUNING_QS_MARGIN` | 75 | **engine_test.py** | Delta pruning aggressiveness |

**Note on MAX_QS_MOVES and MAX_QS_MOVES_DIVISOR interaction**: 
- `MAX_QS_MOVES_DIVISOR = [D1, D2, D3]` divides `MAX_QS_DEPTH` into 4 zones:
  - Zone 1: depth > MAX_QS_DEPTH/D1 → uses MAX_QS_MOVES[0] (12 moves)
  - Zone 2: MAX_QS_DEPTH/D1 >= depth > MAX_QS_DEPTH/D2 → uses MAX_QS_MOVES[1] (6 moves)
  - Zone 3: MAX_QS_DEPTH/D2 >= depth > MAX_QS_DEPTH/D3 → uses MAX_QS_MOVES[2] (4 moves)
  - Zone 4: depth <= MAX_QS_DEPTH/D3 → uses MAX_QS_MOVES[3] (2 moves)
- Example with defaults: MAX_QS_DEPTH=22, divisors=[4, 2.0, 1.33]
  - Zone 1 (depth > 5.5): 12 moves
  - Zone 2 (5.5 >= depth > 11): 6 moves
  - Zone 3 (11 >= depth > 16.5): 4 moves
  - Zone 4 (depth <= 16.5): 2 moves

**Note**: After finding optimal values with engine_test.py, always validate with a few stockfish.sh games to ensure no regression in playing strength.

---

### 🟢 QS TIME CONTROL (use **stockfish.sh**)

These parameters interact with time management and only matter in timed games:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `QS_SOFT_STOP_DIVISOR` | 9.0 | When to soft-stop in QS |
| `QS_TIME_CRITICAL_FACTOR` | 0.86 | Time pressure threshold |
| `MAX_QS_MOVES_TIME_CRITICAL` | 5 | Move limit under time pressure |
| `QS_TIME_CHECK_INTERVAL` | 40 | How often to check time in QS |
| `QS_TIME_BUDGET_FRACTION` | 0.35 | QS time allocation |
| `QS_TT_SUPPORTED` | False | QS transposition table |

---

### 🔵 MAIN SEARCH DEPTH CONTROL

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MIN_NEGAMAX_DEPTH` | 4 | **stockfish.sh** | Minimum depth before stopping - time-critical |
| `MIN_PREFERRED_DEPTH` | 5 | **stockfish.sh** | Target minimum depth - affects time usage |
| `TACTICAL_MIN_DEPTH` | 5 | **stockfish.sh** | Min depth for tactical positions |
| `UNSTABLE_MIN_DEPTH` | 5 | **stockfish.sh** | Min depth when scores are unstable |
| `MAX_NEGAMAX_DEPTH` | 20 | **Neither** | Upper bound, rarely reached |

---

### 🟣 TIME MANAGEMENT (use **stockfish.sh** exclusively)

These parameters ONLY matter in timed games and cannot be evaluated with engine_test.py:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `EMERGENCY_TIME_RESERVE` | 0.50 | Time buffer to avoid flag |
| `ESTIMATED_BRANCHING_FACTOR` | 4.0 | For predicting search time |
| `TIME_SAFETY_MARGIN_RATIO` | 0.45 | When to start new depth |
| `MAX_SEARCH_TIME` | 30 | Maximum time per move |

**⚠️ Important**: Time management bugs cause flag losses. Always test with multiple game counts (20+ games minimum).

---

### 🟤 ASPIRATION WINDOWS

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `ASPIRATION_WINDOW` | 75 | **Both** | Start with engine_test.py for accuracy, validate with stockfish.sh |
| `MAX_AW_RETRIES` | 1 | **Both** | How many retries before full window |
| `MAX_AW_RETRIES_TACTICAL` | 3 | **engine_test.py** | Extra retries for tactical positions |

---

### ⚫ LATE MOVE REDUCTIONS (LMR)

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `LMR_MOVE_THRESHOLD` | 2 | **engine_test.py** | After which move to apply LMR |
| `LMR_MIN_DEPTH` | 4 | **engine_test.py** | Minimum depth for LMR |

**Note**: LMR affects tactical accuracy. Test with engine_test.py first, but if WAC scores drop, the ELO might still be fine (or vice versa). Validate with stockfish.sh.

---

### ⚪ NULL MOVE PRUNING

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `NULL_MOVE_REDUCTION` | 2 | **engine_test.py** | R value for null move |
| `NULL_MOVE_MIN_DEPTH` | 3 | **engine_test.py** | When to apply null move |

**Note**: Null move can miss tactical sequences. engine_test.py will catch issues where the engine misses forced wins/losses.

---

### 🔘 SINGULAR EXTENSIONS

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `SINGULAR_MARGIN` | 130 | **engine_test.py** | Score margin for singular move |
| `SINGULAR_EXTENSION` | 1 | **engine_test.py** | Extension amount |

---

### ⬛ PRUNING TECHNIQUES

**SEE Pruning** (use **engine_test.py**):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `SEE_PRUNING_ENABLED` | False | Whether to use SEE pruning |
| `SEE_PRUNING_MAX_DEPTH` | 6 | Depth limit for SEE pruning |

**Futility Pruning** (use **engine_test.py** primarily):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `FUTILITY_PRUNING_ENABLED` | True | Whether to use futility pruning |
| `FUTILITY_MARGIN` | [0, 50, 100, 350] | Margins per depth (list format) |
| `FUTILITY_MAX_DEPTH` | 3 | Maximum depth for futility |

**Razoring** (use **engine_test.py** primarily):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `RAZORING_ENABLED` | False | Whether to use razoring |
| `RAZORING_MARGIN` | [0, 125, 250] | Margins per depth (list format) |
| `RAZORING_MAX_DEPTH` | 2 | Maximum depth for razoring |

---

### 📊 INFRASTRUCTURE

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MAX_TABLE_SIZE` | 200,000 | **stockfish.sh** | Affects memory/speed tradeoff over many games |
| `MAX_THREADS` | 2 | **stockfish.sh** | Multiprocessing (1 or less disables MP) |
| `IS_BLAS_ENABLED` | False | **stockfish.sh** | BLAS acceleration |
| `IS_PONDERING_ENABLED` | True | **stockfish.sh** | Thinking on opponent's time |

---

### 🛠️ DIAGNOSTIC & DEBUG

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `IS_DIAGNOSTIC` | False | **Neither** | Master switch for diagnostic output |
| `DEBUG_MODE` | False | **Neither** | Runtime toggle via UCI "debug on/off" |

---

## Tuning Workflow

### Step 1: Tactical Parameters (engine_test.py)

```bash
# Baseline
python engine_test.py

# Test a parameter change
QS_DEPTH_MIN_NN_EVAL=8 python engine_test.py

# Test list parameters (use quotes and brackets)
MAX_QS_MOVES="[14, 7, 5, 3]" python engine_test.py

# Compare success rates
```

**Target**: Maximize WAC/Eigenmann success rate without excessive time increase.

### Step 2: Time/Positional Parameters (stockfish.sh)

```bash
# Run baseline games
./stockfish.sh 1500 20

# Test parameter change
MIN_NEGAMAX_DEPTH=5 ./stockfish.sh 1500 20

# Compare results
```

**Target**: Maximize win rate against same Stockfish ELO level.

### Step 3: Validation

After tuning tactical parameters with engine_test.py:
```bash
# Validate that tactical improvements translate to playing strength
./stockfish.sh 1500 20
```

---

## Quick Reference Table

| Category | Primary Test | Secondary Test |
|----------|--------------|----------------|
| Pruning (SEE, Futility, Razoring) | engine_test.py | stockfish.sh |
| QS Move Limits | engine_test.py | stockfish.sh |
| QS Time Control | stockfish.sh | - |
| NN Evaluation Thresholds | stockfish.sh | engine_test.py |
| Time Management | stockfish.sh | - |
| LMR/Null Move | engine_test.py | stockfish.sh |
| Aspiration Windows | engine_test.py | stockfish.sh |
| Search Depth Minimums | stockfish.sh | - |

---

## Parameter Interaction Groups

Some parameters should be tuned together as they interact:

### Group 1: QS Move Limits
- `MAX_QS_MOVES` (list: [Q1, Q2, Q3, Q4])
- `MAX_QS_MOVES_DIVISOR` (list: [D1, D2, D3])
- `MAX_QS_DEPTH`

**Note**: These are list parameters. To override via environment:
```bash
MAX_QS_MOVES="[14, 7, 5, 3]" python engine_test.py
MAX_QS_MOVES_DIVISOR="[5, 2.5, 1.5]" python engine_test.py
```

### Group 2: QS Time Control
- `QS_TIME_BUDGET_FRACTION`
- `QS_SOFT_STOP_DIVISOR`
- `QS_TIME_CRITICAL_FACTOR`
- `MAX_QS_MOVES_TIME_CRITICAL`

### Group 3: NN in QS
- `QS_DEPTH_MIN_NN_EVAL`
- `QS_DEPTH_MAX_NN_EVAL`
- `QS_DELTA_MAX_NN_EVAL`
- `STAND_PAT_MAX_NN_EVAL`

### Group 4: Main Search Time
- `MIN_NEGAMAX_DEPTH`
- `MIN_PREFERRED_DEPTH`
- `TIME_SAFETY_MARGIN_RATIO`
- `ESTIMATED_BRANCHING_FACTOR`
- `EMERGENCY_TIME_RESERVE`

---

## Environment Variable Examples

```bash
# Integer parameters
MAX_QS_DEPTH=25 python engine_test.py

# Float parameters
QS_SOFT_STOP_DIVISOR=8.5 ./stockfish.sh

# Boolean parameters (accepts: true/false, 1/0, yes/no, on/off)
FUTILITY_PRUNING_ENABLED=false python engine_test.py
IS_NN_ENABLED=0 ./stockfish.sh

# String parameters
NN_TYPE="NNUE" python engine_test.py
L1_QUANTIZATION="INT16" python engine_test.py

# List parameters (use Python list syntax in quotes)
FUTILITY_MARGIN="[0, 75, 150, 400]" python engine_test.py
MAX_QS_MOVES="[15, 8, 5, 3]" python engine_test.py
MAX_QS_MOVES_DIVISOR="[5.0, 2.5, 1.5]" python engine_test.py

# Multiple parameters
QS_DEPTH_MIN_NN_EVAL=8 QS_DELTA_MAX_NN_EVAL=125 ./stockfish.sh
```

---

## Summary

| If the parameter affects... | Use... |
|-----------------------------|--------|
| Move finding accuracy | engine_test.py |
| Time allocation/management | stockfish.sh |
| Search depth decisions | stockfish.sh |
| Pruning aggressiveness | engine_test.py (validate with stockfish.sh) |
| Neural network usage | stockfish.sh |
| Positional evaluation | stockfish.sh |