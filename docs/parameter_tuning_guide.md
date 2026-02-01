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
| `L1_QUANTIZATION` | "NONE" | **stockfish.sh** | Speed vs accuracy tradeoff, needs ELO measurement |
| `FULL_NN_EVAL_FREQ` | 3000 | **stockfish.sh** | Performance optimization, measure ELO impact |

**NN in Quiescence Search** (use **stockfish.sh** for all):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `QS_DEPTH_MIN_NN_EVAL` | 6 | Deeper NN use affects positional accuracy |
| `QS_DEPTH_MAX_NN_EVAL` | 999 | Controls when NN is used in QS |
| `QS_DELTA_MAX_NN_EVAL` | 75 | Threshold for triggering NN eval |
| `STAND_PAT_MAX_NN_EVAL` | 200 | Affects quiet position evaluation |

---

### 🟡 QUIESCENCE SEARCH PARAMETERS

These control the tactical search - **use engine_test.py** as primary, validate with stockfish.sh:

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MAX_QS_DEPTH` | 22 | **engine_test.py** | Directly affects tactical resolution |
| `MAX_QS_MOVES_Q1` | 10 | **engine_test.py** | Move limits at shallow QS |
| `MAX_QS_MOVES_Q2` | 5 | **engine_test.py** | Move limits at medium QS |
| `MAX_QS_MOVES_Q3` | 2 | **engine_test.py** | Move limits at deep QS |
| `MAX_QS_MOVES_Q4` | 1 | **engine_test.py** | Move limits at deepest QS |
| `MAX_QS_MOVES_Q1_DIVISOR` | 4.0 | **engine_test.py** | Depth threshold calculation |
| `MAX_QS_MOVES_Q2_DIVISOR` | 2.0 | **engine_test.py** | Depth threshold calculation |
| `MAX_QS_MOVES_Q3_DIVISOR` | 1.33 | **engine_test.py** | Depth threshold calculation |
| `CHECK_QS_MAX_DEPTH` | 5 | **engine_test.py** | How deep to search checks in QS |
| `DELTA_PRUNING_QS_MIN_DEPTH` | 5 | **engine_test.py** | When delta pruning kicks in |
| `DELTA_PRUNING_QS_MARGIN` | 75 | **engine_test.py** | Delta pruning aggressiveness |

**Note**: After finding optimal values with engine_test.py, always validate with a few stockfish.sh games to ensure no regression in playing strength.

---

### 🟢 QS TIME CONTROL (use **stockfish.sh**)

These parameters interact with time management and only matter in timed games:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `QS_SOFT_STOP_DIVISOR` | 7.0 | When to soft-stop in QS |
| `QS_TIME_CRITICAL_FACTOR` | 0.88 | Time pressure threshold |
| `MAX_QS_MOVES_TIME_CRITICAL` | 5 | Move limit under time pressure |
| `QS_TIME_CHECK_INTERVAL` | 25 | How often to check time in QS |
| `QS_TIME_BUDGET_FRACTION` | 0.35 | QS time allocation |
| `QS_TT_SUPPORTED` | False | QS transposition table |

---

### 🔵 MAIN SEARCH DEPTH CONTROL

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MIN_NEGAMAX_DEPTH` | 3 | **stockfish.sh** | Minimum depth before stopping - time-critical |
| `MIN_PREFERRED_DEPTH` | 5 | **stockfish.sh** | Target minimum depth - affects time usage |
| `TACTICAL_MIN_DEPTH` | 5 | **stockfish.sh** | Min depth for tactical positions |
| `UNSTABLE_MIN_DEPTH` | 5 | **stockfish.sh** | Min depth when scores are unstable |
| `MAX_NEGAMAX_DEPTH` | 20 | **Neither** | Upper bound, rarely reached |

---

### 🟣 TIME MANAGEMENT (use **stockfish.sh** exclusively)

These parameters ONLY matter in timed games and cannot be evaluated with engine_test.py:

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `EMERGENCY_TIME_RESERVE` | 0.75 | Time buffer to avoid flag |
| `ESTIMATED_BRANCHING_FACTOR` | 4.0 | For predicting search time |
| `TIME_SAFETY_MARGIN_RATIO` | 0.55 | When to start new depth |
| `MAX_SEARCH_TIME` | 30 | Maximum time per move |

**⚠️ Important**: Time management bugs cause flag losses. Always test with multiple game counts (20+ games minimum).

---

### 🟤 ASPIRATION WINDOWS

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `ASPIRATION_WINDOW` | 75 | **Both** | Start with engine_test.py for accuracy, validate with stockfish.sh |
| `MAX_AW_RETRIES` | 2 | **Both** | How many retries before full window |
| `MAX_AW_RETRIES_TACTICAL` | 3 | **engine_test.py** | Extra retries for tactical positions |

---

### ⚫ LATE MOVE REDUCTIONS (LMR)

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `LMR_MOVE_THRESHOLD` | 2 | **engine_test.py** | After which move to apply LMR |
| `LMR_MIN_DEPTH` | 3 | **engine_test.py** | Minimum depth for LMR |

**Note**: LMR affects tactical accuracy. Test with engine_test.py first, but if WAC scores drop, the ELO might still be fine (or vice versa). Validate with stockfish.sh.

---

### ⚪ NULL MOVE PRUNING

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `NULL_MOVE_REDUCTION` | 3 | **engine_test.py** | R value for null move |
| `NULL_MOVE_MIN_DEPTH` | 4 | **engine_test.py** | When to apply null move |

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
| `FUTILITY_MARGIN` | [0,150,300,450] | Margins per depth |
| `FUTILITY_MAX_DEPTH` | 3 | Maximum depth for futility |

**Razoring** (use **engine_test.py** primarily):

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `RAZORING_ENABLED` | False | Whether to use razoring |
| `RAZORING_MARGIN` | [0,125,250] | Margins per depth |
| `RAZORING_MAX_DEPTH` | 2 | Maximum depth for razoring |

---

### 📊 INFRASTRUCTURE

| Parameter | Default | Test Method | Rationale |
|-----------|---------|-------------|-----------|
| `MAX_TABLE_SIZE` | 200,000 | **stockfish.sh** | Affects memory/speed tradeoff over many games |
| `MAX_MP_CORES` | 1 | **stockfish.sh** | Multiprocessing, needs game testing |
| `IS_SHARED_TT_MP` | False | **stockfish.sh** | MP transposition table sharing |
| `IS_BLAS_ENABLED` | False | **stockfish.sh** | BLAS acceleration |

---

## Tuning Workflow

### Step 1: Tactical Parameters (engine_test.py)

```bash
# Baseline
python engine_test.py

# Test a parameter change
QS_DEPTH_MIN_NN_EVAL=8 python engine_test.py

# Compare success rates
```

**Target**: Maximize WAC/Eigenmann success rate without excessive time increase.

### Step 2: Time/Positional Parameters (stockfish.sh)

```bash
# Run baseline games
./stockfish.sh NeuroFish-baseline 1500 20

# Test parameter change
MIN_NEGAMAX_DEPTH=4 ./stockfish.sh NeuroFish-test 1500 20

# Compare results
```

**Target**: Maximize win rate against same Stockfish ELO level.

### Step 3: Validation

After tuning tactical parameters with engine_test.py:
```bash
# Validate that tactical improvements translate to playing strength
./stockfish.sh NeuroFish-optimized 1500 20
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
- `MAX_QS_MOVES_Q1/Q2/Q3/Q4`
- `MAX_QS_MOVES_Q1/Q2/Q3_DIVISOR`
- `MAX_QS_DEPTH`

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

## Summary

| If the parameter affects... | Use... |
|-----------------------------|--------|
| Move finding accuracy | engine_test.py |
| Time allocation/management | stockfish.sh |
| Search depth decisions | stockfish.sh |
| Pruning aggressiveness | engine_test.py (validate with stockfish.sh) |
| Neural network usage | stockfish.sh |
| Positional evaluation | stockfish.sh |
