# NeuroFish Time & Depth Management Guide

## Overview

NeuroFish uses a sophisticated time and depth management system with multiple interacting components. Understanding these interactions is crucial for tuning performance.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UCI TIME ALLOCATION                             │
│  (uci.py: converts clock time → movetime for this move)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ITERATIVE DEEPENING LOOP                          │
│  (engine.py: find_best_move - searches depth 1, 2, 3... until stopped) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  DEPTH    │   │   TIME    │   │  SCORE    │
            │ MINIMUMS  │   │  CHECKS   │   │ STABILITY │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         NEGAMAX SEARCH                                  │
│           (recursive alpha-beta with pruning techniques)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUIESCENCE SEARCH                                 │
│  (searches captures/checks until position is "quiet")                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: UCI Time Allocation (uci.py)

The UCI layer converts the chess clock information into a single `movetime` value.

### Input Parameters (from GUI)
```
go wtime 180000 btime 180000 winc 2000 binc 2000 movestogo 40
```

### Time Allocation Algorithm

```python
# Constants
OVERHEAD_MS = 500      # Buffer for Python/NN startup overhead
MIN_RESERVE_MS = 1500  # Always keep 1.5s in reserve

# Four time phases based on remaining time
if time_left < 3000:           # EMERGENCY: < 3 seconds
    movetime = (time_left - 500) / 10
    
elif time_left < 10000:        # CRITICAL: < 10 seconds  
    base_time = (time_left - MIN_RESERVE_MS) / 20
    movetime = base_time + increment * 0.5
    
elif time_left < 30000:        # LOW: < 30 seconds
    base_time = (time_left - MIN_RESERVE_MS) / effective_moves
    movetime = min(base_time + increment * 0.7, time_left / 12)
    
else:                          # NORMAL: >= 30 seconds
    base_time = (time_left - MIN_RESERVE_MS) / effective_moves
    movetime = min(base_time + increment * 0.8, time_left / 10)
```

### Key Tuning Points

| Constant | Location | Effect |
|----------|----------|--------|
| `OVERHEAD_MS` | uci.py | Larger = more conservative, less time per move |
| `MIN_RESERVE_MS` | uci.py | Larger = safer against flag, but less time per move |
| Divisors (10, 12, 20) | uci.py | Larger = more conservative time allocation |
| Increment multipliers (0.5, 0.7, 0.8) | uci.py | How aggressively to use increment |

### Visualization: Time Allocation

```
Clock: 180s remaining, 2s increment, 40 moves to go

Normal calculation:
  effective_moves = max(40, 25) = 40
  base_time = (180000 - 1500) / 40 = 4462ms
  with_increment = 4462 + 2000 * 0.8 = 6062ms
  max_for_move = (180000 - 1500) / 10 = 17850ms
  movetime = min(6062, 17850) - 500 = 5.56 seconds
```

---

## Part 2: TimeControl State Machine (engine.py)

The `TimeControl` class manages search termination with multiple stop signals.

### State Variables

```python
class TimeControl:
    time_limit = None      # Target time (from UCI allocation)
    start_time = None      # When search started
    stop_search = False    # HARD STOP - always honored immediately
    soft_stop = False      # SOFT STOP - honored after MIN_DEPTH
    hard_stop_time = None  # Absolute deadline (150% of time_limit)
```

### Stop Signal Hierarchy

```
Priority 1: stop_search = True
  └── Source: UCI "stop" command, or hard_stop_time exceeded
  └── Effect: Immediate termination, even mid-move
  └── Checked: Every negamax node, every QS node

Priority 2: soft_stop = True (after MIN_DEPTH)
  └── Source: time_limit exceeded (100%)
  └── Effect: Stop after current depth completes
  └── Checked: Between depths, mid-move in some cases

Priority 3: Time estimation (predictive)
  └── Source: Not enough time for next depth
  └── Effect: Don't start new depth
  └── Checked: Before starting each depth
```

### check_time() Function

```python
def check_time():
    elapsed = time.perf_counter() - TimeControl.start_time
    
    # At 100% of time: set soft_stop
    if elapsed >= TimeControl.time_limit:
        TimeControl.soft_stop = True
    
    # At 150% of time: force hard stop
    if current_time >= TimeControl.hard_stop_time:
        TimeControl.stop_search = True
```

### When check_time() is Called

| Location | Frequency | Purpose |
|----------|-----------|---------|
| Negamax entry | Every node | Catch runaway searches |
| Between root moves | Every move at depth | Fine-grained control |
| QS entry | Every `QS_TIME_CHECK_INTERVAL` nodes | Prevent QS explosion |
| QS moves | Every 5 moves | Additional safeguard |

---

## Part 3: Depth Management

### Minimum Depth Parameters

```python
MIN_NEGAMAX_DEPTH = 3      # Absolute minimum before soft_stop honored
MIN_PREFERRED_DEPTH = 5    # Target minimum for normal positions
TACTICAL_MIN_DEPTH = 5     # Minimum when position is tactical
UNSTABLE_MIN_DEPTH = 5     # Minimum when scores are unstable
```

### Depth Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    START DEPTH d                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ stop_search=T?  │──Yes──► STOP IMMEDIATELY
                    └────────┬────────┘
                             │No
                             ▼
                    ┌─────────────────┐
                    │ d <= current_   │
                    │ min_depth?      │──Yes──► MUST COMPLETE
                    └────────┬────────┘         (ignore soft_stop)
                             │No
                             ▼
                    ┌─────────────────┐
                    │ soft_stop=T?    │──Yes──► STOP AFTER THIS DEPTH
                    └────────┬────────┘
                             │No
                             ▼
                    ┌─────────────────┐
                    │ Enough time for │
                    │ next depth?     │──No───► STOP (predictive)
                    └────────┬────────┘
                             │Yes
                             ▼
                       SEARCH DEPTH d
```

### Dynamic Minimum Depth Selection

```python
# Determined at the start of each depth
if score_unstable:
    current_min_depth = UNSTABLE_MIN_DEPTH      # 5
elif is_tactical_position:
    current_min_depth = TACTICAL_MIN_DEPTH      # 5
else:
    current_min_depth = MIN_PREFERRED_DEPTH     # 5

# Example: If soft_stop triggers at depth 3 but min=5,
# the engine continues to depth 5 anyway
```

### Score Instability Detection

```python
_SCORE_INSTABILITY_THRESHOLD = 200  # centipawns

# After completing each depth:
if depth > 2:
    score_diff = abs(current_score - previous_score)
    if score_diff > 200:  # Big swing!
        score_unstable = True
        is_tactical_position = True
        # Clear soft_stop to force deeper search
        TimeControl.soft_stop = False
```

---

## Part 4: Time Estimation (Predictive Stopping)

### The Problem

Starting a new depth that can't complete wastes time and may return worse results than the completed shallower search.

### The Solution: Branching Factor Estimation

```python
ESTIMATED_BRANCHING_FACTOR = 4.0
TIME_SAFETY_MARGIN_RATIO = 0.55

# Before starting depth d+1:
remaining_time = time_limit - elapsed
estimated_next_depth_time = last_depth_time * ESTIMATED_BRANCHING_FACTOR

if remaining_time < estimated_next_depth_time * TIME_SAFETY_MARGIN_RATIO:
    # Don't start - probably won't finish
    break
```

### Example Calculation

```
Depth 6 completed in 0.8 seconds
Time remaining: 2.0 seconds

Estimated depth 7 time: 0.8 * 4.0 = 3.2 seconds
Required buffer: 3.2 * 0.55 = 1.76 seconds

2.0 >= 1.76? Yes → Start depth 7
```

### Emergency Reserve

```python
EMERGENCY_TIME_RESERVE = 0.75  # seconds

# Always keep some time in reserve
if remaining < EMERGENCY_TIME_RESERVE:
    if max_completed_depth >= MIN_PREFERRED_DEPTH:
        break  # Stop to preserve reserve
```

---

## Part 5: Quiescence Search Time Management

QS can explode exponentially. Multiple safeguards prevent this:

### 1. Depth Limits

```python
MAX_QS_DEPTH = 22  # Hard limit

# Move limits decrease with depth
if q_depth <= MAX_QS_DEPTH / 4:    move_limit = 10
elif q_depth <= MAX_QS_DEPTH / 2:  move_limit = 5
elif q_depth <= MAX_QS_DEPTH / 1.33: move_limit = 2
else:                               move_limit = 1
```

### 2. Time Budget

```python
QS_TIME_BUDGET_FRACTION = 0.35  # QS gets max 35% of move time

# In quiescence():
if elapsed > time_limit * QS_TIME_BUDGET_FRACTION:
    soft_stop = True  # QS exceeded its budget
```

### 3. Time-Critical Mode

```python
QS_TIME_CRITICAL_FACTOR = 0.88
MAX_QS_MOVES_TIME_CRITICAL = 5

# When running out of time:
if elapsed > time_limit * 0.88:
    move_limit = min(5, move_limit)  # Restrict moves severely
```

### 4. Soft Stop in Deep QS

```python
QS_SOFT_STOP_DIVISOR = 7.0

# Honor soft_stop earlier in QS than main search
if soft_stop and q_depth > MAX_QS_DEPTH / 7:  # ~depth 3
    return evaluate()  # Exit QS early
```

### QS Time Check Frequency

```python
QS_TIME_CHECK_INTERVAL = 25  # Check every 25 nodes

# In quiescence():
if total_qs_nodes % 25 == 0:
    check_time()
```

---

## Part 6: Parameter Interactions

### Critical Interaction Groups

#### Group A: Main Search Timing
```
EMERGENCY_TIME_RESERVE ←→ TIME_SAFETY_MARGIN_RATIO
         ↓                          ↓
How much buffer to keep    When to stop starting new depths
         ↓                          ↓
         └──────────┬───────────────┘
                    ▼
            Total time usage
```

**Tuning Trade-off**: More reserve = fewer flag losses, but shallower searches.

#### Group B: Depth Minimums
```
MIN_NEGAMAX_DEPTH ←→ MIN_PREFERRED_DEPTH ←→ TACTICAL_MIN_DEPTH
         ↓                   ↓                      ↓
Absolute minimum     Normal target          Tactical target
         ↓                   ↓                      ↓
         └───────────────────┴──────────────────────┘
                              ▼
                    Minimum search quality
```

**Tuning Trade-off**: Higher minimums = better move quality, but time overruns.

#### Group C: QS Time Allocation
```
QS_TIME_BUDGET_FRACTION ←→ QS_SOFT_STOP_DIVISOR
            ↓                        ↓
    Max QS time share        When QS honors soft_stop
            ↓                        ↓
            └────────────┬───────────┘
                         ▼
              QS thoroughness vs speed
```

**Tuning Trade-off**: More QS time = better tactical resolution, but less depth.

---

## Part 7: Critical Time Scenarios

### Scenario 1: Very Low Time (< 3 seconds)

```
Clock: 2.5 seconds remaining

UCI allocation:
  movetime = (2500 - 500) / 10 = 200ms = 0.2 seconds

Engine behavior:
  - hard_stop_time = 0.2 + 0.1 = 0.3 seconds (absolute deadline)
  - Emergency search if time_limit < 0.15s
  - Depth 1-2 only, minimal QS
```

### Scenario 2: Score Instability

```
Depth 4: score = +150 cp
Depth 5: score = -100 cp  (swing of 250 cp!)

Engine behavior:
  - Detects instability (250 > 200 threshold)
  - Sets is_tactical_position = True
  - Clears soft_stop to force deeper search
  - Uses UNSTABLE_MIN_DEPTH = 5 as minimum
```

### Scenario 3: QS Explosion

```
Position with many captures available
QS depth reaches 15 (MAX_QS_DEPTH / 1.5)

Engine behavior:
  - Move limit reduced to 2
  - If soft_stop set and depth > 3: exit QS
  - Time budget check: if > 35% time used, stop
  - Returns NN/classical eval instead of searching deeper
```

### Scenario 4: Aspiration Window Failure

```
Depth 7, aspiration window = [+50, +200]
All moves score < +50 (fail low)

Engine behavior:
  - Widen window, retry (up to MAX_AW_RETRIES times)
  - If max retries hit: full window search
  - Mark position as tactical
  - Future depths use wider windows
```

---

## Part 8: Configuration Parameters Summary

### Time Allocation (uci.py - tune with stockfish.sh)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| `OVERHEAD_MS` | 500 | More conservative, less time per move |
| `MIN_RESERVE_MS` | 1500 | Safer, but shallower searches |

### Search Timing (config.py - tune with stockfish.sh)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| `EMERGENCY_TIME_RESERVE` | 0.75 | More buffer, fewer flags, shallower |
| `ESTIMATED_BRANCHING_FACTOR` | 4.0 | More conservative depth prediction |
| `TIME_SAFETY_MARGIN_RATIO` | 0.55 | Stop starting new depths earlier |

### Depth Control (config.py - tune with stockfish.sh)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| `MIN_NEGAMAX_DEPTH` | 3 | Forces deeper search, risk of timeout |
| `MIN_PREFERRED_DEPTH` | 5 | Better moves, more time pressure |
| `TACTICAL_MIN_DEPTH` | 5 | Better tactical resolution |
| `UNSTABLE_MIN_DEPTH` | 5 | Forces resolution of unclear positions |

### QS Time Control (config.py)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| `MAX_QS_DEPTH` | 22 | Deeper tactical search (engine_test.py) |
| `QS_TIME_BUDGET_FRACTION` | 0.35 | More time for QS (stockfish.sh) |
| `QS_TIME_CHECK_INTERVAL` | 25 | Less overhead but less responsive |
| `QS_SOFT_STOP_DIVISOR` | 7.0 | Later QS soft stop |
| `QS_TIME_CRITICAL_FACTOR` | 0.88 | Later switch to time-critical mode |

### Aspiration Windows (config.py)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| `ASPIRATION_WINDOW` | 75 | Fewer retries but less precise |
| `MAX_AW_RETRIES` | 2 | More attempts before full window |
| `MAX_AW_RETRIES_TACTICAL` | 3 | More retries in tactical positions |

---

## Part 9: Diagnostic Counters

The engine tracks various events for debugging:

```python
_diag = {
    "time_overruns": 0,         # Search exceeded time limit
    "qs_time_cutoff": 0,        # QS terminated early
    "qs_budget_exceeded": 0,    # QS exceeded 35% time
    "shallow_search_d2": 0,     # Best move at depth 2
    "shallow_search_d3": 0,     # Best move at depth 3
    "emergency_reserve_stop": 0, # Stopped to preserve reserve
    "min_depth_forced": 0,      # Forced past soft_stop
    "tactical_extension": 0,    # Extended due to instability
    "score_instability": 0,     # Large score swings
}
```

Enable with `debug on` UCI command or `IS_DIAGNOSTIC=True` environment variable.

---

## Part 10: Tuning Recommendations

### For Time Trouble Issues (flagging)

1. Increase `EMERGENCY_TIME_RESERVE` (0.75 → 1.0)
2. Increase `MIN_RESERVE_MS` in uci.py (1500 → 2000)
3. Decrease `MIN_PREFERRED_DEPTH` (5 → 4)
4. Decrease `QS_TIME_BUDGET_FRACTION` (0.35 → 0.25)

### For Shallow Search Issues (poor moves)

1. Increase `MIN_PREFERRED_DEPTH` (5 → 6)
2. Increase `TACTICAL_MIN_DEPTH` (5 → 6)
3. Decrease `TIME_SAFETY_MARGIN_RATIO` (0.55 → 0.45)
4. Increase `MAX_QS_DEPTH` (22 → 25)

### For QS Explosion Issues

1. Decrease `MAX_QS_DEPTH` (22 → 18)
2. Decrease `QS_TIME_BUDGET_FRACTION` (0.35 → 0.25)
3. Decrease `MAX_QS_MOVES_Q1` (10 → 8)
4. Increase `QS_TIME_CHECK_INTERVAL` (25 → 15 for more checks)

### For Score Instability Issues

1. Increase `_SCORE_INSTABILITY_THRESHOLD` (200 → 250)
2. Increase `ASPIRATION_WINDOW` (75 → 100)
3. Increase `MAX_AW_RETRIES_TACTICAL` (3 → 4)

---

## Appendix: Time Flow Diagram

```
UCI: "go wtime 30000 btime 30000 winc 1000 binc 1000"
                    │
                    ▼
    ┌───────────────────────────────┐
    │ UCI Time Allocation           │
    │ time_left=30000, inc=1000     │
    │ movetime = ~1.5 seconds       │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ find_best_move()              │
    │ time_limit = 1.5s             │
    │ hard_stop = 2.25s             │
    └───────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    Depth 1                 Depth 2
    (0.01s)                 (0.03s)
        │                       │
        ▼                       ▼
    Depth 3                 Depth 4
    (0.08s)                 (0.25s)
        │                       │
        ▼                       ▼
    Depth 5                 Depth 6
    (0.60s)                 Est: 2.4s
        │                       │
        │               ┌───────┘
        │               ▼
        │    remaining=0.53s < 2.4*0.55=1.32s
        │           │
        │           ▼
        │    DON'T START DEPTH 6
        │           │
        └───────────┴───────────┐
                                ▼
                    Return depth 5 result
                    Total time: ~0.97s
```
