import os

# Track which parameters were overridden from environment
_overridden_params = []


def _env_int(key, default):
    """Get integer from environment variable."""
    val = os.environ.get(key)
    if val is not None:
        result = int(val)
        _overridden_params.append((key, default, result))
        return result
    return default


def _env_float(key, default):
    """Get float from environment variable."""
    val = os.environ.get(key)
    if val is not None:
        result = float(val)
        _overridden_params.append((key, default, result))
        return result
    return default


def _env_bool(key, default):
    """Get boolean from environment variable (accepts true/false/1/0/yes/no)."""
    val = os.environ.get(key)
    if val is None:
        return default
    result = val.lower() in ('true', '1', 'yes', 'on')
    _overridden_params.append((key, default, result))
    return result


def _env_str(key, default):
    """Get string from environment variable."""
    val = os.environ.get(key)
    if val is not None:
        _overridden_params.append((key, default, val))
        return val
    return default


MAX_SCORE = _env_int('MAX_SCORE', 10_000)
TANH_SCALE = _env_int('TANH_SCALE', 410)  # Stockfish value

# -------- DIAGNOSTIC CONTROL --------
# Set IS_DIAGNOSTIC = True for development/debugging builds
# Or enable via UCI "debug on" command
DIAGNOSTIC = _env_bool('DIAGNOSTIC', False)  # Master switch for diagnostic output
debug_mode = _env_bool('DEBUG_MODE', False)  # Runtime toggle via UCI "debug on/off"

PONDERING_ENABLED = _env_bool('PONDERING_ENABLED', True)

# Multiprocessing configuration
THREADS = _env_int('THREADS', 2)  # 1 or less disables multiprocessing, UCI option "Threads"

LAZY_SMP_MOVE_ORDER_RANDOMNESS = _env_int('LAZY_SMP_MOVE_ORDER_RANDOMNESS', 2)  # ±N centipawns noise (reduced from 5)
LAZY_SMP_DEPTH_OFFSET = _env_int('LAZY_SMP_DEPTH_OFFSET', 1)  # Stagger worker starting depths by this amount

MULTI_CORE_BLAS = _env_bool('MULTI_CORE_BLAS', False)

NN_ENABLED = _env_bool('NN_ENABLED', True)
FULL_NN_EVAL_FREQ = _env_int('FULL_NN_EVAL_FREQ', 3000)  # Increase to 50_000 after initial testing

L1_QUANTIZATION = _env_str('L1_QUANTIZATION', "INT8")  # Options: "NONE" (FP32), "INT8", "INT16"

# Note when NN related parameters are optimized, use real games as positional understanding will be reflected.
# The non-NN parameters are primarily about tactics, and they can be quickly tuned using test positions.
QS_DEPTH_MIN_NN_EVAL = _env_int('QS_DEPTH_MIN_NN_EVAL', 5)  #
NEGAMAX_PLY_MIN_NN_EVAL = _env_int('NEGAMAX_PLY_MIN_NN_EVAL', 4)  #
QS_DEPTH_MAX_NN_EVAL = _env_int('QS_DEPTH_MAX_NN_EVAL', 8)  # NN evaluation depth limit (reduced from 999, now sensible with adaptive QS)
QS_DELTA_MAX_NN_EVAL = _env_int('QS_DELTA_MAX_NN_EVAL', 100)  # Score difference, below it will trigger a NN evaluation
STAND_PAT_MAX_NN_EVAL = _env_int('STAND_PAT_MAX_NN_EVAL',
                                 200)  # Absolute value of stand-pat, below it will trigger a NN evaluation.

# ===== ADAPTIVE QUIESCENCE DEPTH =====
# QS depth now scales with negamax depth to prevent shallow-depth explosion
# Formula: max_qs_depth = QS_DEPTH_BASE + (negamax_depth * QS_DEPTH_PER_PLY)
# Examples:
#   D1: QS_DEPTH = 4 + (1 × 2) = 6
#   D2: QS_DEPTH = 4 + (2 × 2) = 8
#   D3: QS_DEPTH = 4 + (3 × 2) = 10
#   D4: QS_DEPTH = 4 + (4 × 2) = 12
#   D5: QS_DEPTH = 4 + (5 × 2) = 14
QS_DEPTH_BASE = _env_int('QS_DEPTH_BASE', 8)  # Minimum QS depth (increased from 4 to fix leaf node issue)
QS_DEPTH_PER_PLY = _env_int('QS_DEPTH_PER_PLY', 2)  # Add this many QS plies per negamax depth

# Limit moves examined per QS ply to prevent explosion
# MAX_QS_DEPTH is now deprecated in favor of adaptive calculation, but kept for compatibility
MAX_QS_DEPTH = _env_int('MAX_QS_DEPTH', 18)  # Fallback if adaptive calculation disabled
_max_qs_moves_default = [5, 4, 3, 2]  # Reduced from [12, 6, 4, 2]
_max_q_moves_env = os.environ.get('MAX_QS_MOVES')
if _max_q_moves_env:
    MAX_QS_MOVES = eval(_max_q_moves_env)
    _overridden_params.append(('MAX_QS_MOVES', _max_qs_moves_default, MAX_QS_MOVES))
else:
    MAX_QS_MOVES = _max_qs_moves_default

# MAX_QS_MOVES_DIVISOR divisors divide adaptive QS max depth into segments
_max_qs_moves_divisor_default = [4, 2.0, 1.33]
_max_q_moves_divisor_env = os.environ.get('MAX_QS_MOVES_DIVISOR')
if _max_q_moves_divisor_env:
    MAX_QS_MOVES_DIVISOR = eval(_max_q_moves_divisor_env)
    _overridden_params.append(('MAX_QS_MOVES_DIVISOR', _max_qs_moves_divisor_default, MAX_QS_MOVES_DIVISOR))
else:
    MAX_QS_MOVES_DIVISOR = _max_qs_moves_divisor_default

QS_SOFT_STOP_DIVISOR = _env_float('QS_SOFT_STOP_DIVISOR', 2.5)  # Soft-stop earlier (increased from 4.5)
QS_TIME_CRITICAL_FACTOR = _env_float('QS_TIME_CRITICAL_FACTOR', 0.86)
MAX_QS_MOVES_TIME_CRITICAL = _env_int('MAX_QS_MOVES_TIME_CRITICAL', 5)
DELTA_PRUNING_QS_MIN_DEPTH = _env_int('DELTA_PRUNING_QS_MIN_DEPTH', 5)
DELTA_PRUNING_QS_MARGIN = _env_int('DELTA_PRUNING_QS_MARGIN', 75)
CHECK_QS_MAX_DEPTH = _env_int('CHECK_QS_MAX_DEPTH', 5)
QS_TIME_CHECK_INTERVAL = _env_int('QS_TIME_CHECK_INTERVAL', 30)  # More frequent checks (reduced from 80)
QS_TIME_BUDGET_FRACTION = _env_float('QS_TIME_BUDGET_FRACTION', 0.35)  # Tighter budget (reduced from 0.45)
QS_TT_SUPPORTED = _env_bool('QS_TT_SUPPORTED', False)

# Minimum depth requirements
# Tuning of depth adjustment should be done playing against stockfish (not using engine_test.py)
MIN_NEGAMAX_DEPTH = _env_int('MIN_NEGAMAX_DEPTH', 4)  # Minimum depth before soft_stop is honored
MIN_PREFERRED_DEPTH = _env_int('MIN_PREFERRED_DEPTH', 5)  # Preferred minimum depth
TACTICAL_MIN_DEPTH = _env_int('TACTICAL_MIN_DEPTH', 5)  # Minimum depth for tactical positions
UNSTABLE_MIN_DEPTH = _env_int('UNSTABLE_MIN_DEPTH', 5)  # Minimum depth when score instability

# Time management
# Tuning of time management should be done playing against stockfish (not using engine_test.py)
EMERGENCY_TIME_RESERVE = _env_float('EMERGENCY_TIME_RESERVE', 0.50)  # Always keep at least 0.5s
ESTIMATED_BRANCHING_FACTOR = _env_float('ESTIMATED_BRANCHING_FACTOR', 4.0)
TIME_SAFETY_MARGIN_RATIO = _env_float('TIME_SAFETY_MARGIN_RATIO', 0.45)  # Only start new depth if 70%+ time available

ASPIRATION_WINDOW = _env_int('ASPIRATION_WINDOW', 100)  # Increased from 75 for stability
MAX_AW_RETRIES = _env_int('MAX_AW_RETRIES', 1)  # Base retries (tactical positions get +1)
MAX_AW_RETRIES_TACTICAL = _env_int('MAX_AW_RETRIES_TACTICAL', 3)  # More retries for tactical positions

LMR_MOVE_THRESHOLD = _env_int('LMR_MOVE_THRESHOLD', 2)
LMR_MIN_DEPTH = _env_int('LMR_MIN_DEPTH', 4)  # minimum depth to apply LMR

NULL_MOVE_REDUCTION = _env_int('NULL_MOVE_REDUCTION', 2)  # R value (usually 2 or 3)
NULL_MOVE_MIN_DEPTH = _env_int('NULL_MOVE_MIN_DEPTH', 3)

SINGULAR_MARGIN = _env_int('SINGULAR_MARGIN', 130)  # Score difference in centipawns
SINGULAR_EXTENSION = _env_int('SINGULAR_EXTENSION', 1)  # Extra depth

# SEE Pruning - prune losing captures at low depths
SEE_PRUNING_ENABLED = _env_bool('SEE_PRUNING_ENABLED', False)
SEE_PRUNING_MAX_DEPTH = _env_int('SEE_PRUNING_MAX_DEPTH', 6)  # Only apply at shallow depths

# Futility Pruning - skip quiet moves when position is hopeless
FUTILITY_PRUNING_ENABLED = _env_bool('FUTILITY_PRUNING_ENABLED', True)
# Note: FUTILITY_MARGIN is a list - use JSON format in env var, e.g. "[0,150,300,450]"
_futility_default = [0, 150, 300, 450]
_futility_env = os.environ.get('FUTILITY_MARGIN')
if _futility_env:
    FUTILITY_MARGIN = eval(_futility_env)
    _overridden_params.append(('FUTILITY_MARGIN', _futility_default, FUTILITY_MARGIN))
else:
    FUTILITY_MARGIN = _futility_default
FUTILITY_MAX_DEPTH = _env_int('FUTILITY_MAX_DEPTH', 3)  # Only apply at depth <= 3

# Razoring - drop into quiescence when far below alpha
RAZORING_ENABLED = _env_bool('RAZORING_ENABLED', False)
RAZORING_MAX_DEPTH = _env_int('RAZORING_MAX_DEPTH', 2)
_razoring_default = [0, 300, 500]
_razoring_env = os.environ.get('RAZORING_MARGIN')
if _razoring_env:
    RAZORING_MARGIN = eval(_razoring_env)
    _overridden_params.append(('RAZORING_MARGIN', _razoring_default, RAZORING_MARGIN))
else:
    RAZORING_MARGIN = _razoring_default

MAX_NEGAMAX_DEPTH = _env_int('MAX_NEGAMAX_DEPTH', 20)
MAX_SEARCH_TIME = _env_int('MAX_SEARCH_TIME', 30)

# Transposition table
MAX_TABLE_SIZE = _env_int('MAX_TABLE_SIZE', 750_000)  # Increased from 200_000
NUM_SHARDS_TABLES = _env_int('NUM_SHARDS_TABLES', 16)

# Book
OPENING_BOOK_ENABLED = _env_bool('OPENING_BOOK_ENABLED', True)

# Resign
RESIGN_THRESHOLD = _env_int('RESIGN_THRESHOLD', -500)
RESIGN_MOVES = _env_int('RESIGN_MOVES', 3)


def print_overridden_params():
    """Print parameters overridden from environment."""
    if _overridden_params:
        print("=" * 60)
        print("CONFIGURATION OVERRIDES FROM ENVIRONMENT:")
        print("=" * 60)
        for key, default, value in _overridden_params:
            print(f"  {key}: {default} -> {value}")
        print("=" * 60)