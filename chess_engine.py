import heapq
import os
import re
import sys
import time
import traceback
import threading
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Optional
import chess
import chess.polyglot
import numpy as np

import config
from cached_board import CachedBoard, move_to_int, int_to_move
from nn_evaluator import DNNEvaluator, NNUEEvaluator, NNEvaluator

SCRIPT_DIR = Path(__file__).resolve().parent
HOME_DIR = "neurofish"
DNN_MODEL_FILEPATH = SCRIPT_DIR / f"../{HOME_DIR}" / 'model' / 'dnn.pt'
NNUE_MODEL_FILEPATH = SCRIPT_DIR / f"../{HOME_DIR}" / 'model' / 'nnue.pt'


MODEL_PATH = str(DNN_MODEL_FILEPATH if config.NN_TYPE == "DNN" else NNUE_MODEL_FILEPATH)

def is_debug_enabled() -> bool:
    """Check if diagnostic output is enabled (either via IS_DIAGNOSTIC or UCI debug on)."""
    return config.DIAGNOSTIC or config.debug_mode


def set_debug_mode(enabled: bool):
    """Set debug mode from UCI command."""
    config.debug_mode = enabled


class TimeControl:
    time_limit = None  # in seconds
    start_time = None
    stop_search = False  # Set by UCI 'stop' command - always honored
    soft_stop = False  # Set by time limit - ignored until MIN_DEPTH reached
    hard_stop_time = None  # FIX V3: Absolute time when search MUST stop (150% of time_limit)
    is_ponder_search = False  # True during ponder search - skip early abort before NN init


TTEntry = namedtuple("TTEntry", ["depth", "score", "flag", "best_move_int"])
TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND = 0, 1, 2

transposition_table = {}
qs_transposition_table = {}
dnn_eval_cache = {}
killer_moves = [[None, None] for _ in range(config.MAX_NEGAMAX_DEPTH + 1)]  # Legacy - kept for compatibility
killer_moves_int = [[0, 0] for _ in range(config.MAX_NEGAMAX_DEPTH + 1)]  # Primary: Integer-based killer moves

# Array-based history heuristic for O(1) access
# Max move_int is from_sq(63) | to_sq(63)<<6 | promo(5)<<12 = 63 + 63*64 + 5*4096 = 24575
HISTORY_TABLE_SIZE = 25000
history_heuristic = np.zeros(HISTORY_TABLE_SIZE, dtype=np.int32)


def pv_int_to_moves(pv_int: List[int]) -> List[chess.Move]:
    """Convert integer PV to chess.Move list for UCI output."""
    return [int_to_move(m) for m in pv_int if m != 0]


kpi = {
    "nodes": 0,
    "pos_eval": 0,
    "nn_evals": 0,
    "beta_cutoffs": 0,
    "tt_hits": 0,
    "qs_tt_hits": 0,
    "dec_hits": 0,  # DNN evaluation cache hits
    "q_depth": 0,
    "see_prunes": 0,  # SEE pruning
    "futility_prunes": 0,  # Futility pruning
    "razoring_prunes": 0,  # Razoring
}

# -------- DIAGNOSTIC COUNTERS (for debugging) --------
# These track anomalies without flooding output. Warnings printed only when threshold exceeded.
_diag = {
    "tt_illegal_moves": 0,  # TT returned illegal move
    "score_out_of_bounds": 0,  # Score exceeded MAX_SCORE
    "time_overruns": 0,  # Search ran over time limit
    "pv_illegal_moves": 0,  # PV contained illegal move
    "eval_drift": 0,  # Incremental vs full eval mismatch
    "qs_depth_exceeded": 0,  # QS hit MAX_QS_DEPTH
    "aspiration_retries": 0,  # Aspiration window retries hit max
    "best_move_none": 0,  # Search completed but best_move is None (fallback used)
    "score_instability": 0,  # Large score swings between depths
    # Time and QS diagnostic counters
    "qs_time_cutoff": 0,  # QS terminated early due to time
    "qs_move_limit": 0,  # QS move limit reached
    "qs_shallow_protected": 0,  # NEW: Shallow QS protected from aggressive limits
    "qs_budget_exceeded": 0,  # NEW: QS exceeded its time budget fraction
    "fallback_shallow_search": 0,  # Used shallow search fallback instead of pure NN
    "aw_tactical_skip": 0,  # Skipped aspiration window due to tactical position
    "time_critical_abort": 0,  # Search aborted due to critical time pressure
    # Depth tracking counters
    "shallow_search_d2": 0,  # NEW: Bestmove at depth 2
    "shallow_search_d3": 0,  # NEW: Bestmove at depth 3
    "shallow_search_total": 0,  # NEW: Bestmove at depth <= 3
    "tactical_extension": 0,  # Search extended due to score instability
    "min_depth_forced": 0,  # Forced deeper search due to MIN_ACCEPTABLE_DEPTH
    "mid_depth_abort": 0,  # NEW: Search aborted mid-depth due to soft_stop
    "critical_time_search": 0,  # FIX V3: Emergency search due to critically low time
    "emergency_reserve_stop": 0,  # FIX V4: Stopped to preserve emergency time reserve
    "unstable_min_depth": 0,  # FIX V4: Used higher min depth due to score instability
    "bestmove_depth_sum": 0,  # NEW: Sum of depths at bestmove (for average)
    "bestmove_count": 0,  # NEW: Count of bestmoves (for average)
}
_DIAG_WARN_THRESHOLD = 3  # Print warning after this many occurrences
_DIAG_SAMPLE_RATE = 100  # Check expensive diagnostics every N nodes
_SCORE_INSTABILITY_THRESHOLD = 200  # cp swing that triggers warning

# Track QS statistics for current search
_qs_stats = {
    "max_depth_reached": 0,
    "total_nodes": 0,
    "time_cutoffs": 0,
}


def _diag_warn(key: str, msg: str):
    """Record diagnostic event and warn if threshold exceeded (only when diagnostics enabled)."""
    _diag[key] += 1
    if not is_debug_enabled():
        return  # Skip output when diagnostics disabled
    count = _diag[key]
    # Print first occurrence, then at threshold, then every 10x threshold
    if count == 1 or count == _DIAG_WARN_THRESHOLD or (
            count > _DIAG_WARN_THRESHOLD and count % (_DIAG_WARN_THRESHOLD * 10) == 0):
        print(f"info string DIAG[{key}={count}]: {msg}", flush=True)


def diag_summary() -> str:
    """Return summary of diagnostic counters (non-zero only). Only meaningful when diagnostics enabled."""
    non_zero = {k: v for k, v in _diag.items() if v > 0 and not k.startswith("bestmove_")}
    summary_parts = []

    if non_zero:
        summary_parts.append(", ".join(f"{k}={v}" for k, v in non_zero.items()))

    # Add average depth if we have data
    if _diag["bestmove_count"] > 0:
        avg_depth = _diag["bestmove_depth_sum"] / _diag["bestmove_count"]
        summary_parts.append(f"avg_depth={avg_depth:.1f}")

    if summary_parts:
        return "DIAG_SUMMARY: " + " | ".join(summary_parts)
    return "DIAG_SUMMARY: all clear"


def diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)

def configure_multi_thread_blas() -> None:
    if not config.MULTI_THREAD_BLAS:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        print("✓ Disabling multi-threaded BLAS", file=sys.stderr)
    else:
        del os.environ['OPENBLAS_NUM_THREADS']
        del os.environ['MKL_NUM_THREADS']
        del os.environ['OMP_NUM_THREADS']
        print("✓ Allowing multi-threaded BLAS", file=sys.stderr)

configure_multi_thread_blas()

# Track positions seen in the current game (cleared on ucinewgame)
game_position_history: dict[int, int] = {}  # zobrist_hash -> count

# Thread-local storage for nn_evaluator instances
# Each thread gets its own evaluator to avoid state conflicts
_thread_local = threading.local()

# NN evaluator pool for reuse across search threads (avoids ~0.15s creation per thread)
_nn_eval_pool: list = []  # Pool of idle NNEvaluator instances
_nn_eval_pool_lock = threading.Lock()

# Main thread's nn_evaluator (also serves as template for creating thread-local copies)
nn_evaluator: None | NNEvaluator = None

def configure_nn_type():
    global nn_evaluator
    if config.NN_TYPE == "DNN":
        nn_evaluator = DNNEvaluator.create(CachedBoard(), config.NN_TYPE, MODEL_PATH)  # Loads model
    else:
        nn_evaluator = NNUEEvaluator.create(CachedBoard(), config.NN_TYPE, MODEL_PATH)

configure_nn_type()

def get_nn_evaluator() -> NNEvaluator:
    """
    Get the nn_evaluator for the current thread.

    For the main thread, returns the global nn_evaluator.
    For worker threads, creates and caches a thread-local instance
    (reused from pool when possible to avoid ~0.15s creation overhead).
    """
    # Check if we're in main thread - use global evaluator
    if threading.current_thread() is threading.main_thread():
        return nn_evaluator

    # For worker threads, use thread-local evaluator
    if not hasattr(_thread_local, 'nn_evaluator'):
        # Try to get an evaluator from the pool first
        pooled = None
        with _nn_eval_pool_lock:
            if _nn_eval_pool:
                pooled = _nn_eval_pool.pop()

        if pooled is not None:
            thread_name = threading.current_thread().name
            diag_print(f"Reusing pooled nn_evaluator for {thread_name}")
            _thread_local.nn_evaluator = pooled
        else:
            # Create a new evaluator instance for this thread using the factory method
            thread_name = threading.current_thread().name
            diag_print(f"Creating thread-local nn_evaluator for {thread_name}")
            _thread_local.nn_evaluator = NNEvaluator.create(CachedBoard(), config.NN_TYPE, MODEL_PATH)

    return _thread_local.nn_evaluator


def return_nn_evaluator_to_pool():
    """
    Return this thread's nn_evaluator to the pool for reuse.
    Call this at the end of a search thread before it exits.
    """
    if hasattr(_thread_local, 'nn_evaluator'):
        with _nn_eval_pool_lock:
            _nn_eval_pool.append(_thread_local.nn_evaluator)
        del _thread_local.nn_evaluator


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}


def clear_game_history():
    """Clear game position history (call on ucinewgame)."""
    game_position_history.clear()
    # Also reset diagnostic counters for fresh tracking
    for key in _diag:
        _diag[key] = 0


def is_draw_by_repetition(board: CachedBoard) -> bool:
    """
    Check for threefold repetition combining game history and search path.

    A position is a draw if it occurs 3+ times total:
    - game_position_history counts occurrences in the actual game (including current position)
    - board.is_repetition() counts occurrences in the search tree

    Key insight: The ROOT position of search is already counted in game_history.
    If we return to that position during search, is_repetition(2) will be True,
    but we shouldn't double-count.

    Simple approach:
    - If game_count >= 3: already 3-fold in game alone
    - If game_count == 2: one more occurrence makes it 3-fold
      → is_repetition(2) means we've seen this position again in search
    - If game_count == 1: need 2 more occurrences
      → is_repetition(2) means we've seen this position 2+ times in current search line
    - If game_count == 0: need 3 occurrences in search alone
      → is_repetition(3)
    """
    key = board.zobrist_hash()
    game_count = game_position_history.get(key, 0)

    # Already occurred 3+ times in game
    if game_count >= 3:
        return True

    # Occurred twice in game - need to see it once more in search
    # is_repetition(2) = position occurred 2+ times including current
    if game_count == 2:
        # Any occurrence in search (beyond root) makes it 3-fold
        # is_repetition(2) catches: we're at this position AND we were here before in search
        if board.is_repetition(2):
            return True

    # Occurred once in game - need to see it twice more in search
    # is_repetition(3) = position occurred 3+ times including current
    if game_count == 1:
        if board.is_repetition(2):
            # We've seen it twice in search (including current), plus once in game = 3
            return True

    # Never in game - need 3 occurrences in search
    if game_count == 0:
        if board.is_repetition(3):
            return True

    return False


def check_time():
    """Check if time limit exceeded. Sets soft_stop flag, and stop_search if way over."""
    if TimeControl.time_limit is None:
        return

    current_time = time.perf_counter()
    elapsed = current_time - TimeControl.start_time

    # Soft stop at 100% of time limit
    if elapsed >= TimeControl.time_limit:
        TimeControl.soft_stop = True

    # FIX V3: HARD stop at 150% of time limit - force immediate termination
    # This prevents the massive overruns (7+ seconds when given 0.05s)
    if TimeControl.hard_stop_time and current_time >= TimeControl.hard_stop_time:
        TimeControl.stop_search = True
        TimeControl.soft_stop = True


def evaluate_classical(board: CachedBoard, skip_game_over: bool = False) -> int:
    """
    Evaluate the board position from the side-to-move perspective.

    Added skip_game_over parameter to avoid redundant checks when
    caller already knows the position is not game over.

    Returns:
        Score in centipawns. Positive = good for side to move.
    """
    # Check for game over conditions (skip if caller already checked)
    if not skip_game_over and board.is_game_over():
        if board.is_checkmate():
            # Side to move is checkmated - worst possible score
            return -config.MAX_SCORE + board.ply()
        else:
            return 0  # Stalemate or other draw

    kpi['pos_eval'] += 1
    score = board.material_evaluation()
    return score


def evaluate_nn(board: CachedBoard, skip_game_over: bool = False) -> int:
    """
    Added skip_game_over parameter to avoid redundant checks when
    caller already knows the position is not game over.
    """
    if not skip_game_over and board.is_game_over():
        if board.is_checkmate():
            # Side to move is checkmated - worst possible score
            return -config.MAX_SCORE + board.ply()
        else:
            return 0  # Stalemate or other draw

    key = board.zobrist_hash()
    if key in dnn_eval_cache:
        kpi['dec_hits'] += 1
        return dnn_eval_cache[key]

    # Get thread-local evaluator
    evaluator = get_nn_evaluator()

    # assert (board.is_quiet_position())
    kpi['nn_evals'] += 1
    score = evaluator.evaluate_centipawns(board)
    dnn_eval_cache[key] = score

    # Occasionally do a full evaluation to rule out any drift errors.
    if kpi['nn_evals'] % config.FULL_NN_EVAL_FREQ == 0:
        full_score = evaluator.evaluate_full_centipawns(board)
        if abs(full_score - score) > 10:
            _diag_warn("eval_drift", f"Incr={score} vs Full={full_score}, diff={abs(full_score - score)}")
    return score


def move_score_q_search(board: CachedBoard, move: chess.Move) -> int:
    """
    Score a move for quiescence search ordering.
    Uses cached move info for efficiency.
    """
    score = 0

    # Use cached is_capture and piece types
    if board.is_capture_cached(move):
        victim_type = board.get_victim_type(move)
        attacker_type = board.get_attacker_type(move)
        if victim_type and attacker_type:
            score += 10 * PIECE_VALUES[victim_type] - PIECE_VALUES[attacker_type]

    return score


def move_score(board: CachedBoard, move: chess.Move, depth: int) -> int:
    """
    Score a move for move ordering.
    Uses cached move info for efficiency.
    """
    score = 0

    # Use cached is_capture
    is_capture = board.is_capture_cached(move)

    # Killer moves (quiet moves only)
    if not is_capture and depth is not None and 0 <= depth < len(killer_moves):
        if move == killer_moves[depth][0]:
            score += 9000
        elif move == killer_moves[depth][1]:
            score += 8000

    if is_capture:
        # Use cached victim and attacker types
        victim_type = board.get_victim_type(move)
        attacker_type = board.get_attacker_type(move)
        if victim_type and attacker_type:
            score += 10 * PIECE_VALUES[victim_type] - PIECE_VALUES[attacker_type]

    # Use array-based history heuristic for O(1) access
    move_int = move_to_int(move)
    if move_int < HISTORY_TABLE_SIZE:
        score += history_heuristic[move_int]

    # Use cached gives_check
    if board.gives_check_cached(move):
        score += 50

    return score


def move_score_int(board: CachedBoard, move_int: int, depth: int) -> int:
    """
    OPTIMIZATION: Score a move for move ordering using integer representation.

    This avoids chess.Move object creation and uses integer-keyed caches.
    """
    score = 0

    # Use cached is_capture (integer version) - has fast bitboard path for non-captures
    is_capture = board.is_capture_int(move_int)

    # Killer moves (quiet moves only) - now using integer comparison
    if not is_capture and depth is not None and 0 <= depth < len(killer_moves_int):
        if move_int == killer_moves_int[depth][0]:
            score += 9000
        elif move_int == killer_moves_int[depth][1]:
            score += 8000

    if is_capture:
        # OPTIMIZED: Use pre-computed MVV-LVA score (1 lookup instead of 3)
        score += board.get_mvv_lva_int(move_int)

    # Array-based history heuristic for O(1) access
    if move_int < HISTORY_TABLE_SIZE:
        score += history_heuristic[move_int]

    # Use cached gives_check (integer version)
    if board.gives_check_int(move_int):
        score += 50

    return score


def move_score_q_search_int(board: CachedBoard, move_int: int) -> int:
    """
    OPTIMIZED: Score a move for quiescence search using pre-computed MVV-LVA.

    This uses board.get_mvv_lva_int() which returns the pre-computed score,
    eliminating 3 dict lookups (is_capture, victim_type, attacker_type).
    """
    return board.get_mvv_lva_int(move_int)


def ordered_moves_int(board: CachedBoard, depth: int, pv_move_int: int = 0, tt_move_int: int = 0) -> List[int]:
    """
    OPTIMIZATION: Return legal moves as integers, ordered by expected quality.

    This completely avoids chess.Move object creation in the move ordering hot path.

    Args:
        board: The board position
        depth: Current search depth (for killer moves)
        pv_move_int: Principal variation move as integer (0 if none)
        tt_move_int: Transposition table move as integer (0 if none)

    Returns:
        List of integer moves, sorted by score (best first)
    """
    # Ensure move info is precomputed using integer keys
    import  random
    board.precompute_move_info_int()

    scored_moves = []
    for move_int in board.get_legal_moves_int():
        if move_int == tt_move_int and tt_move_int != 0:
            score = 1000000
        elif move_int == pv_move_int and pv_move_int != 0:
            score = 900000
        else:
            score = move_score_int(board, move_int, depth)
        scored_moves.append((score, move_int))

    # Sort by score descending, with random tiebreaker
    scored_moves.sort(key=lambda tup: (tup[0], random.random()), reverse=True)
    return [move_int for _, move_int in scored_moves]


def ordered_moves_q_search_int(board: CachedBoard, last_capture_sq: int = -1) -> List[int]:
    """
    OPTIMIZATION: Return legal moves for quiescence search as integers.

    FIX: Added last_capture_sq parameter to prioritize recaptures on that square.
    This ensures capture sequences are properly resolved before soft-stop.
    """
    board.precompute_move_info_int()

    moves_int = board.get_legal_moves_int()
    moves_with_scores = []

    for m in moves_int:
        score = move_score_q_search_int(board, m)
        to_sq = (m >> 6) & 0x3F

        # FIX: Prioritize recaptures on the last capture square with a large bonus
        if 0 <= last_capture_sq == to_sq:
            score += 100000  # Ensure recaptures are searched first

        moves_with_scores.append((score, m))

    # OPTIMIZED: Only get top moves instead of sorting all
    # Use MAX_QS_MOVES[0] (=12) as upper bound - actual limit applied in quiescence()
    top_moves = heapq.nlargest(config.MAX_QS_MOVES[0], moves_with_scores)
    return [m for _, m in top_moves]


def see(board: CachedBoard, move: chess.Move) -> int:
    """
    Simplified Static Exchange Evaluation (SEE).

    Returns the estimated material gain/loss in centipawns from the capture.
    Positive = likely winning exchange, Negative = likely losing exchange.

    This is a simplified SEE based on MVV-LVA (Most Valuable Victim - Least Valuable Attacker).
    For a full SEE, we would need to simulate the entire exchange sequence.
    """
    # Get victim and attacker types from cache
    victim_type = board.get_victim_type(move)
    attacker_type = board.get_attacker_type(move)

    if victim_type is None:
        # Not a capture - check for promotion gain
        if move.promotion:
            return PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]
        return 0

    victim_value = PIECE_VALUES.get(victim_type, 0)
    attacker_value = PIECE_VALUES.get(attacker_type, 0)

    # Handle promotions - add the promotion piece value minus pawn value
    promotion_gain = 0
    if move.promotion:
        promotion_gain = PIECE_VALUES.get(move.promotion, 0) - PIECE_VALUES[chess.PAWN]

    # Simple MVV-LVA approximation:
    # If we capture something worth more than our piece, it's likely good.
    # If we capture something worth less with a valuable piece, it might be bad
    # unless we're protected (which we don't check in this simplified version).

    # Basic heuristic: assume the opponent can recapture with a pawn if our piece
    # lands on a square that could be defended. This is conservative.

    # Simple approximation:
    # - If victim >= attacker: good trade (we win at least the difference)
    # - If victim < attacker: potentially bad (we might lose our piece)

    if victim_value >= attacker_value:
        # Winning or equal trade
        return victim_value - attacker_value + promotion_gain
    else:
        # We're capturing with a more valuable piece
        # Pessimistic assumption: we might lose our attacker
        # But we do win the victim first
        # Net: victim - attacker (likely negative)
        return victim_value - attacker_value + promotion_gain


def see_ge(board: CachedBoard, move: chess.Move, threshold: int = 0) -> bool:
    """
    Check if SEE value of move is >= threshold.
    Used for SEE pruning - returns True if the capture is at least break-even.
    """
    return see(board, move) >= threshold


def see_ge_int(board: CachedBoard, move_int: int, threshold: int = 0) -> bool:
    """
    Check if SEE value of move is >= threshold (integer move version).
    """
    return see(board, int_to_move(move_int)) >= threshold


def should_stop_search(current_depth: int) -> bool:
    """
    Determine if search should stop.
    - TimeControl.stop_search (from UCI 'stop'): Always honored immediately
    - TimeControl.soft_stop (from time limit): Only honored after MIN_NEGAMAX_DEPTH

    Note: MIN_ACCEPTABLE_DEPTH enforcement is done in find_best_move
    to allow more nuanced control based on position type.
    """
    if TimeControl.stop_search:
        return True
    if TimeControl.soft_stop and current_depth > config.MIN_NEGAMAX_DEPTH:
        return True
    return False


def quiescence(board: CachedBoard, alpha: int, beta: int, q_depth: int,
               last_capture_sq: int = -1) -> Tuple[int, List[int]]:
    """
    Quiescence search with improved time management and move limits.

    Uses integer moves throughout for performance.

    FIX: Added last_capture_sq parameter to track capture sequences.
    When the last move was a capture, we MUST search at least one recapture
    on that square before allowing soft-stop. This prevents horizon effect
    where we stop right before an obvious recapture.

    Args:
        board: Current board position
        alpha: Alpha bound
        beta: Beta bound
        q_depth: Current quiescence search depth
        last_capture_sq: Square where the last capture occurred (-1 if none)

    Returns:
        Tuple of (score, pv_int) where pv_int is the list of integer moves in the principal variation.
    """
    global _qs_stats

    kpi['q_depth'] = max(kpi['q_depth'], q_depth)
    kpi['nodes'] += 1
    _qs_stats["total_nodes"] += 1
    _qs_stats["max_depth_reached"] = max(_qs_stats["max_depth_reached"], q_depth)

    # -------- AGGRESSIVE: Check time every QS_TIME_CHECK_INTERVAL nodes --------
    if _qs_stats["total_nodes"] % config.QS_TIME_CHECK_INTERVAL == 0:
        check_time()
        # NEW: Also check QS time budget
        if TimeControl.time_limit and TimeControl.start_time:
            elapsed = time.perf_counter() - TimeControl.start_time
            if elapsed > TimeControl.time_limit * config.QS_TIME_BUDGET_FRACTION:
                TimeControl.soft_stop = True
                _diag_warn("qs_budget_exceeded",
                           f"QS exceeded {config.QS_TIME_BUDGET_FRACTION * 100:.0f}% time budget at depth {q_depth}")

    # FIX V3: Always check time at QS entry when time is short
    if TimeControl.time_limit and TimeControl.time_limit < 1.0:
        check_time()  # Critical time - check every QS entry

    # Hard stop always honored immediately
    if TimeControl.stop_search:
        # Skip game_over check - we got here via legal move
        if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
            return evaluate_nn(board, skip_game_over=True), []
        else:
            return evaluate_classical(board, skip_game_over=True), []

    # -------- EARLIER soft stop in QS - honor at shallower depth --------
    # FIX: Don't soft-stop if we're in a capture sequence (last move was a capture)
    # This prevents horizon effect where we stop right before an obvious recapture
    in_capture_sequence = (last_capture_sq >= 0)
    if TimeControl.soft_stop and q_depth > round(config.MAX_QS_DEPTH // config.QS_SOFT_STOP_DIVISOR):
        if not in_capture_sequence:
            _qs_stats["time_cutoffs"] += 1
            _diag_warn("qs_time_cutoff", f"QS soft-stopped at depth {q_depth}")
            # Skip game_over check - we got here via legal move
            if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
                return evaluate_nn(board, skip_game_over=True), []
            else:
                return evaluate_classical(board, skip_game_over=True), []
        # If in capture sequence, continue but with tighter limits (handled below)

    # -------- Hard depth limit to prevent search explosion --------
    if q_depth > config.MAX_QS_DEPTH:
        # DIAG: Track QS depth limit hits
        _diag_warn("qs_depth_exceeded", f"QS hit depth {q_depth}, fen={board.fen()[:40]}")
        # Return static evaluation when QS goes too deep
        # Skip game_over check - we got here via legal move
        if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
            return evaluate_nn(board, skip_game_over=True), []
        else:
            return evaluate_classical(board, skip_game_over=True), []

    # -------- Draw detection --------
    if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
        return get_draw_score(board), []

    # -------- Always compute key for TT --------
    key = board.zobrist_hash()

    if config.QS_TT_SUPPORTED and key in qs_transposition_table:
        kpi['qs_tt_hits'] += 1
        stored_score = qs_transposition_table[key]
        if stored_score >= beta:
            return stored_score, []  # ✅ Return stored score, not beta

    # Defer is_game_over check - only needed for rare stalemate cases
    # We check it at the end if moves_searched == 0 and not in check
    is_check = board.is_check()
    best_pv = []
    best_score = -config.MAX_SCORE  # ✅ Track best score separately

    # Stand-pat is not valid when in check. It is likely that there will be a move that is better
    # than stand-pat
    if not is_check:
        is_dnn_eval = False
        if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MIN_NN_EVAL:
            stand_pat = evaluate_nn(board)
            is_dnn_eval = True
        else:
            stand_pat = evaluate_classical(board)

        # if classical evaluation of stand_pat is close to beta, use NN evaluation.
        if (not is_dnn_eval and config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL
                and abs(stand_pat) < config.STAND_PAT_MAX_NN_EVAL
                and abs(stand_pat - beta) < config.QS_DELTA_MAX_NN_EVAL):
            stand_pat = evaluate_nn(board)
            is_dnn_eval = True

        if stand_pat >= beta:
            kpi['beta_cutoffs'] += 1
            return stand_pat, []  # ✅ Return stand_pat, not beta

        # stand_pat too low - can't possibly reach alpha
        if stand_pat + PIECE_VALUES[chess.QUEEN] < alpha:
            return stand_pat, []  # ✅ Return stand_pat (it's the best we can do)

        # If stand_pat is going to be alpha or stan_pat and alpha are close, use NN evaluation.
        if (not is_dnn_eval and config.NN_ENABLED
                and q_depth <= config.QS_DEPTH_MAX_NN_EVAL
                and abs(stand_pat) < config.STAND_PAT_MAX_NN_EVAL
                and (stand_pat > alpha or abs(stand_pat - alpha) < config.QS_DELTA_MAX_NN_EVAL)):
            stand_pat = evaluate_nn(board)

        best_score = stand_pat  # ✅ Initialize best_score with stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    # -------- DYNAMIC move limit based on depth AND time pressure --------
    move_limit = None
    for i in range(len(config.MAX_QS_MOVES_DIVISOR)):
        if q_depth <= round(config.MAX_QS_DEPTH / config.MAX_QS_MOVES_DIVISOR[i]):
            move_limit = config.MAX_QS_MOVES[i]
            break
    if move_limit is None:
        move_limit = config.MAX_QS_MOVES[-1]

    time_critical = TimeControl.soft_stop or (
            TimeControl.time_limit and TimeControl.start_time and
            (time.perf_counter() - TimeControl.start_time) > TimeControl.time_limit * config.QS_TIME_CRITICAL_FACTOR
    )
    if time_critical:
        move_limit = min(config.MAX_QS_MOVES_TIME_CRITICAL, move_limit)  # Time critical (5 moves)

    # Pre-compute move info for this position (using integer version)
    board.precompute_move_info_int()

    moves_searched = 0
    # FIX: Get ordered moves, prioritizing recaptures on the last capture square
    all_moves = ordered_moves_q_search_int(board, last_capture_sq)

    for move_int in all_moves:
        # -------- NEW: Enforce move limit to prevent explosion --------
        # FIX: But always allow recaptures on the last capture square
        to_sq = (move_int >> 6) & 0x3F
        is_recapture = (0 <= last_capture_sq == to_sq)

        if not is_check and moves_searched >= move_limit:
            # Allow one recapture even past move limit
            if not is_recapture:
                _diag_warn("qs_move_limit", f"QS move limit ({move_limit}) at depth {q_depth}")
                break

        if not is_check:
            # Use cached is_capture (integer version - faster)
            is_capture_move = board.is_capture_int(move_int)
            # if the move is not a capture then if the move gives check and depth < TACTICAL_QS_MAX_DEPTH,
            # go ahead with exploration; otherwise skip the move
            if not is_capture_move:
                if q_depth > config.CHECK_QS_MAX_DEPTH:
                    continue
                if not board.gives_check_int(move_int):
                    continue

            # -------- Delta pruning (only when not in check) --------
            if (q_depth >= config.DELTA_PRUNING_QS_MIN_DEPTH and is_capture_move
                    and not board.gives_check_int(move_int)):
                victim_type = board.get_victim_type_int(move_int)
                if victim_type:
                    gain = PIECE_VALUES[victim_type]
                    if best_score + gain + config.DELTA_PRUNING_QS_MARGIN < alpha:  # ✅ Use best_score
                        continue

        # Compute is_capture_move if not already done (when in_check is True)
        if is_check:
            is_capture_move = board.is_capture_int(move_int)

        # Convert to chess.Move for push (needed by evaluator)
        move = int_to_move(move_int)
        should_update_nn = q_depth <= config.QS_DEPTH_MAX_NN_EVAL
        if should_update_nn:
            push_move(board, move, get_nn_evaluator())
        else:
            board.push(move)  # Only push board, not evaluator

        # FIX: Pass capture square to child if this move is a capture
        # This ensures recaptures are always searched
        child_capture_sq = to_sq if is_capture_move else -1
        score, child_pv = quiescence(board, -beta, -alpha, q_depth + 1, child_capture_sq)
        score = -score
        board.pop()

        if should_update_nn:
            get_nn_evaluator().pop()

        moves_searched += 1

        # FIX: Track if we've searched a recapture on the last capture square
        if in_capture_sequence and to_sq == last_capture_sq:
            in_capture_sequence = False  # We've resolved the capture, can soft-stop now

        # -------- IMPROVED: Check time more frequently --------
        if moves_searched % 5 == 0:
            check_time()
            if TimeControl.stop_search:
                break  # Exit loop gracefully, return best_score below
            # Also check soft_stop in deep QS - but not if we're still in capture sequence
            if TimeControl.soft_stop and q_depth > round(config.MAX_QS_DEPTH // config.QS_SOFT_STOP_DIVISOR):
                if not in_capture_sequence:
                    _qs_stats["time_cutoffs"] += 1
                    break

        if score > best_score:  # ✅ Track best_score
            best_score = score
            best_pv = [move_int] + child_pv  # Store integer move in PV

        if score >= beta:
            kpi['beta_cutoffs'] += 1
            if config.QS_TT_SUPPORTED:
                qs_transposition_table[key] = score
            return score, best_pv  # ✅ Return score, not beta

        if score > alpha:
            alpha = score

    # -------- No moves searched = check for checkmate/stalemate --------
    if moves_searched == 0:
        if is_check:
            return -config.MAX_SCORE + board.ply(), []  # Checkmate
        # Deferred stalemate check - only when not in check and no moves
        # This is rare, so we save the is_game_over call for most nodes
        if board.is_game_over():
            return 0, []  # Stalemate

    if config.QS_TT_SUPPORTED:
        qs_transposition_table[key] = best_score  # ✅ Store best_score

    return best_score, best_pv  # ✅ Return best_score, not alpha


def get_draw_score(board: CachedBoard) -> int:
    """
    Return score for draw positions with capped contempt.

    Uses -material (proportional contempt) but capped to avoid situations
    where we'd rather lose material than draw.

    When winning: contempt ranges from 0 to -300 (draw is bad)
    When losing: contempt ranges from 0 to +150 (draw is good)
    """
    material = board.material_evaluation()

    if material > 0:
        return max(-300, -material)
    elif material < 0:
        return min(150, -material)
    return 0


def negamax(board: CachedBoard, depth: int, alpha: int, beta: int, allow_singular: bool = True) -> Tuple[
    int, List[int]]:
    """
    Negamax search with alpha-beta pruning.

    Uses integer moves throughout for performance.

    Returns:
        Tuple of (score, pv_int) where pv_int is the list of integer moves in the principal variation.
    """
    kpi['nodes'] += 1

    check_time()
    # Only check hard stop here - soft_stop is handled at root level
    # This ensures current depth completes before stopping
    if TimeControl.stop_search:
        # Skip game_over check - we got here via legal move
        return evaluate_classical(board, skip_game_over=True), []

    # -------- Draw detection --------
    if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
        return get_draw_score(board), []

    key = board.zobrist_hash()
    alpha_orig = alpha
    beta_orig = beta  # ✅ FIX 1

    best_move_int = 0  # Integer move
    best_pv = []
    max_eval = -config.MAX_SCORE

    # -------- TT Lookup --------
    entry = transposition_table.get(key)
    tt_move_int = 0  # Integer TT move
    if entry and entry.depth >= depth:
        kpi['tt_hits'] += 1
        if entry.flag == TT_EXACT:
            return entry.score, extract_pv_from_tt_int(board, depth)
        elif entry.flag == TT_LOWER_BOUND:
            alpha = max(alpha, entry.score)
        elif entry.flag == TT_UPPER_BOUND:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score, []  # ✅ FIX 2 (no PV on cutoff)

        tt_move_int = entry.best_move_int
        # DIAG: Verify TT move is legal (corruption detection)
        if tt_move_int != 0:
            legal_moves = board.get_legal_moves_int()
            if tt_move_int not in legal_moves:
                _diag_warn("tt_illegal_moves", f"TT move {tt_move_int} illegal in {board.fen()[:50]}")
                tt_move_int = 0  # Ignore corrupted TT move

    # -------- Quiescence if depth == 0 --------
    if depth == 0:
        return quiescence(board, alpha, beta, 1)

    in_check = board.is_check()

    # -------- Razoring (drop into quiescence when far below alpha) --------
    # At low depths, if we're way below alpha, just go to quiescence
    if (config.RAZORING_ENABLED
            and not in_check
            and depth <= config.RAZORING_MAX_DEPTH
            and depth >= 1):
        # Get static eval (use material for speed)
        static_eval = evaluate_classical(board)
        margin = config.RAZORING_MARGIN[depth] if depth < len(config.RAZORING_MARGIN) else config.RAZORING_MARGIN[-1]

        if static_eval + margin <= alpha:
            # Position is so bad that even with a margin, we're below alpha
            # Drop into quiescence to see if tactics can save us
            qs_score, qs_pv = quiescence(board, alpha, beta, 1)
            if qs_score <= alpha:
                kpi['razoring_prunes'] += 1
                return qs_score, qs_pv

    # -------- Null Move Pruning (not when in check or in zugzwang-prone positions) --------
    if (depth >= config.NULL_MOVE_MIN_DEPTH
            and not in_check
            # and not on_expected_pv  # Don't null-move on PV
            and board.has_non_pawn_material()
            and board.occupied.bit_count() > 6):
        push_move(board, chess.Move.null(), get_nn_evaluator())
        score, _ = negamax(board, depth - 1 - config.NULL_MOVE_REDUCTION, -beta, -beta + 1, allow_singular=False)
        score = -score
        board.pop()
        get_nn_evaluator().pop()
        if score >= beta:
            return beta, []

    # -------- Singular Extension Check (simplified) --------
    singular_extension_applicable = False
    singular_move_int = 0  # Integer move

    if (allow_singular
            and depth >= 6
            and tt_move_int != 0
            and not in_check
            and key in transposition_table
            and transposition_table[key].flag != TT_UPPER_BOUND
            and transposition_table[key].depth >= depth - 3):

        reduced_depth = max(1, depth // 2 - 1)
        reduced_beta = transposition_table[key].score - config.SINGULAR_MARGIN

        # Only search a few top moves, not all
        move_count = 0
        highest_score = -config.MAX_SCORE

        for move_int in ordered_moves_int(board, depth, tt_move_int=tt_move_int):
            if move_int == tt_move_int:
                continue
            if move_count >= 3:  # Limit moves checked
                break

            move = int_to_move(move_int)
            push_move(board, move, get_nn_evaluator())
            score, _ = negamax(board, reduced_depth, -reduced_beta - 1, -reduced_beta, allow_singular=False)
            score = -score
            board.pop()
            get_nn_evaluator().pop()
            move_count += 1

            highest_score = max(highest_score, score)
            if highest_score >= reduced_beta:
                break

        if highest_score < reduced_beta:
            singular_extension_applicable = True
            singular_move_int = tt_move_int

    # -------- Move Ordering --------
    moves_int = ordered_moves_int(board, depth, tt_move_int=tt_move_int)
    if not moves_int:
        if in_check:
            return -config.MAX_SCORE + board.ply(), []
        return 0, []

    # -------- Futility Pruning Setup --------
    # Compute static eval once for futility pruning (only needed at low depths)
    futility_pruning_applicable = False
    static_eval = None
    if (config.FUTILITY_PRUNING_ENABLED
            and not in_check
            and depth <= config.FUTILITY_MAX_DEPTH
            and depth >= 1
            and abs(alpha) < config.MAX_SCORE - 100):  # Not near mate scores
        static_eval = evaluate_classical(board)
        futility_margin = config.FUTILITY_MARGIN[depth] if depth < len(config.FUTILITY_MARGIN) else config.FUTILITY_MARGIN[-1]
        # If static eval + margin is still below alpha, futility pruning may apply
        if static_eval + futility_margin <= alpha:
            futility_pruning_applicable = True

    for move_index, move_int in enumerate(moves_int):
        # FIX V3: Check time at start of each move iteration for faster response
        if move_index > 0 and move_index % 3 == 0:  # Check every 3rd move
            check_time()
            if TimeControl.stop_search:
                break  # Hard stop - exit immediately

        # Use cached move info (integer version - faster)
        is_capture = board.is_capture_int(move_int)
        gives_check = board.gives_check_int(move_int)

        # -------- SEE Pruning (prune losing captures at low depths) --------
        if (config.SEE_PRUNING_ENABLED
                and depth <= config.SEE_PRUNING_MAX_DEPTH
                and is_capture
                and not in_check
                and move_int != tt_move_int
                and move_index > 0):  # Don't prune the first move
            # Prune captures with negative SEE at low depths
            # Threshold increases with depth (more aggressive pruning at lower depths)
            see_threshold = -20 * depth  # e.g., -20 at depth 1, -40 at depth 2, etc.
            if not see_ge_int(board, move_int, see_threshold):
                kpi['see_prunes'] += 1
                continue  # Skip this losing capture

        # -------- Futility Pruning (skip quiet moves when position is hopeless) --------
        if (futility_pruning_applicable
                and not is_capture
                and not gives_check
                and move_int != tt_move_int
                and move_index > 0):  # Don't prune the first move or TT move
            # Check if the move is not a killer move (use integer comparison)
            is_killer = False
            if depth is not None and 0 <= depth < len(killer_moves_int):
                is_killer = (move_int == killer_moves_int[depth][0] or move_int == killer_moves_int[depth][1])

            if not is_killer:
                # This quiet move is unlikely to raise the score above alpha
                kpi['futility_prunes'] += 1
                continue

        # Convert to chess.Move for push (needed by evaluator)
        move = int_to_move(move_int)
        push_move(board, move, get_nn_evaluator())
        child_in_check = board.is_check()

        # -------- Extensions --------
        extension = 0
        if singular_extension_applicable and move_int == singular_move_int:
            extension = config.SINGULAR_EXTENSION
        elif child_in_check:
            extension = 1

        new_depth = depth - 1 + extension

        # -------- LMR --------
        # Use integer killer moves for comparison
        if depth is not None and 0 <= depth < len(killer_moves_int):
            km0_int = killer_moves_int[depth][0]
            km1_int = killer_moves_int[depth][1]
        else:
            km0_int = 0
            km1_int = 0

        reduce = (
                depth >= config.LMR_MIN_DEPTH
                and move_index >= config.LMR_MOVE_THRESHOLD
                and not child_in_check
                and not is_capture
                and not gives_check
                and move_int != km0_int
                and move_int != km1_int
                and extension == 0
        )

        if reduce:
            reduction = 1
            if depth >= 6 and move_index >= 6:
                reduction = 2
            score, child_pv = negamax(board, new_depth - reduction, -alpha - 1, -alpha, allow_singular=True)
            score = -score
            if score > alpha:
                score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                score = -score
        else:
            if move_index > 0:
                score, child_pv = negamax(board, new_depth, -alpha - 1, -alpha, allow_singular=True)
                score = -score
                if alpha < score < beta:
                    score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                    score = -score
            else:
                score, child_pv = negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                score = -score

        board.pop()
        get_nn_evaluator().pop()

        if score > max_eval:
            max_eval = score
            best_move_int = move_int
            best_pv = [move_int] + child_pv  # Store integer move in PV

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Update killer moves and history for quiet moves
            # Use integer killer moves storage
            if not is_capture and depth is not None and 0 <= depth < len(killer_moves_int):
                if killer_moves_int[depth][0] != move_int:
                    killer_moves_int[depth][1] = killer_moves_int[depth][0]
                    killer_moves_int[depth][0] = move_int
                # Array-based history heuristic
                if move_int < HISTORY_TABLE_SIZE:
                    history_heuristic[move_int] = min(
                        history_heuristic[move_int] + depth * depth, 10_000
                    )
            kpi['beta_cutoffs'] += 1
            break

    # -------- Store in TT --------
    if max_eval <= alpha_orig:
        flag = TT_UPPER_BOUND
    elif max_eval >= beta_orig:  # ✅ FIX 1
        flag = TT_LOWER_BOUND
    else:
        flag = TT_EXACT

    old = transposition_table.get(key)
    if old is None or depth >= old.depth:
        transposition_table[key] = TTEntry(depth, max_eval, flag, best_move_int)

    return max_eval, best_pv


def extract_pv_from_tt_int(board: CachedBoard, max_depth: int) -> List[int]:
    """
    Extract the principal variation from the transposition table as integer moves.

    Returns integer moves instead of chess.Move objects.

    Args:
        board: Current board position
        max_depth: Maximum depth to extract

    Returns:
        List of integer moves forming the PV
    """
    pv = []
    seen_keys = set()

    for _ in range(max_depth):
        key = board.zobrist_hash()

        # Prevent infinite loops from repetitions
        if key in seen_keys:
            break
        seen_keys.add(key)

        entry = transposition_table.get(key)
        if entry is None or entry.best_move_int == 0:
            break

        move_int = entry.best_move_int
        legal_moves = board.get_legal_moves_int()
        if move_int not in legal_moves:
            # DIAG: Track when TT contains illegal PV move
            _diag_warn("pv_illegal_moves", f"PV move {move_int} illegal at depth {len(pv)}")
            break

        pv.append(move_int)
        move = int_to_move(move_int)
        push_move(board, move, get_nn_evaluator())

    # Restore board state
    for _ in range(len(pv)):
        board.pop()
        get_nn_evaluator().pop()

    return pv


def extract_pv_from_tt(board: CachedBoard, max_depth: int) -> List[chess.Move]:
    """
    Extract the principal variation from the transposition table.

    Legacy wrapper - converts integer PV to chess.Move objects.

    Args:
        board: Current board position
        max_depth: Maximum depth to extract

    Returns:
        List of moves forming the PV
    """
    pv_int = extract_pv_from_tt_int(board, max_depth)
    return pv_int_to_moves(pv_int)


def validate_pv_int(board: CachedBoard, pv_int: List[int]) -> List[int]:
    """
    Validate a principal variation by replaying moves and truncating at the first illegal move.

    This guards against TT race conditions in Lazy SMP where a move stored by one thread
    for a different position can leak into another thread's PV via a shared transposition
    table entry, producing a move that is illegal in the PV's actual position.

    Args:
        board: Current root board position (not modified).
        pv_int: List of integer moves forming the PV.

    Returns:
        Validated PV truncated at the first illegal move.
    """
    if not pv_int:
        return pv_int

    validated = []
    pushed = 0
    try:
        for move_int in pv_int:
            if move_int == 0:
                break
            move = int_to_move(move_int)
            if move not in board.get_legal_moves_list():
                _diag_warn("pv_validation_truncated",
                           f"PV validation: move {move.uci()} illegal at ply {pushed} in {board.fen()[:50]}")
                break
            validated.append(move_int)
            board.push(move)
            pushed += 1
    finally:
        # Restore board state
        for _ in range(pushed):
            board.pop()

    return validated


def age_heuristic_history():
    """Vectorized history aging using numpy."""
    global history_heuristic
    history_heuristic = (history_heuristic * 3) // 4


def control_dict_size(table, max_dict_size):
    """
    Control dictionary size by removing oldest entries.
    OPTIMIZED: Instead of popping one at a time (O(n) for each pop),
    we rebuild the dict with only the most recent entries.
    """
    current_size = len(table)
    if current_size > max_dict_size:
        # Calculate how many entries to keep
        entries_to_keep = max_dict_size * 3 // 4  # Keep 75%

        # For a regular dict (Python 3.7+), iteration order is insertion order
        # So we want to keep the LAST entries_to_keep items
        entries_to_remove = current_size - entries_to_keep

        # Much faster: get keys to remove, then delete them
        # Using islice to get first N keys (oldest entries)
        from itertools import islice
        keys_to_remove = list(islice(table.keys(), entries_to_remove))

        for key in keys_to_remove:
            del table[key]


def pv_to_san(board: CachedBoard, pv: List[chess.Move]) -> str:
    """
    Convert a PV (list of moves) to SAN notation string.

    Args:
        board: Starting position
        pv: List of moves

    Returns:
        SAN string representation of the PV
    """
    san_moves = []
    temp_board = board.copy(stack=False)

    for i, move in enumerate(pv):
        if temp_board.turn == chess.WHITE:
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}.")
        elif i == 0:
            # Black to move at start
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}...")

        san_moves.append(temp_board.san(move))
        temp_board.push(move)  # Only push to board, don't modify nn_evaluator

    return " ".join(san_moves)


def find_best_move(fen, max_depth=config.MAX_NEGAMAX_DEPTH, time_limit=None, clear_tt=True,
                   expected_best_moves=None, use_existing_time_control=False) -> \
        Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Finds the best move for a given FEN using iterative deepening negamax with alpha-beta pruning,
    aspiration windows, TT, quiescence, null-move pruning, LMR, singular extensions, and heuristics.

    Args:
        fen: FEN string of the position
        max_depth: Maximum search depth
        time_limit: Time limit in seconds (None for no limit)
        expected_best_moves: For testing - stop early if best move matches
        clear_tt: If True, clear transposition tables before search. Set to False
                  on ponderhit to benefit from warm TT.
        use_existing_time_control: If True, skip TimeControl initialization.
                  Used for ponder searches where TimeControl is set up by the
                  main thread (avoids race with ponderhit updates).

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _qs_stats

    # -------- Initialize time control --------
    if not use_existing_time_control:
        TimeControl.time_limit = time_limit
        TimeControl.stop_search = False
        TimeControl.soft_stop = False  # Reset soft stop
        TimeControl.start_time = time.perf_counter()
        # FIX V3: Set hard stop at 150% of time limit (but at least 0.5s grace period)
        if time_limit:
            grace_period = max(time_limit * 0.5, 0.3)  # At least 0.3s or 50% extra
            TimeControl.hard_stop_time = TimeControl.start_time + time_limit + grace_period
        else:
            TimeControl.hard_stop_time = None
    else:
        diag_print(f"TimeControl: using existing (time_limit={TimeControl.time_limit}, "
                   f"stop={TimeControl.stop_search}, soft={TimeControl.soft_stop})")
    nodes_start = kpi['nodes']
    nps = 0

    # -------- NEW: Reset QS statistics for this search --------
    _qs_stats = {
        "max_depth_reached": 0,
        "total_nodes": 0,
        "time_cutoffs": 0,
    }

    diag_print(f"TimeControl: stop={TimeControl.stop_search}, soft={TimeControl.soft_stop}, time_limit={time_limit}")

    # -------- Clear search tables & heuristics --------
    # -------- DIAGNOSTIC: Track initialization steps --------
    init_start = time.perf_counter()

    for i in range(len(killer_moves)):
        killer_moves[i] = [None, None]
    for i in range(len(killer_moves_int)):
        killer_moves_int[i] = [0, 0]  # Clear integer killer moves
    history_heuristic.fill(0)  # Numpy array clear

    # Only clear TT if requested (preserve on ponderhit for warm cache)
    if clear_tt:
        transposition_table.clear()
        qs_transposition_table.clear()

    # -------- CRITICAL: NN cache trimming can be slow if cache is huge --------
    cache_size_before = len(dnn_eval_cache)
    cache_trim_start = time.perf_counter()
    control_dict_size(dnn_eval_cache, config.MAX_TABLE_SIZE)
    cache_trim_time = time.perf_counter() - cache_trim_start
    cache_size_after = len(dnn_eval_cache)

    if cache_trim_time > 0.1:
        diag_print(f"DEBUG: NN cache trim took {cache_trim_time:.3f}s "
                   f"({cache_size_before} -> {cache_size_after} entries)")

    tables_cleared_time = time.perf_counter() - init_start
    if tables_cleared_time > 0.1:
        diag_print(f"DEBUG: Tables cleared in {tables_cleared_time:.3f}s")

    # -------- Initialize board and evaluator --------
    board_init_start = time.perf_counter()
    board = CachedBoard(fen)
    board_init_time = time.perf_counter() - board_init_start
    if board_init_time > 0.1:
        diag_print(f"DEBUG: Board init took {board_init_time:.3f}s")

    # -------- CRITICAL: Check for stop before NN reset (which can be slow) --------
    # Skip this check during ponder search — we want the NN loaded so the search
    # can do useful work even if ponderhit/stop arrives shortly after.
    # On ponder miss, GUI ignores bestmove anyway, so a small delay is acceptable.
    if TimeControl.stop_search and not TimeControl.is_ponder_search:
        diag_print(f"DEBUG: Stop received before NN reset, aborting")
        # Return first legal move as fallback
        legal = board.get_legal_moves_list()
        if legal:
            return legal[0], 0, [legal[0]], 0, 0
        return chess.Move.null(), 0, [], 0, 0

    nn_reset_start = time.perf_counter()
    get_nn_evaluator().reset(board)
    nn_reset_time = time.perf_counter() - nn_reset_start
    if nn_reset_time > 0.1:
        diag_print(f"DEBUG: NN reset took {nn_reset_time:.3f}s")

    total_init_time = time.perf_counter() - init_start
    if total_init_time > 0.2:
        diag_print(f"DEBUG: Total init took {total_init_time:.3f}s")

    # FIX V3: If time is critically low, do minimal search and return quickly
    if time_limit and time_limit < 0.15:
        _diag_warn("critical_time_search", f"time_limit={time_limit:.3f}s, doing emergency search")
        diag_print(f"CRITICAL_TIME: time_limit={time_limit:.3f}s, doing emergency search")
        # Do a quick depth 1 search and return
        legal_moves = board.get_legal_moves_list()
        if legal_moves:
            # Just evaluate all moves at depth 1 and pick best
            best_move = legal_moves[0]
            best_score = -config.MAX_SCORE
            for move in legal_moves[:10]:  # Only look at first 10 moves
                push_move(board, move, get_nn_evaluator())
                score = -evaluate_nn(board) if config.NN_ENABLED else -evaluate_classical(board)
                board.pop()
                get_nn_evaluator().pop()
                if score > best_score:
                    best_score = score
                    best_move = move
            elapsed = time.perf_counter() - TimeControl.start_time
            nps = int(10 / elapsed) if elapsed > 0 else 0
            return best_move, best_score, [best_move], 10, nps
        return chess.Move.null(), 0, [], 0, 0

    # Start with no result - fallback computed lazily only if needed
    best_move_int = 0  # Integer move
    best_score = 0
    best_pv = []  # List of integer moves
    prev_depth_score = None  # For score stability tracking

    last_depth_time = 0.0
    max_completed_depth = 0  # Track highest completed depth

    # -------- Track if position appears tactical (large score swings) --------
    is_tactical_position = False
    score_unstable = False  # FIX V4: Track if we've seen score instability

    for depth in range(1, max_depth + 1):
        depth_start_time = time.perf_counter()
        check_time()

        # -------- Debug output for search progress --------
        if depth <= 3:
            diag_print(f"DEBUG: Starting depth {depth}, stop={TimeControl.stop_search}, soft={TimeControl.soft_stop}")

        # FIX V4: Determine minimum required depth based on position type and stability
        if score_unstable:
            current_min_depth = config.UNSTABLE_MIN_DEPTH  # Highest minimum for unstable scores
            if depth == config.UNSTABLE_MIN_DEPTH:
                _diag["unstable_min_depth"] += 1
        elif is_tactical_position:
            current_min_depth = config.TACTICAL_MIN_DEPTH  # Higher minimum for tactical
        else:
            current_min_depth = config.MIN_PREFERRED_DEPTH  # Normal minimum

        # Check if we should stop (but respect minimum depths)
        if depth > current_min_depth and should_stop_search(depth):
            diag_print(f"DEBUG: Stopping at depth {depth} due to should_stop_search (min={current_min_depth})")
            break

        # NEW: If we haven't reached minimum acceptable depth, clear soft_stop to force deeper search
        if depth <= current_min_depth and TimeControl.soft_stop and not TimeControl.stop_search:
            if max_completed_depth > 0 and max_completed_depth < config.MIN_PREFERRED_DEPTH:
                TimeControl.soft_stop = False
                _diag["min_depth_forced"] += 1
                diag_print(
                    f"MIN_DEPTH: Forcing depth {depth}, only completed {max_completed_depth}, need {config.MIN_PREFERRED_DEPTH}")

        # After MIN_DEPTH, check if we have enough time to likely complete the next depth
        if depth > config.MIN_NEGAMAX_DEPTH and time_limit is not None and last_depth_time > 0:
            elapsed = time.perf_counter() - TimeControl.start_time
            remaining = time_limit - elapsed
            estimated_next_depth_time = last_depth_time * config.ESTIMATED_BRANCHING_FACTOR

            # Only apply time estimation if we've reached minimum acceptable depth
            if max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                if remaining < estimated_next_depth_time * config.TIME_SAFETY_MARGIN_RATIO:
                    # Not enough time to likely complete this depth
                    break

            # FIX V4: Emergency reserve - stop if we're eating into the reserve
            if remaining < config.EMERGENCY_TIME_RESERVE and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                _diag["emergency_reserve_stop"] += 1
                diag_print(f"EMERGENCY_RESERVE: Only {remaining:.2f}s left, stopping at depth {max_completed_depth}")
                break

        age_heuristic_history()

        # -------- Root TT move --------
        root_key = board.zobrist_hash()
        entry = transposition_table.get(root_key)
        tt_move_int = 0
        if entry and entry.best_move_int != 0:
            legal_int = board.get_legal_moves_int()
            if entry.best_move_int in legal_int:
                tt_move_int = entry.best_move_int

        # -------- Aspiration window --------
        window = config.ASPIRATION_WINDOW
        retries = 0
        depth_completed = False  # Track if this depth completed successfully
        search_aborted = False  # Track if search was aborted mid-depth

        # -------- NEW: Skip aspiration window for tactical positions or when score is extreme --------
        use_full_window = (depth == 1 or is_tactical_position or
                           (prev_depth_score is not None and abs(prev_depth_score) > 500))
        if use_full_window and depth > 1:
            _diag_warn("aw_tactical_skip",
                       f"Skipping AW at depth {depth}, tactical={is_tactical_position}, score={prev_depth_score}")

        while not search_aborted:
            # First iteration or tactical positions use full window
            if use_full_window:
                alpha = -config.MAX_SCORE
                beta = config.MAX_SCORE
            else:
                alpha = best_score - window
                beta = best_score + window
            alpha_orig = alpha

            current_best_score = -config.MAX_SCORE
            current_best_move_int = 0
            current_best_pv = []

            # Get PV move as integer
            pv_move_int = best_move_int if best_move_int != 0 else 0

            for move_index, move_int in enumerate(ordered_moves_int(board, depth, pv_move_int, tt_move_int)):
                check_time()

                # -------- NEW: Debug output for move progress at depth 1 --------
                if depth == 1 and move_index < 3:
                    diag_print(f"DEBUG: depth 1 move {move_index}: {move_int}, stop={TimeControl.stop_search}")

                # Check for stop before searching this move
                if TimeControl.stop_search:
                    search_aborted = True
                    break
                # FIX: Only honor soft_stop mid-depth after reaching MIN_ACCEPTABLE_DEPTH
                if TimeControl.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                    _diag_warn("mid_depth_abort",
                               f"Aborting depth {depth} mid-search (completed={max_completed_depth})")
                    search_aborted = True
                    break

                move = int_to_move(move_int)
                push_move(board, move, get_nn_evaluator())

                if is_draw_by_repetition(board):
                    score = get_draw_score(board)
                    child_pv = []
                else:
                    # Principal variation first, then late moves with PVS
                    if move_index == 0:
                        score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                        score = -score
                    else:
                        score, child_pv = negamax(board, depth - 1, -alpha - 1, -alpha, allow_singular=True)
                        score = -score
                        if score > alpha:
                            score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                            score = -score

                board.pop()
                get_nn_evaluator().pop()

                # Check if search was aborted during negamax
                # FIX: Only honor soft_stop after reaching MIN_ACCEPTABLE_DEPTH
                if TimeControl.stop_search or (TimeControl.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH):
                    search_aborted = True
                    # Still save this move's result if it's our only one
                    if current_best_move_int == 0:
                        current_best_move_int = move_int
                        current_best_score = score
                        current_best_pv = [move_int] + child_pv
                    break

                if score > current_best_score:
                    current_best_score = score
                    current_best_move_int = move_int
                    current_best_pv = [move_int] + child_pv

                if score > alpha:
                    alpha = score

                if alpha >= beta:
                    break

            # If search was aborted, only save partial result if we have NO completed result
            # FIXED: Don't overwrite a completed depth's result with an incomplete one
            if search_aborted:
                if best_move_int == 0 and current_best_move_int != 0:
                    best_move_int = current_best_move_int
                    best_score = current_best_score
                    best_pv = current_best_pv
                break

            # -------- SUCCESS: within aspiration window (or using full window) --------
            if use_full_window or (current_best_score > alpha_orig and current_best_score < beta):
                best_move_int = current_best_move_int
                best_score = current_best_score
                best_pv = current_best_pv
                depth_completed = True
                break

            # -------- FAIL-LOW or FAIL-HIGH: widen window --------
            window *= 2
            retries += 1

            # FIX V4: Use more retries for tactical positions
            max_retries = config.MAX_AW_RETRIES_TACTICAL if is_tactical_position else config.MAX_AW_RETRIES

            # DIAG: Track excessive aspiration retries
            if retries >= max_retries:
                _diag_warn("aspiration_retries",
                           f"depth={depth} hit max retries ({max_retries}), score={current_best_score}")
                # -------- NEW: Mark position as tactical for future depths --------
                is_tactical_position = True

            # Save partial result ONLY if we have NO completed result yet
            # FIXED: Don't overwrite a completed depth's result with an incomplete aspiration retry
            if best_move_int == 0 and current_best_move_int != 0:
                best_move_int = current_best_move_int
                best_score = current_best_score
                best_pv = current_best_pv

            # -------- FALLBACK: full window search --------
            if retries >= max_retries:
                alpha = -config.MAX_SCORE
                beta = config.MAX_SCORE
                current_best_score = -config.MAX_SCORE

                for move_int in ordered_moves_int(board, depth, pv_move_int, tt_move_int):
                    check_time()

                    # Check for stop
                    if TimeControl.stop_search:
                        search_aborted = True
                        break
                    # FIX: Only honor soft_stop after reaching MIN_ACCEPTABLE_DEPTH
                    if TimeControl.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                        search_aborted = True
                        break

                    move = int_to_move(move_int)
                    push_move(board, move, get_nn_evaluator())
                    score, child_pv = negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                    score = -score
                    board.pop()
                    get_nn_evaluator().pop()

                    # Check if search was aborted during negamax
                    # FIX: Only honor soft_stop after reaching MIN_ACCEPTABLE_DEPTH
                    if TimeControl.stop_search or (
                            TimeControl.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH):
                        search_aborted = True
                        if current_best_move_int == 0:
                            current_best_move_int = move_int
                            current_best_score = score
                            current_best_pv = [move_int] + child_pv
                        break

                    if score > current_best_score:
                        current_best_score = score
                        current_best_move_int = move_int
                        current_best_pv = [move_int] + child_pv

                    if score > alpha:
                        alpha = score

                # Save results from fallback search ONLY if completed (not aborted)
                # Or if we have no result yet (best_move_int is 0)
                if not search_aborted:
                    if current_best_move_int != 0:
                        best_move_int = current_best_move_int
                        best_score = current_best_score
                        best_pv = current_best_pv
                    depth_completed = True
                elif best_move_int == 0 and current_best_move_int != 0:
                    # Aborted but no previous result - save partial
                    best_move_int = current_best_move_int
                    best_score = current_best_score
                    best_pv = current_best_pv
                break

        # Record time taken for this depth (only if completed)
        if depth_completed:
            last_depth_time = time.perf_counter() - depth_start_time
            max_completed_depth = depth  # Track highest completed depth

            # Check for score instability (large swings between depths)
            if prev_depth_score is not None and depth > 2:
                score_diff = abs(best_score - prev_depth_score)
                if score_diff > _SCORE_INSTABILITY_THRESHOLD:
                    _diag_warn("score_instability",
                               f"depth {depth}: {prev_depth_score} -> {best_score} (diff={score_diff})")
                    # FIX V4: Mark as unstable and tactical, force deeper search
                    score_unstable = True  # FIX V4: Track instability
                    if not is_tactical_position:
                        is_tactical_position = True
                        _diag["tactical_extension"] += 1
                        # Clear soft_stop to force at least one more depth
                        if TimeControl.soft_stop and not TimeControl.stop_search:
                            TimeControl.soft_stop = False
                            diag_print(
                                f"TACTICAL_EXTENSION: Score swing {score_diff}cp at depth {depth}, forcing min depth {config.UNSTABLE_MIN_DEPTH}")
            prev_depth_score = best_score

        # Break out of depth loop if search was aborted
        if search_aborted:
            break

        # Print progress with PV only if depth completed
        if depth_completed and best_pv:
            elapsed = time.perf_counter() - TimeControl.start_time
            nps = int((kpi['nodes'] - nodes_start) / elapsed) if elapsed > 0 else 0
            validated_pv = validate_pv_int(board, best_pv)
            pv_uci = ' '.join(int_to_move(m).uci() for m in validated_pv)
            print(
                f"info depth {depth} score cp {best_score} nodes {kpi['nodes']} nps {nps} pv {pv_uci}",
                flush=True)

        # Early break to speed up testing
        if best_move_int != 0 and expected_best_moves is not None:
            best_move = int_to_move(best_move_int)
            if best_move in expected_best_moves:
                break

    # -------- SHALLOW SEARCH DIAGNOSTICS --------
    _diag["bestmove_depth_sum"] += max_completed_depth
    _diag["bestmove_count"] += 1

    if 3 >= max_completed_depth > 0:
        _diag["shallow_search_total"] += 1
        if max_completed_depth == 2:
            _diag["shallow_search_d2"] += 1
        elif max_completed_depth == 3:
            _diag["shallow_search_d3"] += 1
        best_move_for_diag = int_to_move(best_move_int) if best_move_int != 0 else None
        diag_print(f"SHALLOW_SEARCH: bestmove selected at depth {max_completed_depth} "
                   f"(tactical={is_tactical_position}, move={best_move_for_diag})")

    # Print QS statistics if significant
    if _qs_stats["max_depth_reached"] > config.MAX_QS_DEPTH // 2 or _qs_stats["time_cutoffs"] > 0:
        diag_print(f"QS_STATS: max_depth={_qs_stats['max_depth_reached']}, "
                   f"nodes={_qs_stats['total_nodes']}, time_cutoffs={_qs_stats['time_cutoffs']}")

    # -------- IMPROVED fallback: shallow tactical search instead of pure NN eval --------
    if best_move_int == 0:
        # DIAG: Track when fallback is needed
        _diag_warn("best_move_none", f"No depth completed, using shallow search fallback, fen={fen[:40]}")
        diag_print(f"Computing shallow search fallback (no depth completed)...")

        # -------- NEW: Debug output to understand why no depth completed --------
        elapsed = time.perf_counter() - TimeControl.start_time if TimeControl.start_time else 0
        diag_print(f"DEBUG fallback: elapsed={elapsed:.2f}s, stop={TimeControl.stop_search}, "
                   f"soft={TimeControl.soft_stop}, time_limit={time_limit}")

        # Reset to clean state for fallback evaluation
        board = CachedBoard(fen)
        get_nn_evaluator().reset(board)
        legal = board.get_legal_moves_list()

        if legal:
            best_move_int = move_to_int(legal[0])  # Default to first legal move
            best_score = -config.MAX_SCORE

            # -------- NEW: Do a shallow 1-ply search with captures/checks --------
            # This provides some tactical awareness without calling quiescence
            _diag_warn("fallback_shallow_search", f"Shallow search with {len(legal)} moves")

            fallback_aborted = False
            for move in legal:
                # -------- CRITICAL: Check for stop signal --------
                if TimeControl.stop_search:
                    diag_print(f"Fallback aborted by stop signal")
                    fallback_aborted = True
                    break

                get_nn_evaluator().push_with_board(board, move)

                # Check for immediate game-ending conditions
                if board.is_checkmate():
                    # We just delivered checkmate!
                    score = config.MAX_SCORE - board.ply()
                    board.pop()
                    get_nn_evaluator().pop()
                    best_move_int = move_to_int(move)
                    best_score = score
                    best_pv = [best_move_int]
                    diag_print(f"Fallback found checkmate: {move.uci()}")
                    break

                if board.is_stalemate() or board.is_insufficient_material():
                    score = 0  # Draw
                else:
                    # Do a 1-ply tactical check: look at opponent's best response
                    # Only examine captures and checks to limit explosion
                    opp_best = config.MAX_SCORE  # From opponent's perspective (we want to minimize this)
                    opp_moves_checked = 0
                    max_opp_moves = 8  # Limit opponent moves to check

                    board.precompute_move_info_int()
                    for opp_move in board.get_legal_moves_list():
                        # -------- CRITICAL: Check for stop signal in inner loop --------
                        if TimeControl.stop_search:
                            break

                        # Use integer for fast lookups
                        opp_move_int = move_to_int(opp_move)

                        # Only consider captures, checks, or promotions
                        is_tactical = (board.is_capture_int(opp_move_int) or
                                       board.gives_check_int(opp_move_int) or
                                       opp_move.promotion is not None)

                        if not is_tactical and opp_moves_checked > 0:
                            continue

                        # ------- Use push_move helper which handles both board and nn_evaluator --------
                        push_move(board, opp_move, get_nn_evaluator())

                        # Check for mate threats
                        if board.is_checkmate():
                            opp_score = config.MAX_SCORE - board.ply()  # Opponent wins
                        else:
                            # Use NN eval for the resulting position (already pushed)
                            opp_score = -evaluate_nn(board)  # Negate because it's opponent's view

                        board.pop()
                        get_nn_evaluator().pop()
                        opp_best = min(opp_best, opp_score)
                        opp_moves_checked += 1

                        if opp_moves_checked >= max_opp_moves:
                            break

                    # If no tactical responses found, use static eval
                    if opp_moves_checked == 0:
                        score = -evaluate_nn(board)
                    else:
                        score = -opp_best  # Our score is negative of opponent's best

                board.pop()
                get_nn_evaluator().pop()

                if score > best_score:
                    best_score = score
                    best_move_int = move_to_int(move)

            best_pv = [best_move_int]
            if not fallback_aborted:
                diag_print(
                    f"Shallow fallback: {len(legal)} moves, best={int_to_move(best_move_int).uci()} score={best_score}cp")
        else:
            # No legal moves - game is over (checkmate or stalemate)
            best_move_int = 0  # Will return null move
            best_score = evaluate_classical(board)
            best_pv = []

    # DIAG: Check for time overrun (only when time limit was set)
    if time_limit is not None:
        elapsed = time.perf_counter() - TimeControl.start_time
        overrun = elapsed - time_limit
        if overrun > 0.5:  # More than 500ms over
            _diag_warn("time_overruns",
                       f"Search overran by {overrun:.2f}s (limit={time_limit:.2f}s, actual={elapsed:.2f}s)")

    # -------- NEW: Additional diagnostic for severe overruns --------
    if time_limit is not None:
        elapsed = time.perf_counter() - TimeControl.start_time
        if elapsed > time_limit * 3:  # More than 3x the time limit
            _diag_warn("time_critical_abort",
                       f"SEVERE overrun: {elapsed:.2f}s vs {time_limit:.2f}s limit, QS_max_depth={_qs_stats['max_depth_reached']}")

    # DIAG: Check score bounds
    if abs(best_score) > config.MAX_SCORE:
        _diag_warn("score_out_of_bounds", f"Score {best_score} exceeds MAX_SCORE {config.MAX_SCORE}")

    # Convert integer move and PV to chess.Move for return value
    # Validate PV to guard against Lazy SMP TT race conditions
    best_move = int_to_move(best_move_int) if best_move_int != 0 else chess.Move.null()
    best_pv_moves = pv_int_to_moves(validate_pv_int(board, best_pv))

    return best_move, best_score, best_pv_moves, kpi['nodes'] - nodes_start, nps


def push_move(board: CachedBoard, move, evaluator: NNEvaluator):
    """Push move on both evaluator and board using the unified interface."""
    evaluator.push_with_board(board, move)


def push_move_int(board: CachedBoard, move_int: int, evaluator: NNEvaluator):
    """Push integer move on both evaluator and board. Converts to chess.Move internally."""
    move = int_to_move(move_int)
    evaluator.push_with_board(board, move)


def tokenize_san_string(san_string: str) -> list[str]:
    tokens = san_string.strip().split()
    san_moves = []
    for tok in tokens:
        if re.match(r"^\d+\.+$", tok):
            continue
        if re.match(r"^\d+\.\.\.$", tok):
            continue
        san_moves.append(tok)
    return san_moves


def pv_from_san_string(fen: str, san_string: str) -> list[chess.Move]:
    board = CachedBoard(fen)
    pv = []
    for ply, san in enumerate(tokenize_san_string(san_string)):
        move = board.parse_san(san)
        pv.append(move)
        board.push(move)  # Only push to board, don't modify nn_evaluator
    return pv


def main():
    """
    Interactive loop to input FEN positions and get the best move and evaluation.
    Tracks KPIs and handles timeouts and interruptions gracefully.
    """

    while True:
        try:
            fen = input("FEN: ").strip()
            if fen.lower() == "exit":
                break
            if fen == "":
                print("Type 'exit' to quit")
                continue

            # Reset KPIs
            for key in kpi:
                kpi[key] = 0

            # Start timer
            start_time = time.perf_counter()
            move, score, pv, _, _ = find_best_move(fen, max_depth=20, time_limit=5)
            end_time = time.perf_counter()

            # Record cache sizes and time
            elapsed_time = end_time - start_time
            kpi.update({
                'tt_size': len(transposition_table),
                'dec_size': len(dnn_eval_cache),
                'time': round(elapsed_time, 2),
                'nps': int(kpi['nodes'] / elapsed_time) if elapsed_time > 0 else 0
            })

            # Print KPIs
            print("\n--- Search KPIs ---")
            for key, value in kpi.items():
                print(f"{key}: {value}")

            board = CachedBoard(fen)
            print("\nBest move:", move)
            print("Evaluation:", score)
            print("PV:", pv_to_san(board, pv))
            print("PV (UCI):", " ".join(m.uci() for m in pv))
            print("-------------------\n")

        except KeyboardInterrupt:
            response = input("\nKeyboardInterrupt detected. Type 'exit' to quit, Enter to continue: ").strip()
            if response.lower() == "exit":
                break
            print("Resuming...\n")
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            continue


if __name__ == '__main__':
    import random

    random.seed(42)  # Fixed seed for reproducible benchmarks
    main()