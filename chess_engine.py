import heapq
import re
import time
import traceback
import threading
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
import random # Do not delete
import config
from cached_board import (
    CachedBoard, int_to_tuple, int_to_uci, move_to_int_from_obj,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE, BLACK,
    PIECE_VALUES,
)
from nn_evaluator import NNUEEvaluator, NNEvaluator

ROOT_DIR = str(Path(__file__).resolve().parent)
MODEL_PATH = ROOT_DIR + '/model/nnue.pt'


def is_debug_enabled() -> bool:
    """Check if diagnostic output is enabled (either via IS_DIAGNOSTIC or UCI debug on)."""
    return config.DIAGNOSTIC or config.debug_mode


def set_debug_mode(enabled: bool):
    """Set debug mode from UCI command."""
    config.debug_mode = enabled


# ======================================================================
# Shared state (module globals) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â intentionally shared across threads
# ======================================================================

class TimeControl:
    time_limit = None  # in seconds
    start_time = None
    stop_search = False  # Set by UCI 'stop' command - always honored
    hard_stop_time = None  # Absolute time when search MUST stop (150% of time_limit)
    is_ponder_search = False  # True during ponder search - skip early abort before NN init


TTEntry = namedtuple("TTEntry", ["depth", "score", "flag", "best_move_int"])
TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND = 0, 1, 2


class ShardedDict:
    """
    Lock-reduced shared dictionary using power-of-two sharding.

    Designed for Python 3.14t free-threading.
    """

    def __init__(self, num_shards: int = 8):
        if num_shards <= 0 or (num_shards & (num_shards - 1)) != 0:
            raise ValueError("num_shards must be a power of two")

        self.num_shards = num_shards
        self.mask = num_shards - 1
        self.shards = [{} for _ in range(num_shards)]

    def _shard(self, key: int):
        return self.shards[key & self.mask]

    # --- dict-like interface ---

    def get(self, key, default=None):
        return self._shard(key).get(key, default)

    def __getitem__(self, key):
        return self._shard(key)[key]

    def __setitem__(self, key, value):
        self._shard(key)[key] = value

    def __delitem__(self, key):
        del self._shard(key)[key]

    def __contains__(self, key):
        return key in self._shard(key)

    def keys(self):
        for shard in self.shards:
            yield from shard.keys()

    def items(self):
        for shard in self.shards:
            yield from shard.items()

    def values(self):
        for shard in self.shards:
            yield from shard.values()

    def __iter__(self):
        return self.keys()

    def pop(self, key, default=None):
        return self._shard(key).pop(key, default)

    def clear(self):
        for shard in self.shards:
            shard.clear()

    def __len__(self):
        return sum(len(s) for s in self.shards)

    def __getattr__(self, name):
        raise AttributeError(f"ShardedDict missing attribute: {name}")

    # --- optional size control ---
    def trim(self, max_total_size: int):
        if len(self) <= max_total_size:
            return

        for i in range(self.num_shards):
            self.shards[i] = {}


class ShardedTT(ShardedDict):
    """
    Sharded Transposition Table with depth-aware replacement.
    """

    def store(self, key: int, entry):
        """
        Store unconditionally.
        """
        self._shard(key)[key] = entry

    def store_if_deeper(self, key: int, entry):
        """
        Store only if:
            - No existing entry
            - New entry depth >= old depth
        """
        shard = self._shard(key)
        old = shard.get(key)
        if old is None or entry.depth >= old.depth:
            shard[key] = entry


transposition_table = ShardedTT(config.NUM_SHARDS_TABLES)
qs_transposition_table = ShardedDict(config.NUM_SHARDS_TABLES)
nn_eval_cache = ShardedDict(config.NUM_SHARDS_TABLES)

# Track positions seen in the current game (cleared on ucinewgame)
game_position_history: dict[int, int] = {}  # zobrist_hash -> count

# Array-based history heuristic size
HISTORY_TABLE_SIZE = 25000


# ======================================================================
# Search generation ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â plain int, only main thread writes, workers read
# ======================================================================

class _SearchControl:
    """Lightweight stop signaling for Lazy SMP. No locks needed ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
    only the main thread writes, workers only read."""
    generation = 1


# ======================================================================
# NN evaluator pool for reuse across search threads
# ======================================================================

_nn_eval_pool: list = []
_nn_eval_pool_lock = threading.Lock()

# Main thread's nn_evaluator (also template for creating copies)
nn_evaluator: None | NNEvaluator = None


def configure_nn_type():
    global nn_evaluator
    nn_evaluator = NNUEEvaluator.create(CachedBoard(), "NNUE", MODEL_PATH)


configure_nn_type()


def get_nn_evaluator_from_pool() -> NNEvaluator:
    """Get an evaluator from the pool, or create a new one."""
    with _nn_eval_pool_lock:
        if _nn_eval_pool:
            return _nn_eval_pool.pop()
    return NNEvaluator.create(CachedBoard(), "NNUE", MODEL_PATH)


def return_nn_evaluator_to_pool(evaluator: NNEvaluator):
    """Return an evaluator to the pool for reuse."""
    with _nn_eval_pool_lock:
        _nn_eval_pool.append(evaluator)


# ======================================================================
# Module-level utility functions (no per-instance state)
# ======================================================================

def pv_int_to_uci(pv_int: List[int]) -> List[str]:
    """Convert integer PV to UCI string list for output."""
    return [int_to_uci(m) for m in pv_int if m != 0]


def clear_game_history():
    """Clear game position history (call on ucinewgame).
    Also resets diagnostic counters on the main engine."""
    game_position_history.clear()
    # Reset main engine's diag counters if it exists
    global _main_engine
    if _main_engine is not None:
        _main_engine.reset_diag()


def is_draw_by_repetition(board: CachedBoard) -> bool:
    """
    Check for threefold repetition combining game history and search path.
    """
    key = board.zobrist_hash()
    game_count = game_position_history.get(key, 0)

    if game_count >= 3:
        return True
    if game_count == 2:
        if board.is_repetition(2):
            return True
    if game_count == 1:
        if board.is_repetition(2):
            return True
    if game_count == 0:
        if board.is_repetition(3):
            return True
    return False


def get_draw_score(board: CachedBoard) -> int:
    """
    Return score for draw positions with capped contempt.
    """
    material = board.material_evaluation_full()
    if material > 0:
        return max(-300, -material)
    elif material < 0:
        return min(150, -material)
    return 0


def push_move_int(board: CachedBoard, move_int: int, evaluator: NNEvaluator):
    """Push integer move on both evaluator and board."""
    evaluator.push_with_board(board, move_int)


def see_int(board: CachedBoard, move_int: int) -> int:
    """Simplified Static Exchange Evaluation (SEE) for integer moves."""
    victim_type = board.get_victim_type_int(move_int)
    attacker_type = board.get_attacker_type_int(move_int)
    _, _, promo = int_to_tuple(move_int)

    if victim_type is None:
        if promo:
            return PIECE_VALUES.get(promo, 0) - PIECE_VALUES[PAWN]
        return 0

    victim_value = PIECE_VALUES.get(victim_type, 0)
    attacker_value = PIECE_VALUES.get(attacker_type, 0)

    promotion_gain = 0
    if promo:
        promotion_gain = PIECE_VALUES.get(promo, 0) - PIECE_VALUES[PAWN]

    return victim_value - attacker_value + promotion_gain


def see_ge_int(board: CachedBoard, move_int: int, threshold: int = 0) -> bool:
    """Check if SEE value of move is >= threshold (integer move version)."""
    return see_int(board, move_int) >= threshold


def move_score_q_search_int(board: CachedBoard, move_int: int) -> int:
    """Score a move for quiescence search using pre-computed MVV-LVA."""
    return board.get_mvv_lva_score_int(move_int)


def ordered_moves_q_search_int(board: CachedBoard, last_capture_sq: int = -1) -> List[int]:
    """Return legal moves for quiescence search as integers."""
    moves_int = board.get_legal_moves_int()
    board.precompute_move_info_int(moves_int)
    moves_with_scores = []

    for m in moves_int:
        score = move_score_q_search_int(board, m)
        to_sq = (m >> 6) & 0x3F
        if 0 <= last_capture_sq == to_sq:
            score += 100000
        moves_with_scores.append((score, m))

    top_moves = heapq.nlargest(config.MAX_QS_MOVES[0], moves_with_scores)
    return [m for _, m in top_moves]


def control_dict_size(table, max_dict_size):
    """Control dictionary size by removing oldest entries."""
    current_size = len(table)
    if current_size > max_dict_size:
        entries_to_keep = max_dict_size * 3 // 4
        entries_to_remove = current_size - entries_to_keep
        from itertools import islice
        keys_to_remove = list(islice(table.keys(), entries_to_remove))
        for key in keys_to_remove:
            del table[key]


def validate_pv_int(board: CachedBoard, pv_int: List[int]) -> List[int]:
    """Validate a PV by replaying moves and truncating at first illegal move."""
    if not pv_int:
        return pv_int

    validated = []
    pushed = 0
    try:
        for move_int in pv_int:
            if move_int == 0:
                break
            legal_moves = board.get_legal_moves_int()
            if move_int not in legal_moves:
                break
            validated.append(move_int)
            board.push(move_int)
            pushed += 1
    finally:
        for _ in range(pushed):
            board.pop()
    return validated


def pv_to_san(board: CachedBoard, pv_int: List[int]) -> str:
    """Convert a PV (list of integer moves) to SAN notation string."""
    san_moves = []
    temp_board = board.copy(stack=False)
    for i, move_int in enumerate(pv_int):
        if temp_board.turn == WHITE:
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}.")
        elif i == 0:
            move_num = temp_board.fullmove_number
            san_moves.append(f"{move_num}...")
        san_moves.append(temp_board.san(move_int))
        temp_board.push(move_int)
    return " ".join(san_moves)


def diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


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


def pv_from_san_string(fen: str, san_string: str) -> List[int]:
    """Parse SAN string into list of integer moves."""
    board = CachedBoard(fen)
    pv = []
    for ply, san in enumerate(tokenize_san_string(san_string)):
        move_int = board.parse_san(san)
        pv.append(move_int)
        board.push(move_int)
    return pv


# ======================================================================
# Diagnostic tracking - used by ChessEngine (per-instance) and
# by diag_summary (aggregated from main engine for UCI output)
# ======================================================================

_DIAG_WARN_THRESHOLD = 3
_DIAG_SAMPLE_RATE = 100
_SCORE_INSTABILITY_THRESHOLD = 200


def _make_diag_dict():
    """Create a fresh diagnostic counter dict."""
    return {
        "tt_illegal_moves": 0,
        "score_out_of_bounds": 0,
        "time_overruns": 0,
        "pv_illegal_moves": 0,
        "eval_drift": 0,
        "qs_depth_exceeded": 0,
        "aspiration_retries": 0,
        "best_move_none": 0,
        "score_instability": 0,
        "qs_time_cutoff": 0,
        "qs_move_limit": 0,
        "qs_shallow_protected": 0,
        "qs_budget_exceeded": 0,
        "fallback_shallow_search": 0,
        "aw_tactical_skip": 0,
        "time_critical_abort": 0,
        "shallow_search_d2": 0,
        "shallow_search_d3": 0,
        "shallow_search_total": 0,
        "tactical_extension": 0,
        "min_depth_forced": 0,
        "mid_depth_abort": 0,
        "critical_time_search": 0,
        "emergency_reserve_stop": 0,
        "unstable_min_depth": 0,
        "bestmove_depth_sum": 0,
        "bestmove_count": 0,
    }


def _make_kpi_dict():
    """Create a fresh KPI counter dict."""
    return {
        "nodes": 0,
        "pos_eval": 0,
        "nn_evals": 0,
        "beta_cutoffs": 0,
        "tt_hits": 0,
        "qs_tt_hits": 0,
        "dec_hits": 0,
        "q_depth": 0,
        "see_prunes": 0,
        "futility_prunes": 0,
        "razoring_prunes": 0,
    }


def _make_qs_stats():
    """Create a fresh QS statistics dict."""
    return {
        "max_depth_reached": 0,
        "total_nodes": 0,
        "time_cutoffs": 0,
    }

# ======================================================================
# ChessEngine class ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â per-instance state, no thread-local hacks
# ======================================================================

class ChessEngine:
    """
    Encapsulates all per-search state for a chess search thread.

    Each search thread (main or Lazy SMP worker) creates its own ChessEngine
    instance with isolated killer moves, history heuristic, KPI counters,
    soft_stop flag, and NN evaluator reference.

    Shared state (TT, nn_eval_cache, game_position_history, TimeControl)
    remains at module level ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â this is intentional for Lazy SMP communication.
    """

    def __init__(self, nn_eval: Optional[NNEvaluator] = None,
                 search_generation: Optional[object] = None,
                 generation: int = 0,
                 worker_id: int = 0):
        """
        Args:
            nn_eval: NN evaluator for this engine. If None, uses the global one
                     (suitable for main thread / single-threaded mode).
            search_generation: Reference to _SearchControl for Lazy SMP stop
                               signaling. None for main thread.
            generation: This search's generation number (for Lazy SMP).
            worker_id: Worker ID for Lazy SMP (0 = main/canonical worker).
        """
        # Per-instance heuristics
        self.killer_moves_int = [[0, 0] for _ in range(config.MAX_NEGAMAX_DEPTH + 1)]
        self.history_heuristic = np.zeros(HISTORY_TABLE_SIZE, dtype=np.int32)

        # Per-instance counters
        self.kpi = _make_kpi_dict()
        self._diag = _make_diag_dict()
        self._qs_stats = _make_qs_stats()

        # Per-instance stop flag
        self.soft_stop = False

        # NN evaluator (per-instance ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â each thread needs its own incremental state)
        self.nn_evaluator = nn_eval if nn_eval is not None else nn_evaluator

        # Lazy SMP stop signaling
        self._search_generation = search_generation  # Reference to _SearchControl
        self._generation = generation

        # Lazy SMP worker identity (0 = main worker, uses optimal ordering)
        self._worker_id = worker_id

        # Track highest completed depth (for result reporting)
        self.completed_depth = 0
        self._rng = random.Random()
        # Per-instance random for evaluation noise (Lazy SMP diversity)
        self._eval_rng = random.Random()
        self._eval_noise = 5  # ±5 centipawns noise range
        self._noise_table = None  # Pre-computed noise lookup table
        self._move_order_salt = 0  # Worker-specific salt for deterministic move ordering

        # Persistent board instance - reused across searches to avoid
        # allocation overhead and keep cache pool warm
        self._board: Optional[CachedBoard] = None

    def reset_for_search(self):
        """Reset per-search state. Call before each new search."""
        for i in range(len(self.killer_moves_int)):
            self.killer_moves_int[i] = [0, 0]
        self.history_heuristic.fill(0)
        self.soft_stop = False
        self.completed_depth = 0
        self._qs_stats = _make_qs_stats()
        # Don't reset kpi or _diag ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â they accumulate across searches for reporting

    def reset_kpi(self):
        """Reset KPI counters."""
        self.kpi = _make_kpi_dict()

    def reset_diag(self):
        """Reset diagnostic counters."""
        self._diag = _make_diag_dict()

    def set_search_randomization(self, seed: int):
        """Set random seed for move ordering diversity.

        In Lazy SMP, each worker should call this with a different seed
        so they explore different parts of the search tree.
        """
        self._rng.seed(seed)
        self._move_order_salt = seed & 0xFFFF  # Worker-specific constant for tiebreaker

    def set_eval_noise(self, noise_range: int, seed: int = None):
        """Set evaluation noise for search diversity.

        Args:
            noise_range: Max noise in centipawns (e.g., 5 means ±5cp)
            seed: Random seed. Different workers should use different seeds.
        """
        self._eval_noise = noise_range
        if seed is not None:
            self._eval_rng.seed(seed)
            self._move_order_salt = seed & 0xFFFF  # Also set move order salt for diversity
        # Pre-compute noise table (64K entries, covers 16 bits of hash)
        if noise_range > 0:
            self._noise_table = np.array(
                [self._eval_rng.randint(-noise_range, noise_range) for _ in range(65536)],
                dtype=np.int8
            )
        else:
            self._noise_table = None

    def _add_eval_noise(self, score: int, zobrist_hash: int) -> int:
        """Add deterministic noise to evaluation score using pre-computed table."""
        if self._noise_table is None:
            return score
        return score + int(self._noise_table[zobrist_hash & 0xFFFF])

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _diag_warn(self, key: str, msg: str):
        """Record diagnostic event and warn if threshold exceeded."""
        self._diag[key] += 1
        if not is_debug_enabled():
            return
        count = self._diag[key]
        if count == 1 or count == _DIAG_WARN_THRESHOLD or (
                count > _DIAG_WARN_THRESHOLD and count % (_DIAG_WARN_THRESHOLD * 10) == 0):
            print(f"info string DIAG[{key}={count}]: {msg}", flush=True)

    def diag_summary(self) -> str:
        """Return summary of diagnostic counters (non-zero only)."""
        non_zero = {k: v for k, v in self._diag.items() if v > 0 and not k.startswith("bestmove_")}
        summary_parts = []
        if non_zero:
            summary_parts.append(", ".join(f"{k}={v}" for k, v in non_zero.items()))
        if self._diag["bestmove_count"] > 0:
            avg_depth = self._diag["bestmove_depth_sum"] / self._diag["bestmove_count"]
            summary_parts.append(f"avg_depth={avg_depth:.1f}")
        if summary_parts:
            return "DIAG_SUMMARY: " + " | ".join(summary_parts)
        return "DIAG_SUMMARY: all clear"

    # ------------------------------------------------------------------
    # Time and stop checking
    # ------------------------------------------------------------------

    def check_time(self):
        """Check if time limit exceeded. Sets self.soft_stop."""
        if TimeControl.time_limit is None:
            return
        current_time = time.perf_counter()
        elapsed = current_time - TimeControl.start_time
        if elapsed >= TimeControl.time_limit:
            self.soft_stop = True
        if TimeControl.hard_stop_time and current_time >= TimeControl.hard_stop_time:
            TimeControl.stop_search = True
            self.soft_stop = True

    def _generation_stop(self) -> bool:
        """Check if this engine should stop due to generation change."""
        if self._search_generation is None:
            return False
        sg = self._search_generation
        return sg.generation == 0 or sg.generation != self._generation

    def should_stop_search(self, current_depth: int) -> bool:
        """Determine if search should stop."""
        if TimeControl.stop_search:
            return True
        if self.soft_stop and current_depth > config.MIN_NEGAMAX_DEPTH:
            return True
        if self._generation_stop():
            return True
        return False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_classical(self, board: CachedBoard, skip_game_over: bool = False) -> int:
        """Evaluate using classical material + PST."""
        if not skip_game_over and board.is_game_over():
            if board.is_checkmate():
                return -config.MAX_SCORE + board.ply()
            else:
                return 0
        self.kpi['pos_eval'] += 1
        return self._add_eval_noise(board.material_evaluation_full(), board.zobrist_hash())

    def evaluate_nn(self, board: CachedBoard, skip_game_over: bool = False) -> int:
        """Evaluate using neural network with cache."""
        if not skip_game_over and board.is_game_over():
            if board.is_checkmate():
                return -config.MAX_SCORE + board.ply()
            else:
                return 0

        key = board.zobrist_hash()
        if key in nn_eval_cache:
            self.kpi['dec_hits'] += 1
            return self._add_eval_noise(nn_eval_cache[key], key)

        self.kpi['nn_evals'] += 1
        score = self.nn_evaluator.evaluate_centipawns(board)
        nn_eval_cache[key] = score
        return self._add_eval_noise(score, key)

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def move_score_int(self, board: CachedBoard, move_int: int, depth: int) -> int:
        """Score a move for move ordering using integer representation."""
        score = 0
        is_capture = board.is_capture_int(move_int)

        if not is_capture and depth is not None and 0 <= depth < len(self.killer_moves_int):
            km = self.killer_moves_int
            if move_int == km[depth][0]:
                score += 9000
            elif move_int == km[depth][1]:
                score += 8000

        if is_capture:
            score += board.get_mvv_lva_score_int(move_int)

        if move_int < HISTORY_TABLE_SIZE:
            score += self.history_heuristic[move_int]

        if board.gives_check_int(move_int):
            score += 50

        return score

    def ordered_moves_int(self, board: CachedBoard, depth: int,
                          pv_move_int: int = 0, tt_move_int: int = 0) -> List[int]:
        """Return legal moves as integers, ordered by expected quality.

        For Lazy SMP workers (worker_id > 0), adds deterministic noise to move scores
        to create search diversity across threads. Worker 0 uses optimal ordering.
        """
        moves_int = board.get_legal_moves_int()
        board.precompute_move_info_int(moves_int)

        # Determine if we should add noise (only for non-primary workers)
        noise_range = config.LAZY_SMP_MOVE_ORDER_RANDOMNESS if self._worker_id > 0 else 0

        scored_moves = []
        for move_int in moves_int:
            if move_int == tt_move_int and tt_move_int != 0:
                score = 1000000
            elif move_int == pv_move_int and pv_move_int != 0:
                score = 900000
            else:
                score = self.move_score_int(board, move_int, depth)
                # Add deterministic noise for Lazy SMP diversity (replaces RNG)
                if noise_range > 0:
                    # Use hash of move XOR worker salt to get deterministic but varied noise
                    noise_hash = ((move_int * 2654435761) ^ self._move_order_salt) & 0xFFFF
                    noise = (noise_hash % (2 * noise_range + 1)) - noise_range
                    score += noise
            scored_moves.append((score, move_int))

        scored_moves.sort(key=lambda tup: tup[0], reverse=True)
        return [move_int for _, move_int in scored_moves]

    def age_heuristic_history(self):
        """Vectorized history aging using numpy."""
        self.history_heuristic = (self.history_heuristic * 3) // 4

    # ------------------------------------------------------------------
    # PV extraction
    # ------------------------------------------------------------------

    def extract_pv_from_tt_int(self, board: CachedBoard, max_depth: int) -> List[int]:
        """Extract the principal variation from the transposition table as integer moves."""
        pv = []
        seen_keys = set()
        for _ in range(max_depth):
            key = board.zobrist_hash()
            if key in seen_keys:
                break
            seen_keys.add(key)
            entry = transposition_table.get(key)
            if entry is None or entry.best_move_int == 0:
                break
            move_int = entry.best_move_int
            legal_moves = board.get_legal_moves_int()
            if move_int not in legal_moves:
                self._diag_warn("pv_illegal_moves", f"PV move {move_int} illegal at depth {len(pv)}")
                break
            pv.append(move_int)
            push_move_int(board, move_int, self.nn_evaluator)
        # Restore board state
        for _ in range(len(pv)):
            board.pop()
            self.nn_evaluator.pop()
        return pv

    # ------------------------------------------------------------------
    # Quiescence search
    # ------------------------------------------------------------------

    def quiescence(self, board: CachedBoard, alpha: int, beta: int, q_depth: int,
                   last_capture_sq: int = -1) -> Tuple[int, List[int]]:
        """Quiescence search with improved time management and move limits."""
        self.kpi['q_depth'] = max(self.kpi['q_depth'], q_depth)
        self.kpi['nodes'] += 1
        self._qs_stats["total_nodes"] += 1
        self._qs_stats["max_depth_reached"] = max(self._qs_stats["max_depth_reached"], q_depth)

        # Check time periodically (inlined to avoid double perf_counter calls)
        if self._qs_stats["total_nodes"] % config.QS_TIME_CHECK_INTERVAL == 0:
            if TimeControl.time_limit is not None:
                current_time = time.perf_counter()
                elapsed = current_time - TimeControl.start_time
                if elapsed >= TimeControl.time_limit:
                    self.soft_stop = True
                if elapsed > TimeControl.time_limit * config.QS_TIME_BUDGET_FRACTION:
                    self.soft_stop = True
                    self._diag_warn("qs_budget_exceeded",
                                    f"QS exceeded {config.QS_TIME_BUDGET_FRACTION * 100:.0f}% time budget at depth {q_depth}")
                if TimeControl.hard_stop_time and current_time >= TimeControl.hard_stop_time:
                    TimeControl.stop_search = True
                    self.soft_stop = True

        if TimeControl.time_limit and TimeControl.time_limit < 1.0:
            self.check_time()

        # Hard stop always honored immediately
        if TimeControl.stop_search or self._generation_stop():
            if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
                return self.evaluate_nn(board, skip_game_over=True), []
            else:
                return self.evaluate_classical(board, skip_game_over=True), []

        # Earlier soft stop in QS
        in_capture_sequence = (last_capture_sq >= 0)
        if self.soft_stop and q_depth > round(config.MAX_QS_DEPTH // config.QS_SOFT_STOP_DIVISOR):
            if not in_capture_sequence:
                self._qs_stats["time_cutoffs"] += 1
                self._diag_warn("qs_time_cutoff", f"QS soft-stopped at depth {q_depth}")
                if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
                    return self.evaluate_nn(board, skip_game_over=True), []
                else:
                    return self.evaluate_classical(board, skip_game_over=True), []

        # Hard depth limit
        if q_depth > config.MAX_QS_DEPTH:
            self._diag_warn("qs_depth_exceeded", f"QS hit depth {q_depth}, fen={board.fen()[:40]}")
            if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL:
                return self.evaluate_nn(board, skip_game_over=True), []
            else:
                return self.evaluate_classical(board, skip_game_over=True), []

        # Draw detection
        if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
            return get_draw_score(board), []

        # TT lookup
        key = board.zobrist_hash()
        if config.QS_TT_SUPPORTED and key in qs_transposition_table:
            self.kpi['qs_tt_hits'] += 1
            stored_score = qs_transposition_table[key]
            if stored_score >= beta:
                return stored_score, []

        is_check = board.is_check()
        best_pv = []
        best_score = -config.MAX_SCORE

        # Stand-pat (not valid when in check)
        if not is_check:
            is_nn_eval = False
            if config.NN_ENABLED and q_depth <= config.QS_DEPTH_MIN_NN_EVAL:
                stand_pat = self.evaluate_nn(board)
                is_nn_eval = True
            else:
                stand_pat = self.evaluate_classical(board)

            if (not is_nn_eval and config.NN_ENABLED and q_depth <= config.QS_DEPTH_MAX_NN_EVAL
                    and abs(stand_pat) < config.STAND_PAT_MAX_NN_EVAL
                    and abs(stand_pat - beta) < config.QS_DELTA_MAX_NN_EVAL):
                stand_pat = self.evaluate_nn(board)
                is_nn_eval = True

            if stand_pat >= beta:
                self.kpi['beta_cutoffs'] += 1
                return stand_pat, []

            if stand_pat + PIECE_VALUES[QUEEN] < alpha:
                return stand_pat, []

            if (not is_nn_eval and config.NN_ENABLED
                    and q_depth <= config.QS_DEPTH_MAX_NN_EVAL
                    and abs(stand_pat) < config.STAND_PAT_MAX_NN_EVAL
                    and (stand_pat > alpha or abs(stand_pat - alpha) < config.QS_DELTA_MAX_NN_EVAL)):
                stand_pat = self.evaluate_nn(board)

            best_score = stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

        # Dynamic move limit
        move_limit = None
        for i in range(len(config.MAX_QS_MOVES_DIVISOR)):
            if q_depth <= round(config.MAX_QS_DEPTH / config.MAX_QS_MOVES_DIVISOR[i]):
                move_limit = config.MAX_QS_MOVES[i]
                break
        if move_limit is None:
            move_limit = config.MAX_QS_MOVES[-1]

        time_critical = self.soft_stop or (
                TimeControl.time_limit and TimeControl.start_time and
                (time.perf_counter() - TimeControl.start_time) > TimeControl.time_limit * config.QS_TIME_CRITICAL_FACTOR
        )
        if time_critical:
            move_limit = min(config.MAX_QS_MOVES_TIME_CRITICAL, move_limit)

        moves_searched = 0
        all_moves = ordered_moves_q_search_int(board, last_capture_sq)

        for move_int in all_moves:
            to_sq = (move_int >> 6) & 0x3F
            is_recapture = (0 <= last_capture_sq == to_sq)

            if not is_check and moves_searched >= move_limit:
                if not is_recapture:
                    self._diag_warn("qs_move_limit", f"QS move limit ({move_limit}) at depth {q_depth}")
                    break

            if not is_check:
                is_capture_move = board.is_capture_int(move_int)
                if not is_capture_move:
                    if q_depth > config.CHECK_QS_MAX_DEPTH:
                        continue
                    if not board.gives_check_int(move_int):
                        continue

                # Delta pruning
                if (q_depth >= config.DELTA_PRUNING_QS_MIN_DEPTH and is_capture_move
                        and not board.gives_check_int(move_int)):
                    victim_type = board.get_victim_type_int(move_int)
                    if victim_type:
                        gain = PIECE_VALUES[victim_type]
                        if best_score + gain + config.DELTA_PRUNING_QS_MARGIN < alpha:
                            continue

            if is_check:
                is_capture_move = board.is_capture_int(move_int)

            should_update_nn = q_depth <= config.QS_DEPTH_MAX_NN_EVAL
            if should_update_nn:
                push_move_int(board, move_int, self.nn_evaluator)
            else:
                board.push(move_int)

            child_capture_sq = to_sq if is_capture_move else -1
            score, child_pv = self.quiescence(board, -beta, -alpha, q_depth + 1, child_capture_sq)
            score = -score
            board.pop()

            if should_update_nn:
                self.nn_evaluator.pop()

            moves_searched += 1

            if in_capture_sequence and to_sq == last_capture_sq:
                in_capture_sequence = False

            if moves_searched % 5 == 0:
                self.check_time()
                if TimeControl.stop_search or self._generation_stop():
                    break
                if self.soft_stop and q_depth > round(config.MAX_QS_DEPTH // config.QS_SOFT_STOP_DIVISOR):
                    if not in_capture_sequence:
                        self._qs_stats["time_cutoffs"] += 1
                        break

            if score > best_score:
                best_score = score
                best_pv = [move_int] + child_pv

            if score >= beta:
                self.kpi['beta_cutoffs'] += 1
                if config.QS_TT_SUPPORTED:
                    qs_transposition_table[key] = score
                return score, best_pv

            if score > alpha:
                alpha = score

        # No moves searched = checkmate/stalemate
        if moves_searched == 0:
            if is_check:
                return -config.MAX_SCORE + board.ply(), []
            if board.is_game_over():
                return 0, []

        if config.QS_TT_SUPPORTED:
            qs_transposition_table[key] = best_score

        return best_score, best_pv

    # ------------------------------------------------------------------
    # Negamax
    # ------------------------------------------------------------------

    def negamax(self, board: CachedBoard, depth: int, alpha: int, beta: int,
                allow_singular: bool = True) -> Tuple[int, List[int]]:
        """Negamax search with alpha-beta pruning."""
        self.kpi['nodes'] += 1

        self.check_time()
        if TimeControl.stop_search or self._generation_stop():
            return self.evaluate_classical(board, skip_game_over=True), []

        # Draw detection
        if is_draw_by_repetition(board) or board.can_claim_fifty_moves():
            return get_draw_score(board), []

        key = board.zobrist_hash()
        alpha_orig = alpha
        beta_orig = beta

        best_move_int = 0
        best_pv = []
        max_eval = -config.MAX_SCORE

        # TT Lookup
        entry = transposition_table.get(key)
        tt_move_int = 0
        if entry and entry.depth >= depth:
            self.kpi['tt_hits'] += 1
            if entry.flag == TT_EXACT:
                return entry.score, self.extract_pv_from_tt_int(board, depth)
            elif entry.flag == TT_LOWER_BOUND:
                alpha = max(alpha, entry.score)
            elif entry.flag == TT_UPPER_BOUND:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score, []
            tt_move_int = entry.best_move_int
            if tt_move_int != 0:
                legal_moves = board.get_legal_moves_int()
                if tt_move_int not in legal_moves:
                    self._diag_warn("tt_illegal_moves",
                                    f"TT move {tt_move_int} illegal in {board.fen()[:50]}")
                    tt_move_int = 0

        # Quiescence if depth == 0
        if depth == 0:
            return self.quiescence(board, alpha, beta, 1)

        in_check = board.is_check()

        # Razoring
        if (config.RAZORING_ENABLED
                and not in_check
                and depth <= config.RAZORING_MAX_DEPTH
                and depth >= 1):
            static_eval = self.evaluate_classical(board)
            margin = config.RAZORING_MARGIN[depth] if depth < len(config.RAZORING_MARGIN) else \
                config.RAZORING_MARGIN[-1]
            if static_eval + margin <= alpha:
                qs_score, qs_pv = self.quiescence(board, alpha, beta, 1)
                if qs_score <= alpha:
                    self.kpi['razoring_prunes'] += 1
                    return qs_score, qs_pv

        # Null Move Pruning
        if (depth >= config.NULL_MOVE_MIN_DEPTH
                and not in_check
                and board.has_non_pawn_material(board.turn)
                and board.occupied.bit_count() > 6):
            # Null move represented as 0 (from=0, to=0, promo=0)
            NULL_MOVE_INT = 0
            push_move_int(board, NULL_MOVE_INT, self.nn_evaluator)
            score, _ = self.negamax(board, depth - 1 - config.NULL_MOVE_REDUCTION, -beta, -beta + 1,
                                    allow_singular=False)
            score = -score
            board.pop()
            self.nn_evaluator.pop()
            if score >= beta:
                return beta, []

        # Singular Extension Check
        singular_extension_applicable = False
        singular_move_int = 0

        tt_entry = transposition_table.get(key)
        if (allow_singular
                and depth >= 6
                and tt_move_int != 0
                and not in_check
                and tt_entry is not None
                and tt_entry.flag != TT_UPPER_BOUND
                and tt_entry.depth >= depth - 3):

            reduced_depth = max(1, depth // 2 - 1)
            reduced_beta = tt_entry.score - config.SINGULAR_MARGIN

            move_count = 0
            highest_score = -config.MAX_SCORE

            for move_int in self.ordered_moves_int(board, depth, tt_move_int=tt_move_int):
                if move_int == tt_move_int:
                    continue
                if move_count >= 3:
                    break
                push_move_int(board, move_int, self.nn_evaluator)
                score, _ = self.negamax(board, reduced_depth, -reduced_beta - 1, -reduced_beta,
                                        allow_singular=False)
                score = -score
                board.pop()
                self.nn_evaluator.pop()
                move_count += 1
                highest_score = max(highest_score, score)
                if highest_score >= reduced_beta:
                    break

            if highest_score < reduced_beta:
                singular_extension_applicable = True
                singular_move_int = tt_move_int

        # Move Ordering
        moves_int = self.ordered_moves_int(board, depth, tt_move_int=tt_move_int)
        if not moves_int:
            if in_check:
                return -config.MAX_SCORE + board.ply(), []
            return 0, []

        # Futility Pruning Setup
        futility_pruning_applicable = False
        static_eval = None
        if (config.FUTILITY_PRUNING_ENABLED
                and not in_check
                and depth <= config.FUTILITY_MAX_DEPTH
                and depth >= 1
                and abs(alpha) < config.MAX_SCORE - 100):
            static_eval = self.evaluate_classical(board)
            futility_margin = config.FUTILITY_MARGIN[depth] if depth < len(
                config.FUTILITY_MARGIN) else config.FUTILITY_MARGIN[-1]
            if static_eval + futility_margin <= alpha:
                futility_pruning_applicable = True

        for move_index, move_int in enumerate(moves_int):
            if move_index > 0 and move_index % 3 == 0:
                self.check_time()
                if TimeControl.stop_search or self._generation_stop():
                    break

            is_capture = board.is_capture_int(move_int)
            gives_check = board.gives_check_int(move_int)

            # SEE Pruning
            if (config.SEE_PRUNING_ENABLED
                    and depth <= config.SEE_PRUNING_MAX_DEPTH
                    and is_capture
                    and not in_check
                    and move_int != tt_move_int
                    and move_index > 0):
                see_threshold = -20 * depth
                if not see_ge_int(board, move_int, see_threshold):
                    self.kpi['see_prunes'] += 1
                    continue

            # Futility Pruning
            if (futility_pruning_applicable
                    and not is_capture
                    and not gives_check
                    and move_int != tt_move_int
                    and move_index > 0):
                is_killer = False
                km = self.killer_moves_int
                if depth is not None and 0 <= depth < len(self.killer_moves_int):
                    is_killer = (move_int == km[depth][0] or move_int == km[depth][1])
                if not is_killer:
                    self.kpi['futility_prunes'] += 1
                    continue

            push_move_int(board, move_int, self.nn_evaluator)
            child_in_check = board.is_check()

            # Extensions
            extension = 0
            if singular_extension_applicable and move_int == singular_move_int:
                extension = config.SINGULAR_EXTENSION
            elif child_in_check:
                extension = 1

            new_depth = depth - 1 + extension

            # LMR
            km = self.killer_moves_int
            if depth is not None and 0 <= depth < len(self.killer_moves_int):
                km0_int = km[depth][0]
                km1_int = km[depth][1]
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
                score, child_pv = self.negamax(board, new_depth - reduction, -alpha - 1, -alpha,
                                               allow_singular=True)
                score = -score
                if score > alpha:
                    score, child_pv = self.negamax(board, new_depth, -beta, -alpha, allow_singular=True)
                    score = -score
            else:
                if move_index > 0:
                    score, child_pv = self.negamax(board, new_depth, -alpha - 1, -alpha,
                                                   allow_singular=True)
                    score = -score
                    if alpha < score < beta:
                        score, child_pv = self.negamax(board, new_depth, -beta, -alpha,
                                                       allow_singular=True)
                        score = -score
                else:
                    score, child_pv = self.negamax(board, new_depth, -beta, -alpha,
                                                   allow_singular=True)
                    score = -score

            board.pop()
            self.nn_evaluator.pop()

            if score > max_eval:
                max_eval = score
                best_move_int = move_int
                best_pv = [move_int] + child_pv

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not is_capture and depth is not None and 0 <= depth < len(self.killer_moves_int):
                    km = self.killer_moves_int
                    if km[depth][0] != move_int:
                        km[depth][1] = km[depth][0]
                        km[depth][0] = move_int
                    if move_int < HISTORY_TABLE_SIZE:
                        self.history_heuristic[move_int] = min(
                            self.history_heuristic[move_int] + depth * depth, 10_000
                        )
                self.kpi['beta_cutoffs'] += 1
                break

        # Store in TT
        if max_eval <= alpha_orig:
            flag = TT_UPPER_BOUND
        elif max_eval >= beta_orig:
            flag = TT_LOWER_BOUND
        else:
            flag = TT_EXACT

        #old = transposition_table.get(key)
        #if old is None or depth >= old.depth:
        #    transposition_table[key] = TTEntry(depth, max_eval, flag, best_move_int)
        transposition_table.store_if_deeper(
            key,
            TTEntry(depth, max_eval, flag, best_move_int)
        )

        return max_eval, best_pv

    # ------------------------------------------------------------------
    # find_best_move ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the main iterative deepening entry point
    # ------------------------------------------------------------------

    def find_best_move(self, fen: str, max_depth: int = config.MAX_NEGAMAX_DEPTH,
                       time_limit: Optional[float] = None,
                       clear_tt: bool = True,
                       expected_best_moves=None,
                       use_existing_time_control: bool = False,
                       on_depth_complete: Optional[Callable] = None,
                       suppress_info: bool = False,
                       start_depth: int = 1) -> Tuple[int, int, List[int], int, float]:
        """
        Finds the best move using iterative deepening negamax.

        Args:
            fen: FEN string of the position
            max_depth: Maximum search depth
            time_limit: Time limit in seconds (None for no limit)
            clear_tt: If True, clear transposition tables before search
            expected_best_moves: For testing - stop early if best move matches
            use_existing_time_control: Skip TimeControl initialization
            on_depth_complete: Callback(depth, move_int, score, pv_int, nodes)
                called after each completed depth for intermediate result reporting.
            suppress_info: If True, suppress UCI info output (for workers).

        Returns:
            Tuple of (best_move_int, score, pv_int, nodes, nps)
            Returns ints ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â returns integers for UCI conversion.
        """
        # Initialize time control
        if not use_existing_time_control:
            TimeControl.time_limit = time_limit
            TimeControl.stop_search = False
            self.soft_stop = False
            TimeControl.start_time = time.perf_counter()
            if time_limit:
                grace_period = max(time_limit * 0.5, 0.3)
                TimeControl.hard_stop_time = TimeControl.start_time + time_limit + grace_period
            else:
                TimeControl.hard_stop_time = None
        else:
            diag_print(f"TimeControl: using existing (time_limit={TimeControl.time_limit}, "
                       f"stop={TimeControl.stop_search})")

        nodes_start = self.kpi['nodes']
        nps = 0

        # Reset QS statistics
        self._qs_stats = _make_qs_stats()

        diag_print(f"TimeControl: stop={TimeControl.stop_search}, soft={self.soft_stop}, time_limit={time_limit}")

        # Clear heuristics for new search
        self.reset_for_search()

        # Only clear TT if requested
        if clear_tt:
            transposition_table.clear()
            qs_transposition_table.clear()

        # Trim nn_eval_cache (safe only when called from main thread / single engine)
        # For Lazy SMP, parallel_find_best_move trims before spawning workers.
        if self._search_generation is None:  # Main thread only
            #control_dict_size(nn_eval_cache, config.MAX_TABLE_SIZE)
            nn_eval_cache.trim(config.MAX_TABLE_SIZE)

        # Initialize board and evaluator
        init_start = time.perf_counter()
        # Reuse persistent board if available, otherwise create new one
        if self._board is None:
            self._board = CachedBoard(fen)
        else:
            self._board.set_fen(fen)
        board = self._board

        # Check for stop before NN reset
        if TimeControl.stop_search and not TimeControl.is_ponder_search:
            diag_print(f"DEBUG: Stop received before NN reset, aborting")
            legal = board.get_legal_moves_int()
            if legal:
                return legal[0], 0, [legal[0]], 0, 0
            return 0, 0, [], 0, 0

        self.nn_evaluator.reset(board)

        total_init_time = time.perf_counter() - init_start
        if total_init_time > 0.2:
            diag_print(f"DEBUG: Total init took {total_init_time:.3f}s")

        # Critical time: emergency search
        if time_limit and time_limit < 0.15:
            self._diag_warn("critical_time_search",
                            f"time_limit={time_limit:.3f}s, doing emergency search")
            legal_moves = board.get_legal_moves_int()
            if legal_moves:
                best_move_int = legal_moves[0]
                best_score = -config.MAX_SCORE
                for move_int in legal_moves[:10]:
                    push_move_int(board, move_int, self.nn_evaluator)
                    score = -self.evaluate_nn(board) if config.NN_ENABLED else -self.evaluate_classical(
                        board)
                    board.pop()
                    self.nn_evaluator.pop()
                    if score > best_score:
                        best_score = score
                        best_move_int = move_int
                elapsed = time.perf_counter() - TimeControl.start_time
                nps = int(10 / elapsed) if elapsed > 0 else 0
                return best_move_int, best_score, [best_move_int], 10, nps
            return 0, 0, [], 0, 0

        # Start iterative deepening
        best_move_int = 0
        best_score = 0
        best_pv = []
        prev_depth_score = None
        last_depth_time = 0.0
        max_completed_depth = 0
        is_tactical_position = False
        score_unstable = False

        for depth in range(start_depth, max_depth + 1):
            depth_start_time = time.perf_counter()
            self.check_time()

            if depth <= 3:
                diag_print(
                    f"DEBUG: Starting depth {depth}, stop={TimeControl.stop_search}, soft={self.soft_stop}")

            # Determine minimum required depth
            if score_unstable:
                current_min_depth = config.UNSTABLE_MIN_DEPTH
                if depth == config.UNSTABLE_MIN_DEPTH:
                    self._diag["unstable_min_depth"] += 1
            elif is_tactical_position:
                current_min_depth = config.TACTICAL_MIN_DEPTH
            else:
                current_min_depth = config.MIN_PREFERRED_DEPTH

            if depth > current_min_depth and self.should_stop_search(depth):
                diag_print(
                    f"DEBUG: Stopping at depth {depth} due to should_stop_search (min={current_min_depth})")
                break

            # Force deeper search if below minimum
            if depth <= current_min_depth and self.soft_stop and not TimeControl.stop_search:
                if max_completed_depth > 0 and max_completed_depth < config.MIN_PREFERRED_DEPTH:
                    self.soft_stop = False
                    self._diag["min_depth_forced"] += 1
                    diag_print(
                        f"MIN_DEPTH: Forcing depth {depth}, only completed {max_completed_depth}, "
                        f"need {config.MIN_PREFERRED_DEPTH}")

            # Time prediction: skip if not enough time
            if depth > config.MIN_NEGAMAX_DEPTH and time_limit is not None and last_depth_time > 0:
                elapsed = time.perf_counter() - TimeControl.start_time
                remaining = time_limit - elapsed
                estimated_next_depth_time = last_depth_time * config.ESTIMATED_BRANCHING_FACTOR
                if max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                    if remaining < estimated_next_depth_time * config.TIME_SAFETY_MARGIN_RATIO:
                        break
                if remaining < config.EMERGENCY_TIME_RESERVE and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                    self._diag["emergency_reserve_stop"] += 1
                    diag_print(
                        f"EMERGENCY_RESERVE: Only {remaining:.2f}s left, stopping at depth {max_completed_depth}")
                    break

            self.age_heuristic_history()

            # Root TT move
            root_key = board.zobrist_hash()
            entry = transposition_table.get(root_key)
            tt_move_int = 0
            if entry and entry.best_move_int != 0:
                legal_int = board.get_legal_moves_int()
                if entry.best_move_int in legal_int:
                    tt_move_int = entry.best_move_int

            # Aspiration window
            window = config.ASPIRATION_WINDOW
            retries = 0
            depth_completed = False
            search_aborted = False

            use_full_window = (depth == 1 or is_tactical_position or
                               (prev_depth_score is not None and abs(prev_depth_score) > 500))
            if use_full_window and depth > 1:
                self._diag_warn("aw_tactical_skip",
                                f"Skipping AW at depth {depth}, tactical={is_tactical_position}, "
                                f"score={prev_depth_score}")

            while not search_aborted:
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

                pv_move_int = best_move_int if best_move_int != 0 else 0

                for move_index, move_int in enumerate(
                        self.ordered_moves_int(board, depth, pv_move_int, tt_move_int)):
                    self.check_time()

                    if depth == 1 and move_index < 3:
                        diag_print(
                            f"DEBUG: depth 1 move {move_index}: {move_int}, stop={TimeControl.stop_search}")

                    # Check for stop
                    if TimeControl.stop_search or self._generation_stop():
                        search_aborted = True
                        break
                    if self.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                        self._diag_warn("mid_depth_abort",
                                        f"Aborting depth {depth} mid-search (completed={max_completed_depth})")
                        search_aborted = True
                        break

                    push_move_int(board, move_int, self.nn_evaluator)

                    if is_draw_by_repetition(board):
                        score = get_draw_score(board)
                        child_pv = []
                    else:
                        if move_index == 0:
                            score, child_pv = self.negamax(board, depth - 1, -beta, -alpha,
                                                           allow_singular=True)
                            score = -score
                        else:
                            score, child_pv = self.negamax(board, depth - 1, -alpha - 1, -alpha,
                                                           allow_singular=True)
                            score = -score
                            if score > alpha:
                                score, child_pv = self.negamax(board, depth - 1, -beta, -alpha,
                                                               allow_singular=True)
                                score = -score

                    board.pop()
                    self.nn_evaluator.pop()

                    # Check if aborted during negamax
                    if TimeControl.stop_search or self._generation_stop() or (
                            self.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH):
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

                    if alpha >= beta:
                        break

                # Handle search abort
                if search_aborted:
                    if best_move_int == 0 and current_best_move_int != 0:
                        best_move_int = current_best_move_int
                        best_score = current_best_score
                        best_pv = current_best_pv
                    break

                # Success: within aspiration window
                if use_full_window or (current_best_score > alpha_orig and current_best_score < beta):
                    best_move_int = current_best_move_int
                    best_score = current_best_score
                    best_pv = current_best_pv
                    depth_completed = True
                    break

                # Fail-low or fail-high: widen window
                window *= 2
                retries += 1

                max_retries = config.MAX_AW_RETRIES_TACTICAL if is_tactical_position else config.MAX_AW_RETRIES

                if retries >= max_retries:
                    self._diag_warn("aspiration_retries",
                                    f"depth={depth} hit max retries ({max_retries}), "
                                    f"score={current_best_score}")
                    is_tactical_position = True

                if best_move_int == 0 and current_best_move_int != 0:
                    best_move_int = current_best_move_int
                    best_score = current_best_score
                    best_pv = current_best_pv

                # Fallback: full window search
                if retries >= max_retries:
                    alpha = -config.MAX_SCORE
                    beta = config.MAX_SCORE
                    current_best_score = -config.MAX_SCORE

                    for move_int in self.ordered_moves_int(board, depth, pv_move_int, tt_move_int):
                        self.check_time()
                        if TimeControl.stop_search or self._generation_stop():
                            search_aborted = True
                            break
                        if self.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH:
                            search_aborted = True
                            break

                        push_move_int(board, move_int, self.nn_evaluator)
                        score, child_pv = self.negamax(board, depth - 1, -beta, -alpha,
                                                       allow_singular=True)
                        score = -score
                        board.pop()
                        self.nn_evaluator.pop()

                        if TimeControl.stop_search or self._generation_stop() or (
                                self.soft_stop and max_completed_depth >= config.MIN_PREFERRED_DEPTH):
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

                    if not search_aborted:
                        if current_best_move_int != 0:
                            best_move_int = current_best_move_int
                            best_score = current_best_score
                            best_pv = current_best_pv
                        depth_completed = True
                    elif best_move_int == 0 and current_best_move_int != 0:
                        best_move_int = current_best_move_int
                        best_score = current_best_score
                        best_pv = current_best_pv
                    break

            # Record time for this depth
            if depth_completed:
                last_depth_time = time.perf_counter() - depth_start_time
                max_completed_depth = depth
                self.completed_depth = max_completed_depth

                # Score instability check
                if prev_depth_score is not None and depth > 2:
                    score_diff = abs(best_score - prev_depth_score)
                    if score_diff > _SCORE_INSTABILITY_THRESHOLD:
                        self._diag_warn("score_instability",
                                        f"depth {depth}: {prev_depth_score} -> {best_score} "
                                        f"(diff={score_diff})")
                        score_unstable = True
                        if not is_tactical_position:
                            is_tactical_position = True
                            self._diag["tactical_extension"] += 1
                            if self.soft_stop and not TimeControl.stop_search:
                                self.soft_stop = False
                                diag_print(
                                    f"TACTICAL_EXTENSION: Score swing {score_diff}cp at depth {depth}, "
                                    f"forcing min depth {config.UNSTABLE_MIN_DEPTH}")
                prev_depth_score = best_score

                # Callback: report intermediate result
                if on_depth_complete:
                    on_depth_complete(depth, best_move_int, best_score, best_pv,
                                      self.kpi['nodes'] - nodes_start)

            if search_aborted:
                break

            # Print UCI progress (main thread only)
            if depth_completed and best_pv and not suppress_info:
                elapsed = time.perf_counter() - TimeControl.start_time
                nps = int((self.kpi['nodes'] - nodes_start) / elapsed) if elapsed > 0 else 0
                validated_pv = validate_pv_int(board, best_pv)
                pv_uci = ' '.join(int_to_uci(m) for m in validated_pv)
                print(
                    f"info depth {depth} score cp {best_score} nodes {self.kpi['nodes']} nps {nps} pv {pv_uci}",
                    flush=True)

            # Early break for testing
            if best_move_int != 0 and expected_best_moves is not None:
                # Convert expected moves to ints if they aren't already
                expected_ints = set()
                for m in expected_best_moves:
                    if isinstance(m, int):
                        expected_ints.add(m)
                    else:
                        expected_ints.add(move_to_int_from_obj(m))
                if best_move_int in expected_ints:
                    break

        # Shallow search diagnostics
        self._diag["bestmove_depth_sum"] += max_completed_depth
        self._diag["bestmove_count"] += 1

        if 3 >= max_completed_depth > 0:
            self._diag["shallow_search_total"] += 1
            if max_completed_depth == 2:
                self._diag["shallow_search_d2"] += 1
            elif max_completed_depth == 3:
                self._diag["shallow_search_d3"] += 1
            best_move_uci = int_to_uci(best_move_int) if best_move_int != 0 else None
            diag_print(f"SHALLOW_SEARCH: bestmove selected at depth {max_completed_depth} "
                       f"(tactical={is_tactical_position}, move={best_move_uci})")

        # QS stats
        if self._qs_stats["max_depth_reached"] > config.MAX_QS_DEPTH // 2 or self._qs_stats[
            "time_cutoffs"] > 0:
            diag_print(f"QS_STATS: max_depth={self._qs_stats['max_depth_reached']}, "
                       f"nodes={self._qs_stats['total_nodes']}, "
                       f"time_cutoffs={self._qs_stats['time_cutoffs']}")

        # Fallback: shallow tactical search
        if best_move_int == 0:
            self._diag_warn("best_move_none",
                            f"No depth completed, using shallow search fallback, fen={fen[:40]}")
            diag_print(f"Computing shallow search fallback (no depth completed)...")

            elapsed = time.perf_counter() - TimeControl.start_time if TimeControl.start_time else 0
            diag_print(f"DEBUG fallback: elapsed={elapsed:.2f}s, stop={TimeControl.stop_search}, "
                       f"soft={self.soft_stop}, time_limit={time_limit}")

            board = self._board
            self.nn_evaluator.reset(board)
            legal = board.get_legal_moves_int()

            if legal:
                best_move_int = legal[0]
                best_score = -config.MAX_SCORE

                self._diag_warn("fallback_shallow_search",
                                f"Shallow search with {len(legal)} moves")

                fallback_aborted = False
                for move_int in legal:
                    if TimeControl.stop_search or self._generation_stop():
                        diag_print(f"Fallback aborted by stop signal")
                        fallback_aborted = True
                        break

                    push_move_int(board, move_int, self.nn_evaluator)

                    if board.is_checkmate():
                        score = config.MAX_SCORE - board.ply()
                        board.pop()
                        self.nn_evaluator.pop()
                        best_move_int = move_int
                        best_score = score
                        best_pv = [best_move_int]
                        diag_print(f"Fallback found checkmate: {int_to_uci(move_int)}")
                        break

                    if board.is_stalemate() or board.is_insufficient_material():
                        score = 0
                    else:
                        # Simple 1-ply lookahead for opponent
                        opp_best = config.MAX_SCORE
                        opp_moves_checked = 0
                        max_opp_moves = 8

                        opp_moves = board.get_legal_moves_int()
                        board.precompute_move_info_int(opp_moves)
                        for opp_move_int in opp_moves:
                            if TimeControl.stop_search or self._generation_stop():
                                break
                            _, _, promo = int_to_tuple(opp_move_int)
                            is_tactical = (board.is_capture_int(opp_move_int) or
                                           board.gives_check_int(opp_move_int) or
                                           promo != 0)
                            if not is_tactical and opp_moves_checked > 0:
                                continue

                            push_move_int(board, opp_move_int, self.nn_evaluator)
                            if board.is_checkmate():
                                opp_score = config.MAX_SCORE - board.ply()
                            else:
                                opp_score = -self.evaluate_nn(board)
                            board.pop()
                            self.nn_evaluator.pop()
                            opp_best = min(opp_best, opp_score)
                            opp_moves_checked += 1
                            if opp_moves_checked >= max_opp_moves:
                                break

                        if opp_moves_checked == 0:
                            score = -self.evaluate_nn(board)
                        else:
                            score = -opp_best

                    board.pop()
                    self.nn_evaluator.pop()

                    if score > best_score:
                        best_score = score
                        best_move_int = move_int

                best_pv = [best_move_int]
                if not fallback_aborted:
                    diag_print(
                        f"Shallow fallback: {len(legal)} moves, "
                        f"best={int_to_uci(best_move_int)} score={best_score}cp")
            else:
                best_move_int = 0
                best_score = self.evaluate_classical(board)
                best_pv = []

        # Time overrun diagnostics
        if time_limit is not None:
            elapsed = time.perf_counter() - TimeControl.start_time
            overrun = elapsed - time_limit
            if overrun > 0.5:
                self._diag_warn("time_overruns",
                                f"Search overran by {overrun:.2f}s (limit={time_limit:.2f}s, "
                                f"actual={elapsed:.2f}s)")
            if elapsed > time_limit * 3:
                self._diag_warn("time_critical_abort",
                                f"SEVERE overrun: {elapsed:.2f}s vs {time_limit:.2f}s limit, "
                                f"QS_max_depth={self._qs_stats['max_depth_reached']}")

        if abs(best_score) > config.MAX_SCORE:
            self._diag_warn("score_out_of_bounds",
                            f"Score {best_score} exceeds MAX_SCORE {config.MAX_SCORE}")

        # Compute final NPS
        elapsed = time.perf_counter() - TimeControl.start_time if TimeControl.start_time else 0
        nodes_searched = self.kpi['nodes'] - nodes_start
        nps = int(nodes_searched / elapsed) if elapsed > 0 else 0

        return best_move_int, best_score, best_pv, nodes_searched, nps


# ======================================================================
# Module-level backward-compatible wrappers
# ======================================================================

# Global KPI and diag dicts for backward compat with uci.py
kpi = _make_kpi_dict()
_diag = _make_diag_dict()

# Main thread engine instance (created lazily)
_main_engine: Optional[ChessEngine] = None


def _get_main_engine() -> ChessEngine:
    """Get or create the main thread's ChessEngine."""
    global _main_engine
    if _main_engine is None:
        _main_engine = ChessEngine(nn_eval=nn_evaluator)
        # Main engine shares the module-level kpi so that uci.py's
        # imported reference (from chess_engine import kpi) stays valid.
        _main_engine.kpi = kpi
    return _main_engine


def find_best_move(fen, max_depth=config.MAX_NEGAMAX_DEPTH, time_limit=None, clear_tt=True,
                   expected_best_moves=None, use_existing_time_control=False) -> \
        Tuple[int, int, List[int], int, float]:
    """
    Main search wrapper: creates/reuses main ChessEngine and calls find_best_move.

    Returns:
        Tuple of (best_move_int, score, pv_int, nodes, nps)
        - best_move_int: Integer move (0 for null/no move)
        - score: Evaluation in centipawns
        - pv_int: Principal variation as list of integer moves
        - nodes: Number of nodes searched
        - nps: Nodes per second
    """
    engine = _get_main_engine()
    # Ensure main engine uses the global nn_evaluator
    engine.nn_evaluator = nn_evaluator

    bm_int, score, pv_int, nodes, nps_val = engine.find_best_move(
        fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt,
        expected_best_moves=expected_best_moves,
        use_existing_time_control=use_existing_time_control)

    board = CachedBoard(fen)
    validated_pv = validate_pv_int(board, pv_int)

    return bm_int, score, validated_pv, nodes, nps_val


def diag_summary() -> str:
    """Return diagnostic summary from main engine."""
    engine = _get_main_engine()
    return engine.diag_summary()


# ======================================================================
# Main (interactive testing)
# ======================================================================

def main():
    """Interactive loop to input FEN positions and get the best move."""
    engine = _get_main_engine()

    while True:
        try:
            fen = input("FEN: ").strip()
            if fen.lower() in ("exit", "quit"):
                break
            if fen == "":
                print("Type 'exit' or 'quit' to quit")
                continue

            engine.reset_kpi()
            start_time = time.perf_counter()
            bm_int, score, pv_int, nodes, nps_val = engine.find_best_move(fen, max_depth=20,
                                                                          time_limit=5)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            best_move_uci = int_to_uci(bm_int) if bm_int != 0 else None

            print("\n--- Search KPIs ---")
            for key, value in engine.kpi.items():
                print(f"{key}: {value}")
            print(f"time: {elapsed_time:.2f}")
            print(f"nps: {nps_val}")

            board = CachedBoard(fen)
            print("\nBest move:", best_move_uci)
            print("Evaluation:", score)
            print("PV:", pv_to_san(board, pv_int))
            print("PV (UCI):", " ".join(int_to_uci(m) for m in pv_int))
            print("-------------------\n")

        except KeyboardInterrupt:
            response = input(
                "\nKeyboardInterrupt detected. Type 'exit' to quit, Enter to continue: ").strip()
            if response.lower() == "exit":
                break
            print("Resuming...\n")
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
            continue


if __name__ == '__main__':
    import random

    random.seed(42)
    main()