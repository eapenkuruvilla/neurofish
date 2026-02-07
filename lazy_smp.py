#!/usr/bin/env python3
"""
Multiprocessing search module for chess engine - Lazy SMP implementation.

Lazy SMP approach:
- All workers search the SAME position independently
- Workers communicate only through shared transposition table
- No explicit synchronization - TT provides implicit coordination
- Different workers start at slightly different depths for diversity

Threading is used instead of multiprocessing because:
- Shared memory (TT, NN cache) without IPC overhead
- NumPy/PyTorch release GIL during heavy operations
- Much simpler implementation
"""

import threading
import time
from typing import List, Tuple, Optional
from queue import Queue, Empty
from multiprocessing import Value

import chess

import config
from cached_board import CachedBoard, int_to_move
from config import MAX_THREADS


def _mp_diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    from chess_engine import is_debug_enabled
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


# Thread pool
_worker_threads: List[threading.Thread] = []
_result_queue: Optional[Queue] = None
_pool_initialized = False
_num_workers = 0

# Shared generation counter for stop signaling
# Value 0 = stop, >0 = valid search generation
_search_generation: Optional[Value] = None


def init_worker_pool(num_workers: int):
    """
    Initialize the worker pool configuration.
    Actual threads are created per-search in Lazy SMP.
    """
    global _pool_initialized, _num_workers, _search_generation, _result_queue

    if num_workers <= 1:
        _pool_initialized = False
        _num_workers = 0
        return

    _num_workers = num_workers
    _result_queue = Queue()

    # Shared generation counter - no lock needed (single writer, multiple readers)
    _search_generation = Value('i', 1, lock=False)

    _pool_initialized = True
    _mp_diag_print(f"Lazy SMP initialized with {num_workers} threads")


def shutdown_worker_pool():
    """Shutdown signal - threads are per-search so just reset state."""
    global _pool_initialized, _num_workers, _worker_threads

    if not _pool_initialized:
        return

    _mp_diag_print("Shutting down Lazy SMP")

    # Signal any running threads to stop
    if _search_generation:
        _search_generation.value = 0

    # Wait for any active threads
    for t in _worker_threads:
        if t.is_alive():
            t.join(timeout=1.0)

    _worker_threads = []
    _pool_initialized = False


def _lazy_smp_worker(worker_id: int, fen: str, max_depth: int, generation: int, result_queue: Queue,
                     search_generation: Value):
    """
    Lazy SMP worker thread.

    Each worker searches the same position independently.
    Workers start at different depths for search diversity.
    Communication happens through shared TT (implicit).
    """
    import chess_engine
    from config import MAX_SCORE

    # Get thread-local nn_evaluator (chess_engine.py handles creation)
    nn_evaluator = chess_engine.get_nn_evaluator()

    # Create board for this thread
    board = CachedBoard(fen)
    nn_evaluator.reset(board)

    # Stagger starting depths for diversity
    # Worker 0: starts at depth 1
    # Worker 1: starts at depth 2
    # Worker 2: starts at depth 1
    # etc.
    start_depth = 1 + (worker_id % 2)

    # Initialize search state
    best_move_int = 0
    best_score = -MAX_SCORE
    best_pv = []
    max_completed_depth = 0

    # NOTE: Do NOT modify TimeControl here - it's shared across all threads
    # and is controlled by the main thread/UCI loop
    # Workers use the generation value for stop signaling instead

    # Clear thread-local heuristics (these are global but OK to share - minor impact)
    for i in range(len(chess_engine.killer_moves)):
        chess_engine.killer_moves[i] = [None, None]
    for i in range(len(chess_engine.killer_moves_int)):
        chess_engine.killer_moves_int[i] = [0, 0]
    chess_engine.history_heuristic.fill(0)

    # Reset node counter
    chess_engine.kpi['nodes'] = 0

    def should_stop() -> bool:
        """Check if search should stop."""
        if search_generation.value == 0:
            return True
        if search_generation.value != generation:
            return True
        # Also check TimeControl for external stop
        if chess_engine.TimeControl.stop_search:
            return True
        return False

    try:
        # Iterative deepening from staggered start
        for depth in range(start_depth, max_depth + 1):
            if should_stop():
                break

            # Workers don't use check_time() - they use generation value for stopping
            # The main parallel_find_best_move handles the time limit

            depth_start = time.perf_counter()

            # Get ordered moves
            board.precompute_move_info_int()
            pv_move_int = best_move_int if best_move_int != 0 else 0

            # Look up TT for this position
            tt_move_int = 0
            zh = board.zobrist_hash()
            if zh in chess_engine.transposition_table:
                tt_entry = chess_engine.transposition_table[zh]
                tt_move_int = tt_entry.best_move_int

            ordered_moves = list(chess_engine.ordered_moves_int(board, depth, pv_move_int, tt_move_int))

            if not ordered_moves:
                break

            depth_best_move_int = 0
            depth_best_score = -MAX_SCORE
            depth_best_pv = []
            search_aborted = False

            alpha = -MAX_SCORE
            beta = MAX_SCORE

            for move_idx, move_int in enumerate(ordered_moves):
                if should_stop():
                    search_aborted = True
                    break

                move = int_to_move(move_int)
                chess_engine.push_move(board, move, nn_evaluator)

                try:
                    if chess_engine.is_draw_by_repetition(board):
                        score = chess_engine.get_draw_score(board)
                        child_pv = []
                    else:
                        # PVS: full window for first move, null window for rest
                        if move_idx == 0:
                            score, child_pv = chess_engine.negamax(board, depth - 1, -beta, -alpha,
                                                             allow_singular=True)
                            score = -score
                        else:
                            # Null window search
                            score, child_pv = chess_engine.negamax(board, depth - 1, -alpha - 1, -alpha,
                                                             allow_singular=True)
                            score = -score
                            if score > alpha and not should_stop():
                                # Re-search with full window
                                score, child_pv = chess_engine.negamax(board, depth - 1, -beta, -alpha,
                                                                 allow_singular=True)
                                score = -score
                finally:
                    board.pop()
                    nn_evaluator.pop()

                if should_stop():
                    search_aborted = True
                    if depth_best_move_int == 0:
                        depth_best_move_int = move_int
                        depth_best_score = score
                        depth_best_pv = [move_int] + child_pv
                    break

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move_int = move_int
                    depth_best_pv = [move_int] + child_pv

                if score > alpha:
                    alpha = score

                if alpha >= beta:
                    break

            # Only save completed depth results
            if not search_aborted and depth_best_move_int != 0:
                best_move_int = depth_best_move_int
                best_score = depth_best_score
                best_pv = depth_best_pv
                max_completed_depth = depth

                # Report completed depth
                if not should_stop():
                    result = (worker_id, best_move_int, best_score, best_pv,
                              chess_engine.kpi['nodes'], max_completed_depth, generation)
                    result_queue.put(result)

                    depth_time = time.perf_counter() - depth_start
                    _mp_diag_print(f"Worker {worker_id} completed depth {depth}: "
                                   f"{int_to_move(best_move_int).uci()} score={best_score} "
                                   f"time={depth_time:.2f}s")

    except Exception as e:
        import traceback
        print(f"info string Worker {worker_id} error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    # Send final result
    if best_move_int != 0 and not should_stop():
        result = (worker_id, best_move_int, best_score, best_pv,
                  chess_engine.kpi['nodes'], max_completed_depth, generation)
        result_queue.put(result)


def parallel_find_best_move(fen: str, max_depth: int = 20, time_limit: Optional[float] = None,
                            clear_tt: bool = True) -> Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Find best move using Lazy SMP parallel search.

    All workers search the same position. They communicate implicitly
    through the shared transposition table.

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _worker_threads, _search_generation

    # Fall back to single-threaded if not initialized
    if not _pool_initialized or _num_workers <= 1:
        import chess_engine
        return chess_engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    import chess_engine

    start_time = time.perf_counter()

    # Reset TimeControl for new search
    chess_engine.TimeControl.stop_search = False
    chess_engine.TimeControl.soft_stop = False
    chess_engine.TimeControl.start_time = start_time
    chess_engine.TimeControl.time_limit = time_limit
    chess_engine.TimeControl.hard_stop_time = None  # Don't use hard stop for MP - use generation instead

    # Increment generation to signal new search
    new_gen = _search_generation.value + 1
    if new_gen <= 0:
        new_gen = 1
    _search_generation.value = new_gen
    current_generation = new_gen

    # Clear TT if requested
    if clear_tt:
        chess_engine.transposition_table.clear()
        chess_engine.qs_transposition_table.clear()

    # Get legal moves to validate results later
    board = CachedBoard(fen)
    legal_moves = list(board.get_legal_moves_list())

    if not legal_moves:
        return chess.Move.null(), 0, [], 0, 0

    if len(legal_moves) == 1:
        return chess_engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    _mp_diag_print(f"Lazy SMP search gen={current_generation} with {_num_workers} threads")
    _mp_diag_print(f"FEN: {fen[:60]}...")

    # Clear result queue
    while not _result_queue.empty():
        try:
            _result_queue.get_nowait()
        except Empty:
            break

    # Worker time limit (slightly less to ensure they stop before main timeout)
    worker_time_limit = time_limit * 0.90 if time_limit else None

    # Start worker threads
    _worker_threads = []
    for i in range(_num_workers):
        t = threading.Thread(
            target=_lazy_smp_worker,
            args=(i, fen, max_depth, current_generation,
                  _result_queue, _search_generation),
            daemon=True
        )
        t.start()
        _worker_threads.append(t)

    # Collect results
    best_result = None  # (move_int, score, pv, nodes, depth)
    total_nodes = 0
    workers_reported = set()

    while True:
        # Check for external stop
        if _search_generation.value == 0:
            _mp_diag_print("Stop signal detected")
            break

        # Check for timeout
        if time_limit and (time.perf_counter() - start_time) >= time_limit:
            _mp_diag_print("Time limit reached, stopping workers")
            _search_generation.value = 0
            # Do NOT set TimeControl.stop_search here - that's for external stop signals
            # Setting it would cause the ponder wait loop to exit prematurely
            break

        try:
            result = _result_queue.get(timeout=0.05)
            worker_id, move_int, score, pv, nodes, depth, gen = result

            # Ignore stale results
            if gen != current_generation:
                continue

            workers_reported.add(worker_id)
            total_nodes = max(total_nodes, nodes)  # Approximate - threads share nodes

            # Keep result from highest depth
            if best_result is None or depth > best_result[4]:
                best_result = (move_int, score, pv, nodes, depth)
                _mp_diag_print(f"New best from worker {worker_id}: "
                               f"{int_to_move(move_int).uci()} score={score} depth={depth}")
            elif depth == best_result[4] and score > best_result[1]:
                # Same depth, better score
                best_result = (move_int, score, pv, nodes, depth)
                _mp_diag_print(f"Better score from worker {worker_id}: "
                               f"{int_to_move(move_int).uci()} score={score} depth={depth}")

        except Empty:
            # Check if all threads have finished
            all_done = all(not t.is_alive() for t in _worker_threads)
            if all_done:
                _mp_diag_print("All workers finished")
                break

    # Ensure all threads stop
    _search_generation.value = 0
    for t in _worker_threads:
        t.join(timeout=0.5)

    # Reset generation to non-zero so next search doesn't start stopped
    _search_generation.value = 1

    _mp_diag_print(f"Collected results from {len(workers_reported)} workers")

    # Process results
    if best_result is None:
        _mp_diag_print("No results from workers, using first legal move")
        return legal_moves[0], 0, [legal_moves[0]], 0, 0

    move_int, score, pv_int, nodes, depth = best_result
    best_move = int_to_move(move_int)

    # Validate move is legal
    if best_move not in legal_moves:
        _mp_diag_print(f"ERROR: Best move {best_move.uci()} is illegal!")
        return legal_moves[0], 0, [legal_moves[0]], 0, 0

    # Convert PV to moves
    best_pv = [int_to_move(m) for m in pv_int if m != 0]

    elapsed = time.perf_counter() - start_time
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0

    return best_move, score, best_pv, total_nodes, nps


def stop_parallel_search():
    """Signal all workers to stop."""
    if _search_generation:
        _search_generation.value = 0


def set_lazy_smp_threads(threads: int):
    """Set number of threads and initialize pool."""
    import config
    if threads > config.MAX_THREADS:
        threads = config.MAX_THREADS

    if threads > 1:
        init_worker_pool(threads)
    else:
        shutdown_worker_pool()


def is_mp_enabled() -> bool:
    """Check if multiprocessing is enabled."""
    return _pool_initialized and _num_workers > 1


def main():
    """Test function."""
    set_lazy_smp_threads(config.MAX_THREADS)
    time.sleep(0.5)

    try:
        while True:
            fen = input("FEN: ").strip()
            if fen.lower() in ("exit", "quit"):
                break
            if fen == "":
                print("Type 'exit' or 'quit' to quit")
                continue

            move, score, pv, nodes, nps = parallel_find_best_move(fen, max_depth=20, time_limit=10)

            print(f"nodes: {nodes}")
            print(f"nps: {nps}")
            print(f"Best move: {move}")
            print(f"Score: {score}")
            print(f"PV: {' '.join(m.uci() for m in pv)}")
            print("-------------------\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    finally:
        stop_parallel_search()
        shutdown_worker_pool()


if __name__ == '__main__':
    main()