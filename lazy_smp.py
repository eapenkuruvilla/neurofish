#!/usr/bin/env python3
"""
Multiprocessing search module for chess engine - Lazy SMP implementation.

Lazy SMP approach:
- All workers search the SAME position independently
- Workers communicate only through shared transposition table
- No explicit synchronization - TT provides implicit coordination
- Different workers use different eval noise seeds for search diversity

Optimizations in this version:
- Pre-populated NN evaluator pool (avoid model loading per search)
- PV validation only at end (not on every depth improvement)
- Callbacks only for depth improvements (skip redundant reports)
"""

import threading
import time
from typing import List, Tuple, Optional
from queue import Queue, Empty

import config
from cached_board import CachedBoard, int_to_uci


def _mp_diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    from chess_engine import is_debug_enabled
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


# Thread pool state
_worker_threads: List[threading.Thread] = []
_result_queue: Optional[Queue] = None
_pool_initialized = False
_num_workers = 0

# Configuration constants
MIN_RESULT_WAIT = 0.15  # Minimum seconds to wait for at least one result
MIN_PREFERRED_DEPTH = 5  # Minimum depth before allowing time-based stop


def init_worker_pool(num_workers: int):
    """Initialize the worker pool configuration and pre-populate NN evaluator pool."""
    global _pool_initialized, _num_workers, _result_queue

    if num_workers <= 1:
        _pool_initialized = False
        _num_workers = 0
        return

    _num_workers = num_workers
    _result_queue = Queue()
    _pool_initialized = True

    # FIX #1: Pre-populate NN evaluator pool to avoid model loading during search
    # Note: chess_engine.py has a global nn_evaluator that gets used when pool is empty
    # We only need (num_workers - 1) additional evaluators in the pool
    from chess_engine import _nn_eval_pool, _nn_eval_pool_lock, MODEL_PATH
    from nn_evaluator import NNEvaluator

    with _nn_eval_pool_lock:
        current_pool_size = len(_nn_eval_pool)
        # We need (num_workers - 1) in pool because get_nn_evaluator_from_pool
        # creates one from global when pool is empty
        needed = num_workers - current_pool_size
        if needed > 0:
            _mp_diag_print(f"Pre-populating NN evaluator pool with {needed} evaluators...")
            for _ in range(needed):
                evaluator = NNEvaluator.create(CachedBoard(), "NNUE", MODEL_PATH)
                _nn_eval_pool.append(evaluator)
            _mp_diag_print(f"NN evaluator pool size: {len(_nn_eval_pool)}")

    _mp_diag_print(f"Lazy SMP initialized with {num_workers} threads")


def shutdown_worker_pool():
    """Shutdown signal - stop any running threads and reset state."""
    global _pool_initialized, _num_workers, _worker_threads

    if not _pool_initialized:
        return

    _mp_diag_print("Shutting down Lazy SMP")

    # Signal any running threads to stop
    from chess_engine import _SearchControl
    _SearchControl.generation = 0

    # Wait for any active threads
    for t in _worker_threads:
        if t.is_alive():
            t.join(timeout=1.0)

    _worker_threads = []
    _pool_initialized = False


def _lazy_smp_worker(worker_id: int, fen: str, max_depth: int, generation: int,
                     result_queue: Queue, start_delay: float = 0.0):
    """
    Lazy SMP worker thread.

    Each worker creates its own ChessEngine instance with isolated state,
    then calls engine.find_best_move() with a callback for intermediate
    result reporting.
    """
    from chess_engine import (ChessEngine, _SearchControl, TimeControl,
                              get_nn_evaluator_from_pool, return_nn_evaluator_to_pool)

    # FIX: Stagger worker starts to reduce contention on pool lock and initial TT access
    if start_delay > 0:
        time.sleep(start_delay)
        # Check if search was cancelled during delay
        if _SearchControl.generation != generation:
            return

    # Get an NN evaluator from the pool (should be pre-populated now)
    nn_eval = get_nn_evaluator_from_pool()

    # Create per-worker engine with isolated state
    engine = ChessEngine(
        nn_eval=nn_eval,
        search_generation=_SearchControl,
        generation=generation
    )

    # Set unique eval noise seed for search diversity
    eval_seed = worker_id * 100003 + generation * 1009
    engine.set_eval_noise(5, seed=eval_seed)
    _mp_diag_print(f"Worker {worker_id} eval noise seed: {eval_seed}")

    # FIX #4: Track last reported depth to skip redundant callbacks
    last_reported_depth = 0

    def on_depth_complete(depth, move_int, score, pv_int, nodes):
        """Callback: report only when depth improves."""
        nonlocal last_reported_depth

        # Skip if this depth was already reported
        if depth <= last_reported_depth:
            return

        # Check if this search is still active
        if _SearchControl.generation != generation:
            return

        last_reported_depth = depth

        # Avoid list() copy if already a list
        pv_copy = pv_int if isinstance(pv_int, list) else list(pv_int)
        result = (worker_id, move_int, score, pv_copy, nodes, depth, generation)
        result_queue.put(result)
        _mp_diag_print(f"Worker {worker_id} completed depth {depth}: "
                       f"{int_to_uci(move_int)} score={score}")

    try:
        engine.find_best_move(
            fen,
            max_depth=max_depth,
            clear_tt=False,
            use_existing_time_control=True,
            on_depth_complete=on_depth_complete,
            suppress_info=True
        )
    except Exception as e:
        import traceback
        print(f"info string Worker {worker_id} error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    # Return NN evaluator to pool for reuse
    return_nn_evaluator_to_pool(nn_eval)


def parallel_find_best_move(fen: str, max_depth: int = 20, time_limit: Optional[float] = None,
                            clear_tt: bool = True,
                            use_existing_time_control: bool = False) -> Tuple[int, int, List[int], int, float]:
    """
    Find best move using Lazy SMP parallel search.

    Returns:
        Tuple of (best_move_int, score, pv_int, nodes, nps)
    """
    global _worker_threads

    # Fall back to single-threaded if not initialized
    if not _pool_initialized or _num_workers <= 1:
        from chess_engine import find_best_move
        return find_best_move(fen, max_depth=max_depth, time_limit=time_limit,
                              clear_tt=clear_tt,
                              use_existing_time_control=use_existing_time_control)

    from chess_engine import (_SearchControl, TimeControl, transposition_table,
                              qs_transposition_table, nn_eval_cache, control_dict_size,
                              pv_int_to_uci, validate_pv_int)

    start_time = time.perf_counter()

    # Reset TimeControl for new search (unless caller manages it)
    if not use_existing_time_control:
        TimeControl.stop_search = False
        TimeControl.start_time = start_time
        TimeControl.time_limit = time_limit
        if time_limit:
            grace_period = max(time_limit * 0.5, 0.3)
            TimeControl.hard_stop_time = start_time + time_limit + grace_period
        else:
            TimeControl.hard_stop_time = None
    else:
        _mp_diag_print(f"Using existing TimeControl (time_limit={TimeControl.time_limit})")
        start_time = TimeControl.start_time or start_time

    # Increment generation to signal new search
    new_gen = _SearchControl.generation + 1
    if new_gen <= 0:
        new_gen = 1
    _SearchControl.generation = new_gen
    current_generation = new_gen

    # Clear TT if requested (main thread only â€” before spawning workers)
    if clear_tt:
        transposition_table.clear()
        qs_transposition_table.clear()

    # Trim nn_eval_cache in main thread (not thread-safe for concurrent access)
    control_dict_size(nn_eval_cache, config.MAX_TABLE_SIZE)

    # Get legal moves to validate results later
    board = CachedBoard(fen)
    legal_moves_int = board.get_legal_moves_int()

    if not legal_moves_int:
        return 0, 0, [], 0, 0  # Null move

    # Single legal move - no need to spawn workers
    if len(legal_moves_int) == 1:
        _mp_diag_print("Single legal move, returning immediately")
        return legal_moves_int[0], 0, [legal_moves_int[0]], 0, 0

    _mp_diag_print(f"Lazy SMP search gen={current_generation} with {_num_workers} threads")

    # Clear result queue
    while not _result_queue.empty():
        try:
            _result_queue.get_nowait()
        except Empty:
            break

    # Start worker threads with staggered delays to reduce contention
    _worker_threads = []
    for i in range(_num_workers):
        # Stagger: worker 0 starts immediately, worker 1 after 10ms, etc.
        start_delay = i * 0.01  # 10ms stagger per worker
        t = threading.Thread(
            target=_lazy_smp_worker,
            args=(i, fen, max_depth, current_generation, _result_queue, start_delay),
            daemon=True
        )
        t.start()
        _worker_threads.append(t)

    # Collect results
    best_result = None  # (move_int, score, pv, nodes, depth)
    worker_nodes = {}
    workers_reported = set()

    # Timing for MIN_DEPTH enforcement
    min_wait_until = start_time + MIN_RESULT_WAIT
    hard_timeout = start_time + (time_limit * 2.5 if time_limit else 60.0)
    soft_limit_passed = False

    while True:
        current_time = time.perf_counter()
        elapsed = current_time - start_time
        best_depth = best_result[4] if best_result else 0

        # Check for external stop via generation
        if _SearchControl.generation != current_generation:
            _mp_diag_print("Generation changed, stopping")
            break

        # Check for external stop via TimeControl (UCI 'stop')
        if TimeControl.stop_search:
            _mp_diag_print("TimeControl.stop_search detected")
            if best_depth >= MIN_PREFERRED_DEPTH or current_time >= hard_timeout:
                _mp_diag_print(f"Stopping workers (depth={best_depth})")
                _SearchControl.generation = 0
                break
            elif not soft_limit_passed:
                soft_limit_passed = True
                _mp_diag_print(f"Soft stop at depth {best_depth}, waiting for {MIN_PREFERRED_DEPTH}")

        # Check for time limit (soft limit)
        if time_limit and elapsed >= time_limit:
            if not soft_limit_passed:
                soft_limit_passed = True
                _mp_diag_print(f"Time limit reached at depth {best_depth}")

            if best_depth >= MIN_PREFERRED_DEPTH:
                _mp_diag_print(f"Reached MIN_DEPTH {best_depth}, stopping workers")
                _SearchControl.generation = 0
                break
            elif current_time >= hard_timeout:
                _mp_diag_print(f"Hard timeout at depth {best_depth}, forcing stop")
                _SearchControl.generation = 0
                break
            elif best_result is None and current_time < min_wait_until:
                pass  # Still waiting for first result

        # Hard timeout safety check
        if current_time >= hard_timeout:
            _mp_diag_print(f"Hard timeout reached")
            _SearchControl.generation = 0
            break

        try:
            # Use get_nowait() to minimize lock holding time
            # Then sleep briefly if queue was empty
            result = _result_queue.get_nowait()
            worker_id, move_int, score, pv, nodes, depth, gen = result

            # Ignore stale results
            if gen != current_generation:
                continue

            workers_reported.add(worker_id)
            worker_nodes[worker_id] = max(worker_nodes.get(worker_id, 0), nodes)
            total_nodes = sum(worker_nodes.values())

            # Keep result from highest depth, or better score at same depth
            if best_result is None or depth > best_result[4]:
                best_result = (move_int, score, pv, nodes, depth)
                _mp_diag_print(f"New best from worker {worker_id}: "
                               f"{int_to_uci(move_int)} score={score} depth={depth}")

                # FIX #3: Print basic UCI info without PV validation
                # (PV will be validated once at the end)
                nps = int(total_nodes / elapsed) if elapsed > 0 else 0
                print(f"info depth {depth} score cp {score} nodes {total_nodes} nps {nps}",
                      flush=True)

                # Stop if we've reached MIN_DEPTH after soft limit passed
                if depth >= MIN_PREFERRED_DEPTH and soft_limit_passed:
                    _mp_diag_print(f"Reached MIN_DEPTH {depth} after soft limit, stopping")
                    _SearchControl.generation = 0
                    break

            elif depth == best_result[4] and score > best_result[1]:
                best_result = (move_int, score, pv, nodes, depth)
                _mp_diag_print(f"Better score from worker {worker_id}: "
                               f"{int_to_uci(move_int)} score={score} depth={depth}")

        except Empty:
            # Brief sleep to avoid busy-waiting, but short enough to be responsive
            time.sleep(0.005)  # 5ms

            # Check if all threads have finished
            all_done = all(not t.is_alive() for t in _worker_threads)
            if all_done:
                _mp_diag_print("All workers finished")
                break

    # Ensure all threads stop
    _SearchControl.generation = 0
    for t in _worker_threads:
        t.join(timeout=0.5)

    # Drain any remaining results from queue
    while not _result_queue.empty():
        try:
            result = _result_queue.get_nowait()
            worker_id, move_int, score, pv, nodes, depth, gen = result
            if gen != current_generation:
                continue
            total_nodes = max(total_nodes, nodes)
            if best_result is None or depth > best_result[4]:
                best_result = (move_int, score, pv, nodes, depth)
                _mp_diag_print(f"Drained better result: depth={depth}")
        except Empty:
            break

    # Reset generation to non-zero so next search doesn't start stopped
    _SearchControl.generation = 1

    _mp_diag_print(f"Collected results from {len(workers_reported)} workers")

    # Warn if search was shallow
    if best_result and best_result[4] < MIN_PREFERRED_DEPTH:
        print(f"info string SHALLOW_SEARCH: bestmove selected at depth {best_result[4]} "
              f"(preferred={MIN_PREFERRED_DEPTH})", flush=True)

    # Process results
    if best_result is None:
        _mp_diag_print("No results from workers, using first legal move")
        return legal_moves_int[0], 0, [legal_moves_int[0]], 0, 0

    move_int, score, pv_int, nodes, depth = best_result

    # Validate move is legal
    if move_int not in legal_moves_int:
        _mp_diag_print(f"ERROR: Best move {int_to_uci(move_int)} is illegal!")
        return legal_moves_int[0], 0, [legal_moves_int[0]], 0, 0

    # FIX #3: Validate PV only once at the end
    board_final = CachedBoard(fen)
    best_pv_int = validate_pv_int(board_final, pv_int)

    elapsed = time.perf_counter() - start_time
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0

    # Print final info with validated PV
    pv_uci = ' '.join(int_to_uci(m) for m in best_pv_int)
    print(f"info depth {depth} score cp {score} nodes {total_nodes} nps {nps} pv {pv_uci}",
          flush=True)

    return move_int, score, best_pv_int, total_nodes, nps


def stop_parallel_search():
    """Signal all workers to stop."""
    from chess_engine import _SearchControl
    _SearchControl.generation = 0


def set_lazy_smp_threads(threads: int):
    """Set number of threads and initialize pool."""
    if threads > 1:
        init_worker_pool(threads)
    else:
        shutdown_worker_pool()


def is_mp_enabled() -> bool:
    """Check if multiprocessing is enabled."""
    return _pool_initialized and _num_workers > 1


def main():
    """Test function."""
    set_lazy_smp_threads(config.THREADS)
    time.sleep(0.5)

    try:
        while True:
            fen = input("FEN: ").strip()
            if fen.lower() in ("exit", "quit"):
                break
            if fen == "":
                print("Type 'exit' or 'quit' to quit")
                continue

            move_int, score, pv_int, nodes, nps = parallel_find_best_move(fen, max_depth=20, time_limit=30)

            print(f"nodes: {nodes}")
            print(f"nps: {nps}")
            print(f"Best move: {int_to_uci(move_int)}")
            print(f"Score: {score}")
            print(f"PV: {' '.join(int_to_uci(m) for m in pv_int)}")
            print("-------------------\n")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    finally:
        stop_parallel_search()
        shutdown_worker_pool()


if __name__ == '__main__':
    main()