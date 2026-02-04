#!/usr/bin/env python3
"""
Multiprocessing search module for chess engine.

Implements Root Move Splitting:
- Divide root moves among worker processes
- Each worker searches its assigned moves to full depth
- Main process collects results and picks the best

Uses shared generation counter for synchronization:
- generation > 0: Valid search generation
- generation = 0: Stop signal (all workers should stop)
- Workers check generation every 10ms (lock-free read)

This eliminates the need for separate stop_event and reduces lock overhead.
"""

import time
from multiprocessing import Process, Queue, Value
from typing import List, Tuple, Optional
from queue import Empty

import chess

from cached_board import CachedBoard, int_to_move, move_to_int
from config import MAX_MP_CORES


def _mp_diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    from engine import is_debug_enabled
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


# Worker pool (persistent workers)
_worker_pool: List[Process] = []
_work_queues: List[Queue] = []
_result_queue: Optional[Queue] = None
_pool_initialized = False

# Shared generation counter (lock=False for fast reads by workers)
# Only main thread writes, workers only read - safe without lock
# Values: 0 = stop signal, >0 = valid search generation
_search_generation: Optional[Value] = None


def init_worker_pool(num_workers: int):
    """
    Initialize persistent worker pool.
    Called once at engine startup when Threads > 1.
    """
    global _worker_pool, _work_queues, _result_queue
    global _pool_initialized, _search_generation

    if _pool_initialized:
        shutdown_worker_pool()

    if num_workers <= 1:
        _pool_initialized = False
        return

    _mp_diag_print(f"Initializing {num_workers} worker processes")

    # Create shared resources
    _result_queue = Queue()

    # Shared generation counter - no lock needed (single writer, multiple readers)
    # Value 0 means "stop", values > 0 are valid search generations
    # Start at 1 to avoid immediate stop signal on startup
    _search_generation = Value('i', 1, lock=False)

    # Create work queues and workers
    _work_queues = []
    _worker_pool = []

    for i in range(num_workers):
        work_queue = Queue()
        _work_queues.append(work_queue)

        p = Process(
            target=_worker_main,
            args=(i, work_queue, _result_queue, _search_generation),
            daemon=True
        )
        p.start()
        _worker_pool.append(p)

    _pool_initialized = True
    _mp_diag_print(f"Worker pool initialized with {num_workers} workers")


def shutdown_worker_pool():
    """Shutdown all workers cleanly."""
    global _worker_pool, _work_queues, _result_queue
    global _pool_initialized, _search_generation

    if not _pool_initialized:
        return

    _mp_diag_print("Shutting down worker pool")

    # Signal workers to stop via generation
    if _search_generation:
        _search_generation.value = 0

    # Send shutdown command to all workers
    for wq in _work_queues:
        try:
            wq.put(None)  # None signals shutdown
        except:
            pass

    # Wait for workers to finish
    for p in _worker_pool:
        try:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        except:
            pass

    # Cleanup
    _worker_pool = []
    _work_queues = []

    _pool_initialized = False


def _worker_main(worker_id: int, work_queue: Queue, result_queue: Queue,
                 search_generation: Value):
    """
    Main function for each worker process.

    Uses search_generation for all synchronization:
    - generation = 0: stop signal
    - generation changed: current work is stale
    """
    import threading
    import engine

    print(f"info string Worker {worker_id} started", flush=True)

    # Track current generation this worker is processing
    # Using a list as a mutable container accessible from nested function
    current_gen_holder = [None]

    # Monitor thread to propagate stop signals to TimeControl
    # Checks generation for both stop (0) and staleness (changed)
    def stop_monitor():
        while True:
            should_stop = False
            gen_value = search_generation.value

            # Check for explicit stop signal (generation = 0)
            if gen_value == 0:
                should_stop = True

            # Check if our current search is stale (generation changed)
            current_gen = current_gen_holder[0]
            if current_gen is not None and current_gen != gen_value:
                should_stop = True

            if should_stop:
                engine.TimeControl.stop_search = True

            time.sleep(0.01)  # Check every 10ms

    monitor_thread = threading.Thread(target=stop_monitor, daemon=True)
    monitor_thread.start()

    while True:
        try:
            # Wait for work
            work = work_queue.get()

            if work is None:
                break

            # Extract work parameters including generation
            fen, moves_to_search, depth, time_limit, generation = work

            # Check if work is already stale before starting
            if generation != search_generation.value:
                _mp_diag_print(
                    f"Worker {worker_id} skipping stale work (gen {generation} != {search_generation.value})")
                continue

            # Update current generation so monitor thread knows what we're working on
            current_gen_holder[0] = generation

            # Reset stop flag before starting new search
            engine.TimeControl.stop_search = False

            # Pass result_queue and generation for per-depth reporting
            result = _search_moves(engine, worker_id, fen, moves_to_search, depth, time_limit,
                                   result_queue, generation, search_generation)

            # Only report final result if still current generation
            if generation == search_generation.value:
                result_queue.put((worker_id, result, generation))

        except Exception as e:
            import traceback
            print(f"info string Worker {worker_id} error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            result_queue.put((worker_id, None, 0))  # Generation 0 for errors

    print(f"info string Worker {worker_id} stopped", flush=True)


def _search_moves(engine, worker_id: int, fen: str, moves: List[chess.Move], max_depth: int,
                  time_limit: Optional[float],
                  result_queue: Queue = None, generation: int = 0,
                  search_generation: Value = None) -> Optional[Tuple]:
    """
    Search a list of root moves using iterative deepening.
    Reports results after each completed depth via result_queue.
    Returns (best_move, best_score, best_pv, nodes, depth) or None if stopped.
    """
    from config import MAX_SCORE
    from cached_board import CachedBoard

    if not moves:
        return None

    board = CachedBoard(fen)
    engine.nn_evaluator.reset(board)

    # Initialize time control for this worker
    engine.TimeControl.time_limit = time_limit
    engine.TimeControl.start_time = time.perf_counter()
    engine.TimeControl.stop_search = False
    engine.TimeControl.soft_stop = False

    # Clear local heuristics (not shared)
    for i in range(len(engine.killer_moves)):
        engine.killer_moves[i] = [None, None]
    engine.history_heuristic.fill(0)  # numpy array - use fill() not clear()

    # Reset node counter for this worker
    engine.kpi['nodes'] = 0

    best_move = moves[0]
    best_score = -MAX_SCORE
    best_pv = [moves[0]]
    depth_reached = 0

    def should_stop() -> bool:
        """Check if this search should stop (generation 0 or stale)."""
        if search_generation is None:
            return False
        gen_value = search_generation.value
        return gen_value == 0 or generation != gen_value

    try:
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if should_stop():
                break

            engine.check_time()
            if engine.TimeControl.stop_search or engine.TimeControl.soft_stop:
                break

            depth_best_move = None
            depth_best_score = -MAX_SCORE
            depth_best_pv = []
            depth_completed = False  # Track if this depth completed successfully

            alpha = -MAX_SCORE
            beta = MAX_SCORE

            for move_idx, move in enumerate(moves):
                if should_stop():
                    break

                engine.check_time()
                if engine.TimeControl.stop_search:
                    break

                engine.push_move(board, move, engine.nn_evaluator)

                try:
                    # Check for draw
                    if engine.is_draw_by_repetition(board):
                        score = engine.get_draw_score(board)
                        child_pv = []
                    else:
                        # Use PVS: full window for first move, null window for rest
                        if move_idx == 0:
                            score, child_pv = engine.negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                            score = -score
                        else:
                            # Null window search
                            score, child_pv = engine.negamax(board, depth - 1, -alpha - 1, -alpha, allow_singular=True)
                            score = -score
                            # Re-search with full window if it might be better
                            if score > alpha:
                                score, child_pv = engine.negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                                score = -score
                finally:
                    board.pop()
                    engine.nn_evaluator.pop()

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                    # Convert child_pv integers to chess.Move objects
                    depth_best_pv = [move] + [int_to_move(m) for m in child_pv if m != 0]

                if score > alpha:
                    alpha = score

            # Check if depth was fully completed (all moves searched without abort)
            if (depth_best_move is not None and
                    not engine.TimeControl.stop_search and
                    not should_stop()):
                depth_completed = True

            # CRITICAL: Only save and report COMPLETED depth results
            # This matches engine.py: "Don't overwrite a completed depth's result with an incomplete one"
            if depth_completed:
                best_move = depth_best_move
                best_score = depth_best_score
                best_pv = depth_best_pv
                depth_reached = depth

                # Report completed depth result immediately
                # Only report if still current generation
                if result_queue is not None and not should_stop():
                    result = (best_move, best_score, best_pv, engine.kpi['nodes'], depth_reached)
                    result_queue.put((worker_id, result, generation))

    except TimeoutError:
        pass

    total_nodes = engine.kpi['nodes']

    # Return final result (best completed depth)
    return (best_move, best_score, best_pv, total_nodes, depth_reached)


def parallel_find_best_move(fen: str, max_depth: int = 20, time_limit: Optional[float] = None, clear_tt: bool = True) -> \
        Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Find best move using parallel root move splitting.

    If MAX_MP_CORES <= 1 or pool not initialized, falls back to single-threaded search.

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _search_generation

    # Fall back to single-threaded if MP disabled
    if MAX_MP_CORES <= 1 or not _pool_initialized:
        import engine
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    from cached_board import CachedBoard
    from config import MAX_SCORE
    import engine

    start_time = time.perf_counter()

    # Increment generation - this signals workers that previous work is stale
    # Generation must be > 0 (0 is the stop signal)
    new_gen = _search_generation.value + 1
    if new_gen <= 0:  # Handle wraparound
        new_gen = 1
    _search_generation.value = new_gen
    current_generation = new_gen

    # Clear TT if requested (engine's local tables)
    if clear_tt:
        engine.transposition_table.clear()
        engine.qs_transposition_table.clear()
        # Keep dnn_cache (evaluations are always valid)

    # Get legal moves
    board = CachedBoard(fen)
    legal_moves = list(board.get_legal_moves_list())

    if not legal_moves:
        return (chess.Move.null(), 0, [], 0, 0)

    if len(legal_moves) == 1:
        # Only one legal move - no need for parallel search
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    # Order moves for better distribution
    ordered_int = engine.ordered_moves_int(board, max_depth, pv_move_int=0, tt_move_int=0)
    ordered = [int_to_move(m) for m in ordered_int]

    # Distribute moves among workers
    # With per-depth reporting, we can use all workers effectively
    # Each worker reports results after each completed depth, so no work is lost
    num_workers = min(len(_worker_pool), len(ordered))
    move_assignments = [[] for _ in range(num_workers)]

    for i, move in enumerate(ordered):
        worker_idx = i % num_workers
        move_assignments[worker_idx].append(move)

    _mp_diag_print(f"Search gen={current_generation} FEN: {fen[:60]}...")
    _mp_diag_print(f"Legal moves: {len(ordered)}, distributing to {num_workers} workers")

    # Drain any stale results from previous searches before sending new work
    try:
        while True:
            _result_queue.get_nowait()
    except Empty:
        pass

    # Send work to workers (include generation to identify stale results)
    # Workers report after EACH completed depth, so we get results progressively
    # Give workers slightly less time so they stop before the main timeout
    worker_time_limit = time_limit * 0.85 if time_limit else None

    for i in range(num_workers):
        if move_assignments[i]:
            work = (fen, move_assignments[i], max_depth, worker_time_limit, current_generation)
            _work_queues[i].put(work)

    # Collect results - workers now report after EACH completed depth
    # We keep the best (highest depth) result from each worker
    worker_best_results = {}  # worker_id -> (result, depth)
    expected_workers = sum(1 for m in move_assignments if m)

    # Collect results until time limit
    while True:
        # Check if we've been asked to stop externally (generation set to 0)
        if _search_generation.value == 0:
            _mp_diag_print("Stop signal detected during result collection")
            break

        try:
            worker_id, result, result_generation = _result_queue.get(timeout=0.1)

            # CRITICAL: Ignore stale results from previous searches
            if result_generation != current_generation:
                _mp_diag_print(
                    f"Ignoring stale result from worker {worker_id} (gen {result_generation} != {current_generation})")
                continue

            if result is not None:
                result_depth = result[4]  # depth_reached is at index 4

                # Keep the result with highest depth from each worker
                if worker_id not in worker_best_results or result_depth > worker_best_results[worker_id][1]:
                    worker_best_results[worker_id] = (result, result_depth)
                    _mp_diag_print(
                        f"Worker {worker_id} reported: {result[0].uci()} score={result[1]} depth={result_depth}")

        except Empty:
            # Check for timeout
            if time_limit and (time.perf_counter() - start_time) >= time_limit:
                _mp_diag_print("Time limit reached, stopping workers")
                _search_generation.value = 0  # Signal all workers to stop
                break

    # Convert worker_best_results to results list
    results = [r[0] for r in worker_best_results.values()]

    _mp_diag_print(f"Collected results from {len(results)} workers out of {expected_workers}")

    # Pick best result (highest score)
    if not results:
        # Return first legal move as fallback instead of recursive call
        _mp_diag_print("No results from workers, using first legal move as fallback")
        return (ordered[0], 0, [ordered[0]], 0, 0)

    best_result = max(results, key=lambda r: r[1])  # r[1] is score
    best_move, best_score, best_pv, _, _ = best_result

    # CRITICAL: Validate that the best move is actually legal in the current position
    # This catches any bugs where stale results slip through
    if best_move not in legal_moves:
        _mp_diag_print(f"ERROR: Best move {best_move.uci()} is illegal! Falling back to first legal move.")
        # Find a valid move from results or use first legal
        for r in sorted(results, key=lambda x: x[1], reverse=True):
            if r[0] in legal_moves:
                best_move, best_score, best_pv, _, _ = r
                _mp_diag_print(f"Using fallback move {best_move.uci()} with score {best_score}")
                break
        else:
            # No valid results, use first legal move
            best_move = ordered[0]
            best_score = 0
            best_pv = [ordered[0]]

    # Calculate total nodes and NPS
    total_nodes = sum(r[3] for r in results)
    elapsed = time.perf_counter() - start_time
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0

    return (best_move, best_score, best_pv, total_nodes, nps)


def stop_parallel_search():
    """Signal all workers to stop their current search."""
    if _search_generation:
        _search_generation.value = 0


def set_mp_cores(cores: int):
    """Set number of MP cores and reinitialize pool if needed."""
    global MAX_MP_CORES

    # Update the module-level MAX_MP_CORES
    import config
    if cores > config.MAX_MP_CORES:
        cores = config.MAX_MP_CORES

    if cores > 1:
        init_worker_pool(cores)
    else:
        shutdown_worker_pool()


def is_mp_enabled() -> bool:
    """Check if multiprocessing is enabled and pool is ready."""
    return _pool_initialized and len(_worker_pool) > 0


def main():
    """
    Interactive loop to input FEN positions and get the best move and evaluation.
    Tracks KPIs and handles timeouts and interruptions gracefully.
    """
    set_mp_cores(MAX_MP_CORES)
    time.sleep(1)  # Wait for workers to start

    try:
        while True:
            try:
                fen = input("FEN: ").strip()
                if fen.lower() in ("exit", "quit"):
                    break
                if fen == "":
                    print("Type 'exit' or 'quit' to quit")
                    continue

                # Start timer
                move, score, pv, total_nodes, nps = parallel_find_best_move(fen, max_depth=20, time_limit=30)

                # Print KPIs
                print(f"nodes: {total_nodes}")
                print(f"nps: {nps}")

                board = CachedBoard(fen)
                from engine import pv_to_san
                print("\nBest move:", move)
                print("Evaluation:", score)
                print("PV:", pv_to_san(board, pv))
                print("PV (UCI):", " ".join(m.uci() for m in pv))
                print("-------------------\n")

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected. Exiting ...")
                break
            except EOFError:
                print("\nEOF detected. Exiting ...")
                break
            except Exception as e:
                print(f"Exception {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue instead of exit - allow user to try another position
                continue
    finally:
        # Always cleanup workers on exit
        print("Cleaning up workers...")
        stop_parallel_search()
        shutdown_worker_pool()
        print("Done.")


if __name__ == '__main__':
    main()