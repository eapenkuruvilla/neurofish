#!/usr/bin/env python3
"""
Multiprocessing search module for chess engine.

Implements Root Move Splitting:
- Divide root moves among worker processes
- Each worker searches its assigned moves to full depth
- Main process collects results and picks the best

Optional Shared TT:
- When IS_SHARED_TT_MP is True, workers share transposition tables
- Uses multiprocessing.Manager().dict() for simplicity
"""

import time
from multiprocessing import Process, Queue, Event, Value, Manager
from typing import List, Tuple, Optional, Dict

import chess

from cached_board import CachedBoard, int_to_move
from config import MAX_MP_CORES, IS_SHARED_TT_MP
from engine import pv_to_san, is_debug_enabled

# Module-level setting for shared TT (can be changed at runtime)
_use_shared_tt = IS_SHARED_TT_MP


def _mp_diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


# Worker pool (persistent workers)
_worker_pool: List[Process] = []
_work_queues: List[Queue] = []
_result_queue: Optional[Queue] = None
_stop_event: Optional[Event] = None
_shared_alpha: Optional[Value] = None
_manager: Optional[Manager] = None
_shared_tt: Optional[Dict] = None
_shared_qs_tt: Optional[Dict] = None
_shared_dnn_cache: Optional[Dict] = None
_pool_initialized = False


def init_worker_pool(num_workers: int):
    """
    Initialize persistent worker pool.
    Called once at engine startup when Threads > 1.
    """
    global _worker_pool, _work_queues, _result_queue, _stop_event
    global _shared_alpha, _manager, _shared_tt, _shared_qs_tt, _shared_dnn_cache
    global _pool_initialized

    if _pool_initialized:
        shutdown_worker_pool()

    if num_workers <= 1:
        _pool_initialized = False
        return

    _mp_diag_print(f"Initializing {num_workers} worker processes")

    # Create shared resources
    _stop_event = Event()
    _result_queue = Queue()
    _shared_alpha = Value('i', -100000)  # Shared best alpha (int, centipawns)

    # Create shared TT if enabled
    if _use_shared_tt:
        _manager = Manager()
        _shared_tt = _manager.dict()
        _shared_qs_tt = _manager.dict()
        _shared_dnn_cache = _manager.dict()
        _mp_diag_print("Using shared transposition tables")
    else:
        _shared_tt = None
        _shared_qs_tt = None
        _shared_dnn_cache = None
        _mp_diag_print("Using independent transposition tables")

    # Create work queues and workers
    _work_queues = []
    _worker_pool = []

    for i in range(num_workers):
        work_queue = Queue()
        _work_queues.append(work_queue)

        p = Process(
            target=_worker_main,
            args=(i, work_queue, _result_queue, _stop_event, _shared_alpha,
                  _shared_tt, _shared_qs_tt, _shared_dnn_cache),
            daemon=True
        )
        p.start()
        _worker_pool.append(p)

    _pool_initialized = True
    _mp_diag_print(f"Worker pool initialized with {num_workers} workers")


def shutdown_worker_pool():
    """Shutdown all workers cleanly."""
    global _worker_pool, _work_queues, _result_queue, _stop_event
    global _manager, _pool_initialized, _shared_tt, _shared_qs_tt, _shared_dnn_cache

    if not _pool_initialized:
        return

    _mp_diag_print("Shutting down worker pool")

    # Signal workers to stop searching
    if _stop_event:
        _stop_event.set()

    # Send shutdown command to all workers
    for wq in _work_queues:
        try:
            wq.put_nowait(None)  # None signals shutdown - use put_nowait to avoid blocking
        except:
            pass

    # Wait for workers to finish with shorter timeout
    for i, p in enumerate(_worker_pool):
        try:
            p.join(timeout=1.0)
            if p.is_alive():
                _mp_diag_print(f"Worker {i} didn't stop gracefully, terminating")
                p.terminate()
                p.join(timeout=0.5)  # Brief wait after terminate
                if p.is_alive():
                    _mp_diag_print(f"Worker {i} still alive after terminate, killing")
                    p.kill()  # Force kill if terminate didn't work
        except Exception as e:
            _mp_diag_print(f"Error shutting down worker {i}: {e}")

    # Clear shared resources BEFORE shutting down manager
    _shared_tt = None
    _shared_qs_tt = None
    _shared_dnn_cache = None

    # Cleanup manager (do this after workers are stopped)
    if _manager:
        try:
            _manager.shutdown()
        except:
            pass
        _manager = None

    # Clear queues
    _worker_pool = []
    _work_queues = []
    _result_queue = None
    _stop_event = None

    _pool_initialized = False
    _mp_diag_print("Worker pool shutdown complete")


def _worker_main(worker_id: int, work_queue: Queue, result_queue: Queue,
                 stop_event: Event, shared_alpha: Value,
                 shared_tt: Optional[Dict], shared_qs_tt: Optional[Dict],
                 shared_dnn_cache: Optional[Dict]):
    """
    Main function for each worker process.
    """
    import threading
    import engine

    print(f"info string Worker {worker_id} started", flush=True)

    # Replace engine's tables with shared tables if provided
    if shared_tt is not None:
        engine.transposition_table = shared_tt
    if shared_qs_tt is not None:
        engine.qs_transposition_table = shared_qs_tt
    if shared_dnn_cache is not None:
        engine.dnn_eval_cache = shared_dnn_cache

    # Monitor thread to propagate stop_event to TimeControl
    # Runs continuously, doesn't exit
    def stop_monitor():
        while True:
            if stop_event.is_set():
                engine.TimeControl.stop_search = True
            time.sleep(0.02)  # Check every 20ms

    monitor_thread = threading.Thread(target=stop_monitor, daemon=True)
    monitor_thread.start()

    while True:
        try:
            # Wait for work
            work = work_queue.get()

            if work is None:
                break

            fen, moves_to_search, depth, time_limit, initial_alpha, search_start_time = work

            result = _search_moves(engine, worker_id, fen, moves_to_search, depth, time_limit, initial_alpha,
                                   stop_event, shared_alpha, search_start_time)

            result_queue.put((worker_id, result))

        except Exception as e:
            import traceback
            print(f"info string Worker {worker_id} error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            result_queue.put((worker_id, None))

    print(f"info string Worker {worker_id} stopped", flush=True)


def _search_moves(engine, worker_id: int, fen: str, moves: List[chess.Move], max_depth: int,
                  time_limit: Optional[float], initial_alpha: int, stop_event: Event, shared_alpha: Value,
                  search_start_time: float) -> Optional[
    Tuple]:
    """
    Search a list of root moves using iterative deepening.
    Returns (best_move, best_score, best_pv, nodes) or None if stopped.
    """
    from config import MAX_SCORE
    from cached_board import CachedBoard

    if not moves:
        return None

    board = CachedBoard(fen)
    engine.nn_evaluator.reset(board)

    # Initialize time control for this worker using SHARED start time
    # This ensures all workers measure time from the same reference point
    engine.TimeControl.time_limit = time_limit
    engine.TimeControl.start_time = search_start_time  # Use coordinator's start time, not local time
    engine.TimeControl.stop_search = False
    engine.TimeControl.soft_stop = False

    # Set hard_stop_time consistent with single-threaded search
    if time_limit:
        grace_period = max(time_limit * 0.5, 0.3)  # At least 0.3s or 50% extra
        engine.TimeControl.hard_stop_time = search_start_time + time_limit + grace_period
    else:
        engine.TimeControl.hard_stop_time = None

    # Clear local heuristics (not shared)
    for i in range(len(engine.killer_moves)):
        engine.killer_moves[i] = [None, None]
    engine.history_heuristic.fill(0)

    # Reset node counter for this worker
    engine.kpi['nodes'] = 0

    best_move = moves[0]
    best_score = -MAX_SCORE
    best_pv = [moves[0]]
    depth_reached = 0

    try:
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if stop_event.is_set():
                break

            engine.check_time()
            if engine.TimeControl.stop_search or engine.TimeControl.soft_stop:
                break

            depth_best_move = None
            depth_best_score = -MAX_SCORE
            depth_best_pv = []

            alpha = -MAX_SCORE
            beta = MAX_SCORE

            for move_idx, move in enumerate(moves):
                if stop_event.is_set():
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
                    depth_best_pv = [move] + [int_to_move(m) for m in child_pv]

                if score > alpha:
                    alpha = score
                    if score > shared_alpha.value:
                        shared_alpha.value = score

            # Save completed depth results
            if depth_best_move is not None:
                best_move = depth_best_move
                best_score = depth_best_score
                best_pv = depth_best_pv
                depth_reached = depth

    except TimeoutError:
        pass

    total_nodes = engine.kpi['nodes']

    return (best_move, best_score, best_pv, total_nodes, depth_reached)


def parallel_find_best_move(fen: str, max_depth: int = 20, time_limit: Optional[float] = None, clear_tt: bool = True) -> \
        Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Find best move using parallel root move splitting.

    If MAX_MP_CORES <= 1 or pool not initialized, falls back to single-threaded search.

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _shared_alpha

    # Fall back to single-threaded if MP disabled
    if MAX_MP_CORES <= 1 or not _pool_initialized:
        import engine
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    from cached_board import CachedBoard
    from config import MAX_SCORE
    import engine

    start_time = time.perf_counter()

    # Clear shared tables if requested
    if clear_tt:
        if _shared_tt is not None:
            _shared_tt.clear()
        if _shared_qs_tt is not None:
            _shared_qs_tt.clear()
        # Keep dnn_cache (evaluations are always valid)

    # Reset stop event
    _stop_event.clear()

    # Reset shared alpha
    _shared_alpha.value = -MAX_SCORE

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
    num_workers = min(len(_worker_pool), len(ordered))
    move_assignments = [[] for _ in range(num_workers)]

    for i, move in enumerate(ordered):
        worker_idx = i % num_workers
        move_assignments[worker_idx].append(move)

    # Send work to workers
    initial_alpha = -MAX_SCORE
    for i in range(num_workers):
        if move_assignments[i]:
            # Pass start_time so workers can track time from the same reference point
            work = (fen, move_assignments[i], max_depth, time_limit, initial_alpha, start_time)
            _work_queues[i].put(work)

    # Collect results
    results = []
    workers_done = 0
    expected_workers = sum(1 for m in move_assignments if m)

    while workers_done < expected_workers:
        try:
            worker_id, result = _result_queue.get(timeout=0.5)
            workers_done += 1
            if result is not None:
                results.append(result)
                _mp_diag_print(f"Collected result from worker {worker_id}: {result[0].uci()} "
                               f"score={result[1]} depth={result[4]}")
        except:
            # Check for timeout
            if time_limit and (time.perf_counter() - start_time) >= time_limit:
                _mp_diag_print("Time limit reached, stopping workers")
                _stop_event.set()

                # IMPORTANT: Wait for workers to finish and report (up to 0.5 seconds)
                # FIXED: Reduced from 5 seconds to prevent time overruns
                grace_period = 0.5
                grace_start = time.perf_counter()

                while workers_done < expected_workers:
                    remaining = grace_period - (time.perf_counter() - grace_start)
                    if remaining <= 0:
                        _mp_diag_print(f"Grace period expired, {expected_workers - workers_done} workers didn't report")
                        break
                    try:
                        worker_id, result = _result_queue.get(timeout=min(0.5, remaining))
                        workers_done += 1
                        if result is not None:
                            results.append(result)
                            _mp_diag_print(f"Collected result from worker {worker_id}: {result[0].uci()} "
                                           f"score={result[1]} depth={result[4]}")
                    except:
                        pass  # Keep waiting

                break  # Exit main loop after grace period

    if workers_done < expected_workers:
        _mp_diag_print(f"Not all workers responded, reported={workers_done}, expected={expected_workers}")

    # Pick best result (highest score)
    if not results:
        # Fallback - shouldn't happen
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    best_result = max(results, key=lambda r: r[1])  # r[1] is score
    best_move, best_score, best_pv, _, _ = best_result

    # Calculate total nodes and NPS
    total_nodes = sum(r[3] for r in results)
    elapsed = time.perf_counter() - start_time
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0

    return (best_move, best_score, best_pv, total_nodes, nps)


def stop_parallel_search():
    """Signal all workers to stop their current search."""
    if _stop_event:
        _stop_event.set()


def set_mp_cores(cores: int):
    """Set number of MP cores and reinitialize pool if needed."""
    if cores > MAX_MP_CORES:
        cores = MAX_MP_CORES

    if cores > 1:
        init_worker_pool(cores)
    else:
        shutdown_worker_pool()


def set_shared_tt(enabled: bool):
    """Set whether to use shared TT. Requires pool reinitialization only if setting changed."""
    global _use_shared_tt

    # Only reinitialize if the setting actually changed
    if enabled != _use_shared_tt:
        _use_shared_tt = enabled
        # Reinitialize pool with new setting if pool is active
        if _pool_initialized:
            current_workers = len(_worker_pool)
            init_worker_pool(current_workers)


def is_mp_enabled() -> bool:
    """Check if multiprocessing is enabled and pool is ready."""
    return MAX_MP_CORES > 1 and _pool_initialized


def clear_shared_tables():
    """Clear all shared tables (call on ucinewgame)."""
    if _shared_tt is not None:
        _shared_tt.clear()
    if _shared_qs_tt is not None:
        _shared_qs_tt.clear()
    if _shared_dnn_cache is not None:
        _shared_dnn_cache.clear()


def main():
    """
    Interactive loop to input FEN positions and get the best move and evaluation.
    Tracks KPIs and handles timeouts and interruptions gracefully.
    """
    set_mp_cores(MAX_MP_CORES)
    set_shared_tt(IS_SHARED_TT_MP)
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
                clear_shared_tables()
                move, score, pv, total_nodes, nps = parallel_find_best_move(fen, max_depth=20, time_limit=30)

                # Print KPIs
                print(f"nodes: {total_nodes}")
                print(f"nps: {nps}")

                board = CachedBoard(fen)
                print("\nBest move:", move)
                print("Evaluation:", score)
                print("PV:", pv_to_san(board, pv))
                print("PV (UCI):", " ".join(m.uci() for m in pv))
                print("-------------------\n")

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt detected. Exiting ...")
                break
            except EOFError:
                # Handle Ctrl+D / EOF
                print("\nEOF detected. Exiting ...")
                break
            except Exception as e:
                print(f"Exception {str(e)}")
                import traceback
                traceback.print_exc()
                # Don't exit on exception - allow user to try another position
                continue
    finally:
        # Always cleanup workers on exit
        print("Shutting down workers...")
        stop_parallel_search()
        shutdown_worker_pool()
        print("Done.")


if __name__ == '__main__':
    main()