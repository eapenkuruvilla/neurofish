#!/usr/bin/env python3
"""
Multiprocessing search module for chess engine.

Implements Coordinated Iterative Deepening with Root Move Splitting:
- Main process controls depth iteration
- For each depth: distribute root moves to workers, collect results, check time
- All workers search the same depth simultaneously
- Time management decisions happen at coordinator level

Optional Shared TT:
- When IS_SHARED_TT_MP is True, workers share transposition tables
- Uses multiprocessing.Manager().dict() for simplicity
"""

import time
from multiprocessing import Process, Queue, Event, Value, Manager
from typing import List, Tuple, Optional, Dict

import chess

from cached_board import CachedBoard, int_to_move, move_to_int
from config import MAX_MP_CORES, IS_SHARED_TT_MP, MIN_NEGAMAX_DEPTH, TIME_SAFETY_MARGIN_RATIO, \
    ESTIMATED_BRANCHING_FACTOR
from engine import pv_to_san, is_debug_enabled

# Module-level setting for shared TT (can be changed at runtime)
_use_shared_tt = IS_SHARED_TT_MP


def _mp_diag_print(msg: str):
    """Print diagnostic info string only when diagnostics are enabled."""
    if is_debug_enabled():
        print(f"info string {msg}", flush=True)


def _validate_pv(fen: str, pv: List[chess.Move]) -> List[chess.Move]:
    """
    Validate PV moves are legal. Returns truncated PV at first illegal move.
    This catches TT corruption or other bugs that produce invalid PVs.
    """
    if not pv:
        return pv

    board = CachedBoard(fen)
    valid_pv = []

    for move in pv:
        if move in board.get_legal_moves_list():
            valid_pv.append(move)
            board.push(move)
        else:
            # Stop at first illegal move
            _mp_diag_print(f"PV validation: illegal move {move.uci()} at position, truncating")
            break

    return valid_pv


# Worker pool (persistent workers)
_worker_pool: List[Process] = []
_work_queues: List[Queue] = []
_result_queue: Optional[Queue] = None
_stop_event: Optional[Event] = None
_shared_alpha: Optional[Value] = None

# Shared tables (optional)
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
    Workers receive single-depth search tasks and return results.
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

            # Unpack work tuple for single-depth search
            fen, moves_to_search, depth, time_limit, search_start_time, pv_move_int = work

            result = _search_moves_single_depth(
                engine, worker_id, fen, moves_to_search, depth,
                time_limit, search_start_time, pv_move_int,
                stop_event, shared_alpha
            )

            result_queue.put((worker_id, result))

        except Exception as e:
            import traceback
            print(f"info string Worker {worker_id} error: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            result_queue.put((worker_id, None))

    print(f"info string Worker {worker_id} stopped", flush=True)


def _search_moves_single_depth(
        engine, worker_id: int, fen: str, moves: List[chess.Move], depth: int,
        time_limit: Optional[float], search_start_time: float, pv_move_int: int,
        stop_event: Event, shared_alpha: Value
) -> Optional[Tuple]:
    """
    Search a list of root moves to exactly ONE depth.
    Returns (best_move, best_score, best_pv, nodes) or None if stopped.

    This is the core worker function for coordinated iterative deepening.
    """
    from config import MAX_SCORE
    from cached_board import CachedBoard

    if not moves:
        return None

    board = CachedBoard(fen)
    engine.nn_evaluator.reset(board)

    # Initialize time control using SHARED start time
    engine.TimeControl.time_limit = time_limit
    engine.TimeControl.start_time = search_start_time
    engine.TimeControl.stop_search = False
    engine.TimeControl.soft_stop = False

    if time_limit:
        grace_period = max(time_limit * 0.5, 0.3)
        engine.TimeControl.hard_stop_time = search_start_time + time_limit + grace_period
    else:
        engine.TimeControl.hard_stop_time = None

    # Reset node counter for this search
    engine.kpi['nodes'] = 0

    best_move = moves[0]
    best_score = -MAX_SCORE
    best_pv = [moves[0]]

    alpha = -MAX_SCORE
    beta = MAX_SCORE

    # Sort moves: PV move first if present
    sorted_moves = list(moves)
    if pv_move_int != 0:
        pv_move = int_to_move(pv_move_int)
        if pv_move in sorted_moves:
            sorted_moves.remove(pv_move)
            sorted_moves.insert(0, pv_move)

    for move_idx, move in enumerate(sorted_moves):
        if stop_event.is_set():
            break

        engine.check_time()
        if engine.TimeControl.stop_search:
            break

        # OPTIMIZATION: Read shared alpha from other workers for better pruning
        # This allows workers to benefit from discoveries made by other workers
        current_shared = shared_alpha.value
        if current_shared > alpha:
            alpha = current_shared

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
                    if score > alpha and not stop_event.is_set():
                        score, child_pv = engine.negamax(board, depth - 1, -beta, -alpha, allow_singular=True)
                        score = -score
        finally:
            board.pop()
            engine.nn_evaluator.pop()

        if score > best_score:
            best_score = score
            best_move = move
            best_pv = [move] + [int_to_move(m) for m in child_pv if m != 0]

        if score > alpha:
            alpha = score
            # Update shared alpha for other workers
            if score > shared_alpha.value:
                shared_alpha.value = score

    total_nodes = engine.kpi['nodes']
    return (best_move, best_score, best_pv, total_nodes)


def _collect_results_with_timeout(expected_workers: int, timeout: float) -> List[Tuple]:
    """
    Collect results from workers with a timeout.
    Returns list of results (may be fewer than expected if timeout).
    """
    results = []
    workers_done = 0
    deadline = time.perf_counter() + timeout

    while workers_done < expected_workers:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            worker_id, result = _result_queue.get(timeout=min(0.1, remaining))
            workers_done += 1
            if result is not None:
                results.append(result)
        except:
            pass  # Queue.get timeout

    return results


def parallel_find_best_move(fen: str, max_depth: int = 20, time_limit: Optional[float] = None, clear_tt: bool = True) -> \
        Tuple[Optional[chess.Move], int, List[chess.Move], int, float]:
    """
    Find best move using coordinated iterative deepening with parallel root move splitting.

    The main process controls depth iteration:
    - For each depth, distribute moves to workers
    - Collect results after all workers complete (or timeout)
    - Check time before proceeding to next depth

    Returns:
        Tuple of (best_move, score, pv, nodes, nps)
    """
    global _shared_alpha

    # Fall back to single-threaded if MP disabled
    if MAX_MP_CORES <= 1 or not _pool_initialized:
        import engine
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    from config import MAX_SCORE
    import engine

    start_time = time.perf_counter()

    # Clear shared tables if requested
    if clear_tt:
        if _shared_tt is not None:
            _shared_tt.clear()
        if _shared_qs_tt is not None:
            _shared_qs_tt.clear()

    # Reset stop event
    _stop_event.clear()

    # Get legal moves
    board = CachedBoard(fen)
    legal_moves = list(board.get_legal_moves_list())

    if not legal_moves:
        return (chess.Move.null(), 0, [], 0, 0)

    if len(legal_moves) == 1:
        # Only one legal move - no need for parallel search
        return engine.find_best_move(fen, max_depth=max_depth, time_limit=time_limit, clear_tt=clear_tt)

    # Initial move ordering
    ordered_int = engine.ordered_moves_int(board, max_depth, pv_move_int=0, tt_move_int=0)
    ordered = [int_to_move(m) for m in ordered_int]

    # Track overall best result
    best_move = ordered[0]
    best_score = -MAX_SCORE
    best_pv = [ordered[0]]
    total_nodes = 0
    last_depth_completed = 0
    last_depth_time = 0.0  # Time taken for last completed depth

    num_workers = min(len(_worker_pool), len(ordered))

    # Coordinated iterative deepening with YBWC
    for depth in range(1, max_depth + 1):
        depth_start_time = time.perf_counter()

        # Check if we should stop before starting this depth
        if _stop_event.is_set():
            _mp_diag_print(f"Stop event set before depth {depth}")
            break

        elapsed = time.perf_counter() - start_time

        # Time management: check if we have enough time for this depth
        if time_limit and depth > MIN_NEGAMAX_DEPTH:
            remaining = time_limit - elapsed
            # Estimate time for this depth based on branching factor
            estimated_time = last_depth_time * ESTIMATED_BRANCHING_FACTOR if last_depth_time > 0 else 0.1

            # Don't start new depth if we probably can't finish
            if remaining < estimated_time * TIME_SAFETY_MARGIN_RATIO:
                _mp_diag_print(
                    f"Stopping before depth {depth}: remaining={remaining:.2f}s, estimated={estimated_time:.2f}s")
                break

        # Get PV move from previous depth
        pv_move = best_pv[0] if best_pv else ordered[0]
        pv_move_int = move_to_int(pv_move)

        # ============================================================
        # YBWC Phase 1: Search PV move with worker 0 only
        # ============================================================
        _shared_alpha.value = -MAX_SCORE

        # Send PV move to worker 0
        work = (fen, [pv_move], depth, time_limit, start_time, pv_move_int)
        _work_queues[0].put(work)

        # Wait for worker 0 to complete PV search
        pv_timeout = time_limit - (time.perf_counter() - start_time) + 0.5 if time_limit else 60.0
        pv_results = _collect_results_with_timeout(1, max(0.1, pv_timeout))

        if not pv_results:
            _mp_diag_print(f"No PV result for depth {depth}, aborting")
            break

        # Extract PV result
        pv_result = pv_results[0]
        pv_score = pv_result[1]
        pv_nodes = pv_result[3]

        # Update shared alpha with PV score - this is the key to YBWC!
        _shared_alpha.value = pv_score

        _mp_diag_print(f"Depth {depth} PV phase: {pv_move.uci()} score={pv_score} nodes={pv_nodes}")

        # ============================================================
        # YBWC Phase 2: Search remaining moves with all workers
        # ============================================================
        remaining_moves = [m for m in ordered if m != pv_move]

        if remaining_moves:
            # Distribute remaining moves among ALL workers
            move_assignments = [[] for _ in range(num_workers)]
            for i, move in enumerate(remaining_moves):
                worker_idx = i % num_workers
                move_assignments[worker_idx].append(move)

            # Dispatch to all workers
            workers_used = 0
            for i in range(num_workers):
                if move_assignments[i]:
                    work = (fen, move_assignments[i], depth, time_limit, start_time, pv_move_int)
                    _work_queues[i].put(work)
                    workers_used += 1

            # Collect results from all workers
            if time_limit:
                remaining_time = time_limit - (time.perf_counter() - start_time)
                phase2_timeout = max(0.1, remaining_time + 0.5)
            else:
                phase2_timeout = 300.0

            phase2_results = _collect_results_with_timeout(workers_used, phase2_timeout)

            # Combine PV result with phase 2 results
            all_results = [pv_result] + phase2_results
        else:
            # Only one legal move - PV result is all we need
            all_results = [pv_result]

        depth_time = time.perf_counter() - depth_start_time
        depth_nodes = sum(r[3] for r in all_results)
        total_nodes += depth_nodes

        # Find best result from this depth
        depth_best = max(all_results, key=lambda r: r[1])

        # Check if depth completed successfully
        # For YBWC, we consider it complete if PV finished and we got all phase 2 results
        if remaining_moves:
            expected_phase2 = sum(1 for m in move_assignments if m)
            got_phase2 = len(all_results) - 1  # Subtract PV result
        else:
            expected_phase2 = 0
            got_phase2 = 0

        if got_phase2 >= expected_phase2:
            # Full depth completed
            best_move, best_score, best_pv, _ = depth_best
            # Validate PV to catch TT corruption or other bugs
            best_pv = _validate_pv(fen, best_pv)
            # Ensure best_move matches PV
            if best_pv:
                best_move = best_pv[0]
            last_depth_completed = depth
            last_depth_time = depth_time

            # Print info line for this depth
            elapsed = time.perf_counter() - start_time
            nps = int(total_nodes / elapsed) if elapsed > 0 else 0
            pv_str = ' '.join(m.uci() for m in best_pv)
            print(f"info depth {depth} score cp {best_score} nodes {total_nodes} nps {nps} pv {pv_str}", flush=True)
        else:
            _mp_diag_print(f"Depth {depth} incomplete: got {got_phase2}/{expected_phase2} phase2 results")
            # Still use partial results if they're better
            if depth_best[1] > best_score:
                best_move, best_score, best_pv, _ = depth_best
                # Validate PV
                best_pv = _validate_pv(fen, best_pv)
                if best_pv:
                    best_move = best_pv[0]

        # Check time after depth completion
        if time_limit:
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit:
                _mp_diag_print(f"Time limit reached after depth {depth}")
                break

    # Final statistics
    elapsed = time.perf_counter() - start_time
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0

    _mp_diag_print(f"Search complete: depth={last_depth_completed}, nodes={total_nodes}, time={elapsed:.2f}s")

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
    """
    set_mp_cores(MAX_MP_CORES)
    set_shared_tt(IS_SHARED_TT_MP)
    time.sleep(0.5)  # Wait for workers to start

    try:
        while True:
            try:
                fen = input("FEN: ").strip()
                if fen.lower() in ("exit", "quit"):
                    break
                if fen == "":
                    print("Type 'exit' or 'quit' to quit")
                    continue

                clear_shared_tables()
                move, score, pv, total_nodes, nps = parallel_find_best_move(fen, max_depth=20, time_limit=30)

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
                print("\nEOF detected. Exiting ...")
                break
            except Exception as e:
                print(f"Exception {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    finally:
        print("Shutting down workers...")
        stop_parallel_search()
        shutdown_worker_pool()
        print("Done.")


if __name__ == '__main__':
    main()