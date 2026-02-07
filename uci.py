#!/usr/bin/env python3
import os
import threading
import time
from pathlib import Path
import chess
import chess.polyglot

import config
from chess_engine import (find_best_move, TimeControl, dnn_eval_cache,
                          clear_game_history, game_position_history, kpi,
                          diag_summary, set_debug_mode,
                          is_debug_enabled, diag_print, return_nn_evaluator_to_pool)
from book_move import init_opening_book, get_book_move
import lazy_smp
from test.uci_config_bridge import register_config_tunables, print_uci_options, \
    apply_uci_option

# Resign settings
RESIGN_THRESHOLD = -500  # centipawns
RESIGN_CONSECUTIVE_MOVES = 3  # must be losing for this many moves
resign_counter = 0

# Pondering settings
PONDER_TIME_LIMIT = 600  # Maximum time for ponder search (safety cap)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOOK_PATH = SCRIPT_DIR / f"../{SCRIPT_DIR}" / 'book' / 'komodo.bin'

search_thread = None
use_book = True
use_ponder = True  # UCI Ponder option (can be disabled by GUI)

# Pondering state
is_pondering = False  # True when engine is in ponder mode
ponder_fen = None  # FEN of the position being pondered

# Ponder time tracking - store time info from "go ponder" command
ponder_time_info = None  # Dict with wtime, btime, winc, binc, movestogo
ponder_start_time = None  # Time when ponder search started
ponder_best_move = None  # Best move found during pondering
ponder_best_score = None  # Score of best move during pondering

"""
uci_options.py

Central registry for all UCI-exposed engine parameters.
This allows clean parameter tuning via cutechess/self-play.

Design goals:
- Single source of truth for tunable parameters
- No env vars
- No hard-coded setoption spaghetti
- Safe for self-play (per-engine instance)
"""

def record_position_hash(board: chess.Board):
    """Record position in game history using Zobrist hash."""
    key = chess.polyglot.zobrist_hash(board)
    game_position_history[key] = game_position_history.get(key, 0) + 1


def uci_loop():
    global search_thread, use_book, use_ponder, RESIGN_THRESHOLD, RESIGN_CONSECUTIVE_MOVES, resign_counter
    global is_pondering, ponder_fen
    global ponder_time_info, ponder_start_time, ponder_best_move, ponder_best_score
    board = chess.Board()
    book_path = DEFAULT_BOOK_PATH

    while True:
        try:
            command = input().strip()
        except EOFError:
            break

        if not command:
            continue

        if command == "uci":
            print("id name Neurofish")
            print("id author Eapen Kuruvilla")
            print("option name OwnBook type check default true")
            print("option name BookPath type string default " + str(DEFAULT_BOOK_PATH))
            print(f"option name ResignThreshold type spin default {RESIGN_THRESHOLD} min -10000 max 0")
            print(f"option name ResignMoves type spin default {RESIGN_CONSECUTIVE_MOVES} min 1 max 10")
            print("option name Ponder type check default true")
            print(f"option name Threads type spin default {config.MAX_THREADS} min 1 max 64")

            print_uci_options()

            print("uciok", flush=True)

        elif command == "isready":
            if search_thread and search_thread.is_alive():
                search_thread.join()
            print("readyok", flush=True)

        elif command == "debug on":
            set_debug_mode(True)
            print("info string Debug mode enabled", flush=True)

        elif command == "debug off":
            set_debug_mode(False)
            print("info string Debug mode disabled", flush=True)

        elif command.startswith("setoption"):
            tokens = command.split()
            if "name" in tokens and "value" in tokens:
                name_idx = tokens.index("name") + 1
                value_idx = tokens.index("value") + 1
                name = " ".join(tokens[name_idx:tokens.index("value")])
                value = " ".join(tokens[value_idx:])

                if name.lower() == "ownbook":
                    use_book = value.lower() == "true"
                elif name.lower() == "bookpath":
                    book_path = value
                    init_opening_book(book_path)
                elif name.lower() == "resignthreshold":
                    RESIGN_THRESHOLD = int(value)
                elif name.lower() == "resignmoves":
                    RESIGN_CONSECUTIVE_MOVES = int(value)
                elif name.lower() == "ponder":
                    use_ponder = value.lower() == "true"
                elif name.lower() == "threads":
                    cores = int(value)
                    lazy_smp.set_lazy_smp_threads(cores)
                else:
                    if apply_uci_option(name, value):
                        diag_print(f"Applied tunable option {name}={value}")

        elif command == "ucinewgame":
            # Print diagnostic summary from previous game (if any issues) when debug enabled
            if is_debug_enabled():
                summary = diag_summary()
                if "all clear" not in summary:
                    print(f"info string {summary}", flush=True)

            board.reset()
            dnn_eval_cache.clear()
            clear_game_history()  # Also resets diagnostic counters
            resign_counter = 0
            is_pondering = False
            ponder_fen = None
            ponder_time_info = None
            ponder_start_time = None
            ponder_best_move = None
            ponder_best_score = None
            TimeControl.is_ponder_search = False

        elif command.startswith("position"):
            # Stop any ongoing search (including ponder) before processing new position
            if search_thread and search_thread.is_alive():
                TimeControl.stop_search = True
                lazy_smp.stop_parallel_search()  # FIX: Also stop MP workers
                search_thread.join()
            is_pondering = False
            ponder_fen = None
            ponder_time_info = None
            ponder_start_time = None
            ponder_best_move = None
            ponder_best_score = None
            TimeControl.is_ponder_search = False

            tokens = command.split()
            if len(tokens) < 2:
                continue

            clear_game_history()

            if tokens[1] == "startpos":
                board.reset()
                move_index = 2
            elif tokens[1] == "fen":
                fen = " ".join(tokens[2:8])
                board.set_fen(fen)
                move_index = 8
            else:
                continue

            # Record starting position
            record_position_hash(board)

            # Apply moves and record each position
            if move_index < len(tokens) and tokens[move_index] == "moves":
                for mv in tokens[move_index + 1:]:
                    board.push_uci(mv)
                    record_position_hash(board)

        elif command.startswith("go"):
            tokens = command.split()
            movetime = None
            max_depth = config.MAX_NEGAMAX_DEPTH
            go_ponder = "ponder" in tokens

            # If pondering is disabled or this is a ponder command but pondering not supported
            if go_ponder and (not config.PONDERING_ENABLED or not use_ponder):
                # Ignore ponder request, just wait
                continue

            if "depth" in tokens:
                max_depth = int(tokens[tokens.index("depth") + 1])

            if go_ponder:
                # Ponder search - set up TimeControl HERE (main thread) to avoid race with ponderhit.
                # The search thread will use use_existing_time_control=True.
                movetime = PONDER_TIME_LIMIT  # Safety cap (used as local fallback only)
                is_pondering = True
                ponder_fen = board.fen()
                ponder_start_time = time.time()
                ponder_best_move = None
                ponder_best_score = None

                # Set up TimeControl for ponder - long safety limit, ponderhit will update it
                TimeControl.time_limit = PONDER_TIME_LIMIT
                TimeControl.stop_search = False
                TimeControl.soft_stop = False
                TimeControl.start_time = time.perf_counter()
                TimeControl.hard_stop_time = TimeControl.start_time + PONDER_TIME_LIMIT + 60
                TimeControl.is_ponder_search = True

                # Store time info for use on ponderhit
                if "wtime" in tokens and "btime" in tokens:
                    ponder_time_info = {
                        'wtime': int(tokens[tokens.index("wtime") + 1]),
                        'btime': int(tokens[tokens.index("btime") + 1]),
                        'winc': int(tokens[tokens.index("winc") + 1]) if "winc" in tokens else 0,
                        'binc': int(tokens[tokens.index("binc") + 1]) if "binc" in tokens else 0,
                        'movestogo': int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else 30,
                        'turn': board.turn  # Store whose turn it is
                    }
                else:
                    ponder_time_info = None
            elif "infinite" in tokens:
                movetime = None
            elif "movetime" in tokens:
                movetime = int(tokens[tokens.index("movetime") + 1]) / 1000.0
            elif "wtime" in tokens and "btime" in tokens:
                wtime = int(tokens[tokens.index("wtime") + 1])
                btime = int(tokens[tokens.index("btime") + 1])
                winc = int(tokens[tokens.index("winc") + 1]) if "winc" in tokens else 0
                binc = int(tokens[tokens.index("binc") + 1]) if "binc" in tokens else 0
                movestogo = int(tokens[tokens.index("movestogo") + 1]) if "movestogo" in tokens else 30

                time_left = wtime if board.turn else btime
                increment = winc if board.turn else binc

                # FIXED: More conservative time management for Python + NN engines
                # Keep larger reserve for overhead (communication, NN startup, etc.)
                OVERHEAD_MS = 500  # 500ms for Python/NN overhead (was 100ms)
                MIN_RESERVE_MS = 1500  # Keep at least 1.5 seconds in reserve

                # Emergency mode: very little time left
                if time_left < 3000:  # Less than 3 seconds
                    movetime = max(0.05, (time_left - 500) / 1000.0 / 10)  # Use 10% of remaining minus 500ms reserve
                elif time_left < 10000:  # Less than 10 seconds
                    # Very conservative: use small fraction of time
                    base_time = (time_left - MIN_RESERVE_MS) / 20
                    with_increment = base_time + increment * 0.5
                    movetime = max(0.1, with_increment / 1000.0 - OVERHEAD_MS / 1000.0)
                elif time_left < 30000:  # Less than 30 seconds
                    effective_moves = max(movestogo, 25)
                    base_time = (time_left - MIN_RESERVE_MS) / effective_moves
                    with_increment = base_time + increment * 0.7
                    max_for_move = (time_left - MIN_RESERVE_MS) / 12
                    movetime = min(with_increment, max_for_move) / 1000.0
                    movetime = max(0.1, movetime - OVERHEAD_MS / 1000.0)
                else:
                    # Normal time management
                    effective_moves = max(movestogo, 25)
                    base_time = (time_left - MIN_RESERVE_MS) / effective_moves
                    with_increment = base_time + increment * 0.8
                    max_for_move = (time_left - MIN_RESERVE_MS) / 10
                    movetime = min(with_increment, max_for_move) / 1000.0
                    movetime = max(0.2, movetime - OVERHEAD_MS / 1000.0)

            TimeControl.stop_search = False

            # Try book move first (but not when pondering - GUI already set up the position)
            book_move = None
            if use_book and not go_ponder:
                book_move = get_book_move(board, min_weight=1, temperature=1.0)

            if book_move:
                print(f"info string Book move: {book_move.uci()}", flush=True)
                # Try to get a ponder move from the book
                board.push(book_move)
                ponder_book_move = get_book_move(board, min_weight=1, temperature=1.0) if use_ponder else None
                board.pop()

                if ponder_book_move:
                    print(f"bestmove {book_move.uci()} ponder {ponder_book_move.uci()}", flush=True)
                else:
                    print(f"bestmove {book_move.uci()}", flush=True)
            else:
                fen = board.fen()

                def search_and_report():
                    global resign_counter, is_pondering, ponder_best_move, ponder_best_score

                    # go_ponder is captured from the enclosing scope (immutable for this search).
                    # is_pondering is the live global that ponderhit can set to False.
                    diag_print(f"search_and_report starting, go_ponder={go_ponder}, is_pondering={is_pondering}, movetime={movetime}")

                    # Reset nodes counter before search
                    kpi['nodes'] = 0

                    # FIX: Initialize ponder move with first legal move as fallback
                    if go_ponder:
                        try:
                            temp_board = chess.Board(fen)
                            legal_moves = list(temp_board.legal_moves)
                            if legal_moves:
                                ponder_best_move = legal_moves[0]
                                ponder_best_score = 0
                                diag_print(f"Initialized ponder_best_move fallback: {ponder_best_move.uci()}")
                        except Exception as e:
                            diag_print(f"Failed to initialize ponder_best_move: {e}")

                    # Use parallel search if MP enabled, otherwise single-threaded
                    # For ponder searches: preserve TT (clear_tt=False) and use pre-configured
                    # TimeControl (use_existing_time_control=True) to avoid race with ponderhit.
                    try:
                        if lazy_smp.is_mp_enabled():
                            diag_print(f"Starting parallel search (ponder={go_ponder})")
                            best_move, score, pv, nodes, nps = lazy_smp.parallel_find_best_move(
                                fen, max_depth=max_depth, time_limit=movetime,
                                clear_tt=(not go_ponder),
                                use_existing_time_control=go_ponder)
                            diag_print(f"Parallel search returned: move={best_move}, score={score}")
                            # Print final info line for MP search
                            if pv:
                                print(
                                    f"info depth {len(pv)} score cp {score} nodes {nodes} nps {nps} pv {' '.join(m.uci() for m in pv)}",
                                    flush=True)
                        else:
                            diag_print(f"Starting single-threaded search (ponder={go_ponder})")
                            best_move, score, pv, _, _ = find_best_move(
                                fen, max_depth=max_depth, time_limit=movetime,
                                clear_tt=(not go_ponder),
                                use_existing_time_control=go_ponder)
                            diag_print(f"Single-threaded search returned: move={best_move}, score={score}")
                    except Exception as e:
                        diag_print(f"Search exception: {e}")
                        best_move = ponder_best_move
                        score = ponder_best_score if ponder_best_score is not None else 0
                        pv = [best_move] if best_move else []

                    # Store best move/score from pondering
                    if go_ponder:
                        if best_move is not None:
                            ponder_best_move = best_move
                            ponder_best_score = score
                            diag_print(f"Updated ponder_best_move: {ponder_best_move.uci()}")

                        # If search completed naturally (hit safety time limit or max depth),
                        # wait for stop (ponder miss) or ponderhit (which sets is_pondering=False).
                        # On ponderhit, TimeControl is updated with real time and the search
                        # has already completed — we just output the result.
                        # On stop (ponder miss), we output bestmove (GUI ignores it).
                        if not TimeControl.stop_search and is_pondering:
                            diag_print(f"Ponder search completed, waiting for stop/ponderhit")
                            while is_pondering and not TimeControl.stop_search:
                                time.sleep(0.01)
                            diag_print(f"Wait ended: stop={TimeControl.stop_search}, pondering={is_pondering}")

                    # Check for resign condition (not during active pondering)
                    should_resign = False
                    if not is_pondering:
                        if score <= RESIGN_THRESHOLD:
                            resign_counter += 1
                            if resign_counter >= RESIGN_CONSECUTIVE_MOVES:
                                should_resign = True
                                print(f"info string Resigning (score {score} cp for {resign_counter} moves)",
                                      flush=True)
                        else:
                            resign_counter = 0

                    # Extract ponder move from PV if available
                    ponder_move = None
                    if use_ponder and pv and len(pv) >= 2:
                        try:
                            pm = pv[1]
                            # Handle case where pv contains integers instead of chess.Move
                            if isinstance(pm, int):
                                from cached_board import int_to_move
                                ponder_move = int_to_move(pm)
                            else:
                                ponder_move = pm
                        except Exception as e:
                            diag_print(f"Error extracting ponder move: {e}")
                            ponder_move = None

                    diag_print(f"About to output bestmove: {best_move}, ponder_move: {ponder_move}")

                    try:
                        # Check for null/invalid move - avoid chess.Move.null() due to potential scoping issues
                        is_null_move = (best_move is None or
                                        (hasattr(best_move, 'uci') and best_move.uci() == '0000'))

                        if is_null_move:
                            print("bestmove 0000", flush=True)
                        elif should_resign:
                            # Output bestmove with resign indication
                            if ponder_move:
                                print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                            else:
                                print(f"bestmove {best_move.uci()}", flush=True)
                            print("info string resign", flush=True)
                        else:
                            if ponder_move:
                                print(f"bestmove {best_move.uci()} ponder {ponder_move.uci()}", flush=True)
                            else:
                                print(f"bestmove {best_move.uci()}", flush=True)
                    except Exception as e:
                        diag_print(f"Error outputting bestmove: {e}")
                        # Fallback: output without ponder move
                        try:
                            if best_move is not None and hasattr(best_move, 'uci'):
                                print(f"bestmove {best_move.uci()}", flush=True)
                            else:
                                print("bestmove 0000", flush=True)
                        except:
                            print("bestmove 0000", flush=True)

                    is_pondering = False
                    TimeControl.is_ponder_search = False
                    # Return NN evaluator to pool for reuse by future search threads
                    return_nn_evaluator_to_pool()
                    diag_print(f"search_and_report finished")

                search_thread = threading.Thread(target=search_and_report)
                search_thread.start()

        elif command == "ponderhit":
            # Ponder hit - the opponent played the move we were pondering on.
            # Instead of stopping and restarting, update TimeControl so the running
            # search continues with the real time budget. The search thread will
            # naturally stop when check_time() fires and output bestmove.
            if is_pondering and search_thread and search_thread.is_alive():
                diag_print(f"ponderhit received, continuing search with real time")

                # Calculate proper time allocation using stored time info
                ponderhit_time_limit = 5.0  # Default fallback

                if ponder_time_info and ponder_start_time:
                    # Get time info
                    time_left = ponder_time_info['wtime'] if ponder_time_info['turn'] else ponder_time_info['btime']
                    increment = ponder_time_info['winc'] if ponder_time_info['turn'] else ponder_time_info['binc']
                    movestogo = ponder_time_info['movestogo']

                    OVERHEAD_MS = 500
                    MIN_RESERVE_MS = 1500

                    # Use similar time management as regular search
                    if time_left < 3000:
                        ponderhit_time_limit = max(0.05, (time_left - 500) / 1000.0 / 10)
                    elif time_left < 10000:
                        base_time = (time_left - MIN_RESERVE_MS) / 20
                        with_increment = base_time + increment * 0.5
                        ponderhit_time_limit = max(0.1, with_increment / 1000.0 - OVERHEAD_MS / 1000.0)
                    elif time_left < 30000:
                        effective_moves = max(movestogo, 25)
                        base_time = (time_left - MIN_RESERVE_MS) / effective_moves
                        with_increment = base_time + increment * 0.7
                        max_for_move = (time_left - MIN_RESERVE_MS) / 12
                        ponderhit_time_limit = min(with_increment, max_for_move) / 1000.0
                        ponderhit_time_limit = max(0.1, ponderhit_time_limit - OVERHEAD_MS / 1000.0)
                    else:
                        # Normal time - be slightly more aggressive since TT is warm
                        effective_moves = max(movestogo, 30)
                        base_time = (time_left - MIN_RESERVE_MS) / effective_moves
                        with_increment = base_time + increment * 0.7
                        max_for_move = (time_left - MIN_RESERVE_MS) / 10
                        ponderhit_time_limit = min(with_increment, max_for_move) / 1000.0
                        ponderhit_time_limit = max(0.2, ponderhit_time_limit - OVERHEAD_MS / 1000.0)

                    # Hard cap at 10% of remaining time
                    hard_cap = (time_left / 1000.0) * 0.10
                    ponderhit_time_limit = min(ponderhit_time_limit, hard_cap)

                    diag_print(f"ponderhit time_limit={ponderhit_time_limit:.2f}s (clock={time_left}ms)")

                # If we have very little time, force immediate stop so search outputs now
                if ponderhit_time_limit < 0.3 and ponder_best_move:
                    diag_print(f"ponderhit: time critical, forcing immediate stop")
                    # Set is_pondering=False so search thread outputs bestmove
                    is_pondering = False
                    TimeControl.is_ponder_search = False
                    # Force the search to stop immediately
                    TimeControl.stop_search = True
                    lazy_smp.stop_parallel_search()
                else:
                    # Discount ponderhit time to absorb MIN_PREFERRED_DEPTH overshoot.
                    # The search clears soft_stop to force depth 5, overshooting by 30-50%.
                    # Reducing the budget ensures hard_stop lands near the intended allocation.
                    ponderhit_raw = ponderhit_time_limit
                    ponderhit_time_limit = max(0.1, ponderhit_time_limit * 0.70)
                    diag_print(f"ponderhit adjusted: {ponderhit_raw:.2f}s -> {ponderhit_time_limit:.2f}s")

                    # Update TimeControl with real time — the running search picks this up.
                    # Order matters to avoid race with check_time():
                    TimeControl.soft_stop = False
                    TimeControl.start_time = time.perf_counter()
                    TimeControl.time_limit = ponderhit_time_limit
                    # Tighter grace for ponderhit (20% vs 50% for normal search)
                    # because TT is warm and we already have a deep result from ponder
                    grace_period = max(ponderhit_time_limit * 0.20, 0.15)
                    TimeControl.hard_stop_time = TimeControl.start_time + ponderhit_time_limit + grace_period
                    TimeControl.is_ponder_search = False

                    is_pondering = False
                    diag_print(f"ponderhit: TimeControl updated, search continues")

            elif is_pondering:
                # Search thread already finished (or wasn't started) — just clear state
                diag_print(f"ponderhit: search thread not alive, clearing ponder state")
                is_pondering = False
                TimeControl.is_ponder_search = False

        elif command == "stop":
            diag_print(f"stop received, is_pondering={is_pondering}")
            TimeControl.stop_search = True
            lazy_smp.stop_parallel_search()  # Signal MP workers to stop

            thread_finished = False
            if search_thread and search_thread.is_alive():
                diag_print(f"Waiting for search thread to finish (timeout=2.0)")
                search_thread.join(timeout=2.0)  # FIX: Add timeout to prevent infinite wait
                thread_finished = not search_thread.is_alive()
                diag_print(f"Search thread finished: {thread_finished}")
                if not thread_finished:
                    diag_print("Warning: search thread did not terminate in time")
            else:
                thread_finished = True
                diag_print(f"Search thread was not alive")

            # FIX: If search thread didn't finish in time (and thus didn't output bestmove),
            # we must output bestmove ourselves to satisfy UCI protocol
            diag_print(
                f"Checking if we need fallback bestmove: thread_finished={thread_finished}")
            if not thread_finished:
                # Output fallback bestmove if thread is stuck
                diag_print(f"Outputting fallback bestmove: ponder_best_move={ponder_best_move}")
                if ponder_best_move:
                    print(f"bestmove {ponder_best_move.uci()}", flush=True)
                else:
                    # Last resort: output null move
                    print("bestmove 0000", flush=True)

            # Reset all ponder state
            is_pondering = False
            TimeControl.is_ponder_search = False
            diag_print(f"stop handling complete")

        elif command == "quit":
            TimeControl.stop_search = True
            lazy_smp.stop_parallel_search()
            if search_thread and search_thread.is_alive():
                search_thread.join(timeout=2.0)  # FIX: Add timeout
            lazy_smp.shutdown_worker_pool()  # Clean shutdown of workers
            break


if __name__ == "__main__":
    if os.path.exists(DEFAULT_BOOK_PATH):
        init_opening_book(str(DEFAULT_BOOK_PATH))
    else:
        print(f"info string Book not found: {DEFAULT_BOOK_PATH}", flush=True)

    register_config_tunables()
    uci_loop()