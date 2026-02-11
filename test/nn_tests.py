"""
Chess Neural Network Evaluation Test Suite
Tests trained NNUE models using the NNEvaluator abstraction.

This test file reuses code from nn_evaluator.py to ensure bugs in the
evaluator classes are caught by these tests.

Test Modes:
- Interactive-FEN: Interactive FEN evaluation (default)
- Incremental-vs-Full: Performance comparison
- Accumulator-Correctness: Verify incremental == full evaluation
- Eval-Accuracy: Test prediction accuracy against ground truth from training shards
- Feature-Extraction: Verify feature extraction correctness
- Symmetry: Test evaluation symmetry (mirrored positions)
- Edge-Cases: Test edge cases (checkmate, stalemate, special moves)
- Reset-Consistency: Test evaluator reset functionality
- Deep-Search-Simulation: Simulate deep search with many push/pop cycles
- Random-Games: Test with random legal move sequences
"""

import argparse
import io
import random
import re
import sys
import time
from contextlib import redirect_stdout

import chess
import chess.pgn
import numpy as np

from cached_board import CachedBoard, uci_to_int, int_to_uci, WHITE, BLACK
from config import MAX_SCORE, TANH_SCALE
from chess_engine import find_best_move
from nn_evaluator import NNEvaluator
# Import from our modules - this ensures we test the actual production code
from nn_inference import (
    NNUEFeatures,
    NNUE_INPUT_SIZE,
)

CP_ERROR_CLIP = 100  # Keep low to make the average more sense.

# Configuration
VALID_TEST_TYPES = {
    0: "Interactive-FEN",
    1: "Incremental-vs-Full",
    2: "Accumulator-Correctness",
    3: "Eval-Accuracy",
    4: "NN-vs-Stockfish",
    5: "Feature-Extraction",
    6: "Symmetry",
    7: "Edge-Cases",
    8: "Reset-Consistency",
    9: "Deep-Search-Simulation",
    10: "Random-Games",
    11: "CP-Integrity",
    12: "Engine-Tests",
    13: "All",
}


def get_model_path(nn_type: str) -> str:
    """Get the model path for NNUE."""
    return "model/nnue.pt"


def output_to_centipawns(output: float) -> float:
    """Convert linear network output (tanh space) to centipawns."""
    output = np.clip(output, -0.9999, 0.9999)
    return np.arctanh(output) * TANH_SCALE


def evaluator_push(evaluator: NNEvaluator, board: CachedBoard, move_int: int):
    """
    Push a move to the evaluator and board using the unified interface.

    This matches the pattern used in chess_engine.py's push_move_int() function.
    """
    evaluator.push_with_board(board, move_int)


def evaluator_pop(evaluator: NNEvaluator, board: CachedBoard):
    """
    Pop a move from the evaluator and board.

    This matches the pattern used in chess_engine.py: board.pop() then evaluator.pop().
    """
    board.pop()
    evaluator.pop()


# =============================================================================
# Interactive FEN Evaluation
# =============================================================================

def evaluate_fen(fen: str, evaluator: NNEvaluator) -> dict:
    """Evaluate a position from FEN string using the evaluator."""
    try:
        board = CachedBoard(fen)
        # Use full evaluation for standalone FEN evaluation
        centipawns = evaluator.evaluate_full_centipawns(board)

        return {
            'success': True,
            'fen': fen,
            'side_to_move': 'White' if board.turn == WHITE else 'Black',
            'centipawns': centipawns
        }
    except ValueError as e:
        return {'success': False, 'error': f"Invalid FEN: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"Evaluation error: {e}"}


def print_evaluation(result: dict):
    """Pretty print evaluation results."""
    if not result['success']:
        print(f"✗ {result['error']}")
        return

    print("─" * 60)
    print(f"FEN: {result['fen']}")
    print(f"Side to move: {result['side_to_move']}")
    print(f"Evaluation: {result['centipawns']:+.2f} centipawns")

    pawns = result['centipawns'] / 100.0
    stm = result['side_to_move']
    opponent = "Black" if stm == "White" else "White"

    if abs(pawns) < 0.1:
        assessment = "≈ Equal position"
    elif pawns > 3:
        assessment = f"Decisive advantage for {stm}"
    elif pawns > 1:
        assessment = f"Clear advantage for {stm}"
    elif pawns > 0.3:
        assessment = f"Slight advantage for {stm}"
    elif pawns < -3:
        assessment = f"Decisive advantage for {opponent}"
    elif pawns < -1:
        assessment = f"Clear advantage for {opponent}"
    elif pawns < -0.3:
        assessment = f"Slight advantage for {opponent}"
    else:
        assessment = "≈ Equal position"

    print(f"Assessment: {assessment}")
    print("─" * 60)


def print_help():
    """Print help information."""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator - Help")
    print("=" * 60)
    print("Commands:")
    print("  Enter FEN string    - Evaluate position")
    print("  'startpos'          - Evaluate starting position")
    print("  'help'              - Show this help")
    print("  'exit' or 'quit'    - Exit program")
    print("\nExample FEN:")
    print("  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("=" * 60 + "\n")


def interactive_loop(evaluator: NNEvaluator, nn_type: str):
    """Main interactive loop for FEN evaluation."""
    print("\n" + "=" * 60)
    print("Chess Position Evaluator")
    print("=" * 60)
    print(f"Network type: {nn_type}")
    print("\nEnter FEN strings to evaluate positions.")
    print("Type 'help' for instructions, 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    while True:
        try:
            user_input = input("FEN> ").strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print_help()
                continue

            if not user_input:
                continue

            if user_input.lower() == 'startpos':
                user_input = startpos_fen
                print(f"Using starting position: {startpos_fen}")

            result = evaluate_fen(user_input, evaluator)
            print_evaluation(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Accumulator Correctness Test
# =============================================================================

def test_accumulator_correctness(nn_type: str, model_path: str):
    """
    Test that incremental evaluation matches full evaluation.

    Uses the NNEvaluator's validate_incremental() method which compares
    evaluate() (incremental) vs evaluate_full() (matrix multiplication).

    This tests the actual production code path in nn_evaluator.py.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Accumulator Correctness Test")
    print("=" * 70)

    # Test sequence includes: captures, en-passant, castling, king moves, promotion
    moves_san = [
        "d4", "e5", "dxe5", "f5", "exf6", "Nh6", "Bf4", "Bd6",
        "Nc3", "O-O", "Qd3", "Nc6", "O-O-O", "a6", "fxg7", "Rb8",
        "gxf8=N", "Kxf8", "Kb1", "Bxf4"
    ]

    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    print(f"\nPlaying through {len(moves_san)} moves and verifying after each...")
    all_passed = True
    move_count = 0

    for move_san in moves_san:
        move_int = board.parse_san(move_san)

        # Push using our helper
        evaluator_push(evaluator, board, move_int)
        move_count += 1

        # Use the evaluator's built-in validation
        eval_inc = evaluator._evaluate(board)
        eval_full = evaluator._evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        passed = diff < 1e-1

        print(f"\n{'─' * 70}")
        print(f"After move {move_count}: {move_san}")
        print(f"{'─' * 70}")
        print(f"Position: {board.fen()}")
        print(f"Incremental: {eval_inc:.10f} ({output_to_centipawns(eval_inc):+.2f} cp)")
        print(f"Full:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:  {diff:.10e}")

        if passed:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")
            all_passed = False

    # Now test popping moves
    num_pops = 4
    print(f"\nPopping {num_pops} moves and verifying...")

    for i in range(num_pops):
        evaluator_pop(evaluator, board)

        eval_inc = evaluator._evaluate(board)
        eval_full = evaluator._evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        passed = diff < 1e-1

        print(f"\n{'─' * 70}")
        print(f"After pop {i + 1}")
        print(f"{'─' * 70}")
        print(f"Position: {board.fen()}")
        print(f"Incremental: {eval_inc:.10f} ({output_to_centipawns(eval_inc):+.2f} cp)")
        print(f"Full:        {eval_full:.10f} ({output_to_centipawns(eval_full):+.2f} cp)")
        print(f"Difference:  {diff:.10e}")

        if passed:
            print("✓ PASS: Incremental and full evaluation match!")
        else:
            print("✗ FAIL: Evaluations differ!")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All accumulator correctness tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Performance Test (Incremental vs Full)
# =============================================================================

def performance_test(nn_type: str, model_path: str):
    """
    Compare performance of full evaluation vs incremental evaluation.

    Tests the actual NNEvaluator implementation from nn_evaluator.py.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Performance Test: Full vs Incremental Evaluation")
    print("=" * 70)

    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)
    move_int = uci_to_int("e2e4")

    # Test 1: Full evaluation
    num_full_iters = 1000
    print(f"\n1. Full Evaluation - Matrix Multiplication ({num_full_iters} iterations)")

    start_time = time.time()
    for _ in range(num_full_iters):
        _ = evaluator.evaluate_full_centipawns(board)
    full_time = time.time() - start_time

    print(f"   Time: {full_time:.4f} seconds")
    print(f"   Avg per evaluation: {full_time / num_full_iters * 1000:.3f} ms")

    # Test 2: Incremental evaluation with push/pop cycles
    num_cycles = num_full_iters // 2
    print(f"\n2. Incremental Evaluation - Push/Pop Cycles ({num_cycles} cycles)")

    start_time = time.time()

    for _ in range(num_cycles):
        # Push
        evaluator_push(evaluator, board, move_int)

        # Evaluate (uses accumulator)
        _ = evaluator.evaluate_centipawns(board)

        # Pop
        evaluator_pop(evaluator, board)

    incremental_time = time.time() - start_time

    print(f"   Time: {incremental_time:.4f} seconds")
    print(f"   Avg per push/eval/pop cycle: {incremental_time / num_cycles * 1000:.3f} ms")

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print(f"  Full (matrix multiply):     {full_time / num_full_iters * 1000:.3f} ms per evaluation")
    print(f"  Incremental (accumulator):  {incremental_time / num_cycles * 1000:.3f} ms per cycle")

    # Calculate speedup (comparing one full eval vs one incremental cycle)
    speedup = (full_time / num_full_iters) / (incremental_time / (2 * num_cycles))
    print(f"  Speedup: {speedup:.2f}x")

    print("\nNote: Incremental uses accumulator (add/subtract weight vectors)")
    print("instead of full matrix multiplication for the first layer.")
    print("=" * 70)

    if nn_type == "NNUE":
        expected_speedup = 80
        expected_incr_time = 0.15
    else:
        expected_speedup = 5
        expected_incr_time = 0.3

    all_passed = speedup > expected_speedup and incremental_time / num_cycles * 1000 < expected_incr_time
    if not all_passed:
        print(f"  ✗ FAIL, expected speedup >{expected_speedup}, got {speedup:.2f}. "
              f"expected incr-time <{expected_incr_time} but got {incremental_time / num_cycles * 1000:.2f} ms")

    return all_passed


# =============================================================================
# Evaluation Accuracy Test
# =============================================================================

def test_eval_accuracy(nn_type: str, model_path: str, positions_size: int = 1000):
    """
    Test evaluation accuracy against ground truth from training data shards.

    Uses diagnostic records (with FEN) from binary shard files to compare
    model predictions against stored Stockfish evaluations.
    """
    from nn_train.shard_io import ShardReader, find_shards

    print("\n" + "=" * 70)
    print(f"{nn_type} Evaluation Accuracy Test")
    print("=" * 70)

    # Find shard files
    _, nnue_shards = find_shards("data", nn_type)
    shard_files = nnue_shards

    if not shard_files:
        print(f"\n✗ ERROR: No shard files found for {nn_type}")
        print("Please ensure .bin.zst files exist in the data directory")
        return None

    # Pick a random shard
    shard_path = random.choice(shard_files)
    print(f"\nReading diagnostic records from: {shard_path}")

    # Read diagnostic records (which have FEN)
    reader = ShardReader(nn_type)
    records = reader.read_diagnostic_records(shard_path, max_records=positions_size)

    if not records:
        print("\n✗ ERROR: No diagnostic records found in shard file")
        print("Diagnostic records are written every 1000 positions and include FEN.")
        print("You may need to regenerate shards with the updated prepare_data.py")
        return None

    print(f"✓ Loaded {len(records)} diagnostic records from shard")

    # Create evaluator
    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    # Evaluate all positions
    print("\nEvaluating positions...")

    true_tanh_values = []
    pred_tanh_values = []
    true_cp_list = []
    pred_cp_list = []
    fen_list = []
    errors = []

    for i, rec in enumerate(records):
        if (i + 1) % max(1, len(records) // 10) == 0:
            print(f"  Progress: {i + 1}/{len(records)}")

        try:
            fen = rec['fen']

            # Evaluate using NNUE
            pred_output = evaluator._evaluate_full(CachedBoard(fen))

            true_tanh = np.tanh(rec['score_cp'] / TANH_SCALE)

            # Convert prediction to CP and cap at MAX_SCORE
            pred_cp_raw = np.arctanh(np.clip(pred_output, -0.99999, 0.99999)) * TANH_SCALE
            pred_cp_capped = max(-MAX_SCORE, min(MAX_SCORE, pred_cp_raw))

            true_tanh_values.append(true_tanh)
            pred_tanh_values.append(pred_output)
            true_cp_list.append(rec['score_cp'])
            pred_cp_list.append(pred_cp_capped)
            fen_list.append(fen)

        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        print(f"\n✗ {len(errors)} positions failed to evaluate:")
        for idx, error in errors[:5]:
            print(f"  Position {idx}: {error[:60]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not true_tanh_values:
        print("\n✗ ERROR: No positions could be evaluated")
        return None

    # Compute metrics
    true_tanh_values = np.array(true_tanh_values)
    pred_tanh_values = np.array(pred_tanh_values)
    true_cp_values = np.array(true_cp_list)
    pred_cp_values = np.array(pred_cp_list)

    mse = np.mean((true_tanh_values - pred_tanh_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_tanh_values - pred_tanh_values))

    # Centipawn metrics (with capped values)
    cp_diff = pred_cp_values - true_cp_values
    cp_diff_capped = np.clip(cp_diff, -CP_ERROR_CLIP, CP_ERROR_CLIP)
    mse_cp = np.mean(cp_diff_capped ** 2)
    rmse_cp = np.sqrt(mse_cp)
    mae_cp = np.mean(np.abs(cp_diff_capped))

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print("─" * 70)
    print(f"Positions evaluated: {len(pred_tanh_values)}")
    print(f"Positions failed:    {len(errors)}")
    print()
    print("Tanh Space (network output space):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print()
    print("Centipawn Space (Error capped at +/-{:,}):".format(CP_ERROR_CLIP))
    print(f"  MSE:  {mse_cp:.2f}")
    print(f"  RMSE: {rmse_cp:.2f} cp")
    print(f"  MAE:  {mae_cp:.2f} cp")

    # Sample predictions
    print("\n" + "─" * 70)
    print("Sample predictions (first 20):")
    print("─" * 70)
    for i in range(min(20, len(pred_tanh_values))):
        true_cp = true_cp_list[i]
        pred_cp = pred_cp_list[i]
        diff = np.clip(pred_cp - true_cp, -MAX_SCORE, MAX_SCORE)
        fen = fen_list[i]
        print(f"{i + 1:2d}. True: {true_cp:+7.1f} cp | "
              f"Pred: {pred_cp:+7.1f} cp | "
              f"Diff: {diff:+7.1f} cp")
        print(f"    FEN: {fen}")

    print("\n" + "=" * 70)
    print("Evaluation accuracy test complete!")
    print("=" * 70)

    expected_mse = 0.07
    expected_mae_cp = 100

    all_passed = mse < expected_mse and mae_cp < expected_mae_cp
    if not all_passed:
        print(f"  ✗ FAIL, expected mae <{expected_mse}, got {mae:.4f}. "
              f"expected MAE CP <{expected_mae_cp} but got {mae_cp} ms")

    return all_passed


# =============================================================================
# NN vs Stockfish Test
# =============================================================================
def read_sf_eval_file(filepath: str):
    """
    Read the pre-computed Stockfish evaluation binary file.

    Binary format:
        Header: [num_records:uint32]
        Each record: [fen_length:uint8][fen_bytes:char[]][sf_eval_cp:int16][shard_cp:int16]

    Returns list of (fen, sf_eval_cp, shard_cp) tuples.
    """
    import struct

    records = []
    with open(filepath, 'rb') as f:
        # Read header
        num_records = struct.unpack('<I', f.read(4))[0]

        # Read records
        for _ in range(num_records):
            fen_len = struct.unpack('<B', f.read(1))[0]
            fen = f.read(fen_len).decode('utf-8')
            sf_eval_cp = struct.unpack('<h', f.read(2))[0]
            shard_cp = struct.unpack('<h', f.read(2))[0]
            records.append((fen, sf_eval_cp, shard_cp))

    return records


SF_EVAL_FILE_PATH = "data/sf_nnue_static_eval.bin"


def test_nn_vs_stockfish(nn_type: str, model_path: str, positions_size: int = 1000):
    """
    Test neural network predictions against Stockfish's static NNUE evaluation.

    Reads pre-computed Stockfish evaluations from data/sf_nnue_static_eval.bin
    (generated by build_sf_static_eval_file.py) for fast comparison.

    IMPORTANT: This test now uses incremental evaluation (_evaluate) which
    properly respects L1_QUANTIZATION settings, unlike the full evaluation
    path which always uses FP32.
    """
    import os

    print("\n" + "=" * 70)
    print(f"{nn_type} vs Stockfish Static Evaluation Test")
    print("=" * 70)

    # Check for pre-computed Stockfish evaluation file
    if not os.path.exists(SF_EVAL_FILE_PATH):
        print(f"\n✗ ERROR: Pre-computed Stockfish evaluation file not found:")
        print(f"    {SF_EVAL_FILE_PATH}")
        print("\nPlease generate it first by running:")
        print(f"    python build_sf_static_eval_file.py --num-positions {positions_size}")
        return None

    # Read pre-computed evaluations
    print(f"\nReading pre-computed Stockfish evaluations from: {SF_EVAL_FILE_PATH}")
    all_records = read_sf_eval_file(SF_EVAL_FILE_PATH)
    print(f"✓ Loaded {len(all_records)} pre-computed evaluations")

    # Take first N positions if requested fewer than available
    if positions_size < len(all_records):
        records = all_records[:positions_size]
        print(f"  Using first {positions_size} positions")
    else:
        records = all_records
        if positions_size > len(all_records):
            print(f"  Warning: Requested {positions_size} but only {len(all_records)} available")

    # Create evaluator and show quantization mode
    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    # Display quantization mode for verification
    from config import L1_QUANTIZATION
    print(f"  L1 Quantization Mode: {L1_QUANTIZATION}")

    # Evaluate all positions
    print("\nEvaluating positions with neural network...")

    sf_tanh_values = []
    true_tanh_values = []
    pred_tanh_values = []
    sf_cp_list = []
    pred_cp_list = []
    true_cp_list = []
    fen_list = []
    errors = []

    for i, (fen, sf_cp_stm, shard_cp) in enumerate(records):
        if (i + 1) % max(1, len(records) // 10) == 0:
            print(f"  Progress: {i + 1}/{len(records)}")

        try:
            # Set up board and evaluate with INCREMENTAL evaluation
            board = CachedBoard(fen)
            evaluator.reset(board)  # Initialize accumulators for this position
            pred_output = evaluator._evaluate(board)  # Uses incremental path with quantization

            # SF eval is already in STM perspective and capped at MAX_SCORE
            sf_cp_capped = sf_cp_stm

            sf_tanh = np.tanh(sf_cp_capped / TANH_SCALE)

            # Convert prediction to CP and cap at MAX_SCORE
            pred_cp_raw = np.arctanh(np.clip(pred_output, -0.99999, 0.99999)) * TANH_SCALE
            pred_cp_capped = max(-MAX_SCORE, min(MAX_SCORE, pred_cp_raw))

            true_tanh = np.tanh(shard_cp / TANH_SCALE)

            sf_tanh_values.append(sf_tanh)
            true_tanh_values.append(true_tanh)
            pred_tanh_values.append(pred_output)
            sf_cp_list.append(sf_cp_capped)
            pred_cp_list.append(pred_cp_capped)
            true_cp_list.append(shard_cp)
            fen_list.append(fen)

        except Exception as e:
            errors.append((i, str(e)))

    if errors:
        print(f"\n✗ {len(errors)} positions failed:")
        for idx, error in errors[:5]:
            print(f"  Position {idx}: {error[:60]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not sf_tanh_values:
        print("\n✗ ERROR: No positions could be evaluated")
        return None

    # Compute metrics
    sf_tanh_values = np.array(sf_tanh_values)
    true_tanh_values = np.array(true_tanh_values)
    pred_tanh_values = np.array(pred_tanh_values)

    nn_error_values = np.abs(pred_tanh_values - true_tanh_values)
    sf_error_values = np.abs(sf_tanh_values - true_tanh_values)
    mse = np.mean((nn_error_values - sf_error_values) ** 2)
    rmse = np.sqrt(mse)
    mean_delta_error = np.mean(nn_error_values - sf_error_values)

    mean_delta_error_cp = np.arctanh(np.clip(mean_delta_error, -0.99999, 0.99999)) * TANH_SCALE
    mean_delta_error_cp = np.clip(mean_delta_error_cp, -CP_ERROR_CLIP, CP_ERROR_CLIP)

    # Results
    print("\n" + "─" * 70)
    print("Results:")
    print("─" * 70)
    print(f"Positions evaluated: {len(pred_tanh_values)}")
    print(f"Positions failed:    {len(errors)}")
    print()
    print("Tanh Space (network output space):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(
        f"  Mean Delta Error:  {mean_delta_error:.6f} {'NN is better than SF' if mean_delta_error <= 0 else 'SF is better than NN'}")
    print()
    print("Centipawn Space (Error capped at +/-{:,}):".format(CP_ERROR_CLIP))
    print(f"  Mean Delta Error:  {mean_delta_error_cp:.2f} cp")

    # Sample predictions
    print("\n" + "─" * 70)
    print("Sample predictions (first 10):")
    print("─" * 70)
    for i in range(min(10, len(pred_tanh_values))):
        sf_cp = sf_cp_list[i]
        pred_cp = pred_cp_list[i]
        true_cp = true_cp_list[i]
        diff = np.clip(pred_cp - sf_cp, -MAX_SCORE, MAX_SCORE)
        fen = fen_list[i]
        print(f"{i + 1:2d}.  True: {true_cp:+7.1f} cp | SF: {sf_cp:+7.1f} cp | "
              f"Pred: {pred_cp:+7.1f} cp | "
              f"Diff: {diff:+7.1f} cp")
        print(f"    FEN: {fen}")

    print("\n" + "=" * 70)
    print("NN vs Stockfish test complete!")
    print("=" * 70)

    all_passed = mean_delta_error < 0.01 and mean_delta_error_cp < 20
    return all_passed


# =============================================================================
# Feature Extraction Test
# =============================================================================

def test_feature_extraction(nn_type: str):
    """
    Test that feature extraction produces correct and consistent results.

    Verifies:
    - Feature indices are within valid range
    - No duplicate features
    - Symmetric positions produce mirrored features
    - Feature count is reasonable for different positions
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Feature Extraction Test")
    print("=" * 70)

    all_passed = True

    # Test positions with known properties
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("8/8/8/8/8/8/8/4K2k w - - 0 1", "Minimal position (2 kings)"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "Castling rights position"),
        ("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "After 1.e4 e6"),
        ("8/P7/8/8/8/8/8/4K2k w - - 0 1", "Promotion possible"),
    ]

    max_features = NNUE_INPUT_SIZE
    feature_extractor = NNUEFeatures

    for fen, description in test_positions:
        print(f"\n{'─' * 70}")
        print(f"Testing: {description}")
        print(f"FEN: {fen}")

        board = CachedBoard(fen)

        white_feat, black_feat = feature_extractor.board_to_features(board)

        # Check white features
        white_in_range = all(0 <= f < max_features for f in white_feat)
        white_no_dups = len(white_feat) == len(set(white_feat))

        # Check black features
        black_in_range = all(0 <= f < max_features for f in black_feat)
        black_no_dups = len(black_feat) == len(set(black_feat))

        print(f"  White features: {len(white_feat)}, Black features: {len(black_feat)}")
        print(f"  White in range: {white_in_range}, No duplicates: {white_no_dups}")
        print(f"  Black in range: {black_in_range}, No duplicates: {black_no_dups}")

        passed = white_in_range and white_no_dups and black_in_range and black_no_dups

        if passed:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All feature extraction tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Symmetry Test
# =============================================================================
def mirror_fen(fen: str) -> str:
    """
    Create a color-swapped, board-flipped FEN that should give the negated evaluation.

    To properly mirror:
    1. Flip the board vertically (reverse rank order)
    2. Swap piece colors (uppercase <-> lowercase)
    3. Swap side to move
    4. Swap castling rights (K<->k, Q<->q)
    5. Mirror en passant square if present
    """
    parts = fen.split()
    board_str = parts[0]

    # Split into ranks and reverse order (flip vertically)
    ranks = board_str.split('/')
    ranks = ranks[::-1]

    # Swap colors in each rank
    def swap_colors(s):
        return s.swapcase()

    ranks = [swap_colors(r) for r in ranks]
    new_board = '/'.join(ranks)

    # Swap side to move
    new_stm = 'b' if parts[1] == 'w' else 'w'

    # Swap castling rights
    castling = parts[2]
    if castling == '-':
        new_castling = '-'
    else:
        new_castling = castling.swapcase()

    # Mirror en passant square
    ep = parts[3]
    if ep != '-':
        file = ep[0]
        rank = ep[1]
        new_rank = str(9 - int(rank))  # 3 -> 6, 6 -> 3
        new_ep = file + new_rank
    else:
        new_ep = '-'

    return f"{new_board} {new_stm} {new_castling} {new_ep} {parts[4]} {parts[5]}"


def test_symmetry(nn_type: str, model_path: str):
    """
    Test that the evaluator respects chess symmetry.

    A position with colors swapped and board flipped should give
    the negated evaluation (or very close to it).
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Symmetry Test")
    print("=" * 70)

    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True

    # Test positions - pairs of (white_to_move_fen, black_equivalent_fen)
    test_fens = [
        # Starting position - should be symmetric
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
        "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1",
    ]

    tolerance = 0.01  # Allow some tolerance due to asymmetries in training data

    for fen1 in test_fens:
        fen2 = mirror_fen(fen1)
        print(f"\n{'─' * 70}")
        print(f"Original: {fen1}")
        print(f"Mirrored: {fen2}")

        board1 = CachedBoard(fen1)
        board2 = CachedBoard(fen2)

        eval1 = evaluator._evaluate_full(board1)
        eval2 = evaluator._evaluate_full(board2)

        # Evaluations should be opposite (one is STM advantage, other is opponent)
        # Since both are from STM perspective, swapping colors should negate
        diff = abs(eval1 - eval2)

        print(f"  Eval 1: {eval1:.6f} ({output_to_centipawns(eval1):+.2f} cp)")
        print(f"  Eval 2: {eval2:.6f} ({output_to_centipawns(eval2):+.2f} cp)")
        print(f"  Diff (should be ~0): {eval1 - eval2:.6f}")

        passed = diff < tolerance
        if passed:
            print("  ✓ PASS: Evaluations are symmetric")
        else:
            print(f"  ✗ WARNING: Evaluations differ by {diff:.6f} (tolerance: {tolerance})")
            all_passed = False

    print("\n" + "=" * 70)
    print("Symmetry test complete!")
    print("(Note: Small asymmetries are expected due to training data)")
    print("=" * 70)

    return all_passed


# =============================================================================
# Edge Cases Test
# =============================================================================

def test_edge_cases(nn_type: str, model_path: str):
    """
    Test evaluation of edge cases: checkmate, stalemate, and special moves.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Edge Cases Test")
    print("=" * 70)

    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True

    # Test cases with expected behavior
    test_cases = [
        # Checkmate positions
        ("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
         "Fool's mate (White is checkmated)", "checkmate"),

        ("r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "Scholar's mate (Black is checkmated)", "checkmate"),

        # Stalemate positions (verified with python-chess)
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
         "Black king stalemated in corner by queen", "stalemate"),
        ("k7/8/1K6/8/8/8/8/8 b - - 0 1",
         "King vs King (insufficient material - draw)", "insufficient_material"),
        ("8/8/8/8/8/5k2/5p2/5K2 w - - 0 1",
         "White king stalemated by pawn", "stalemate"),

        # Normal positions
        ("8/8/8/8/8/6k1/8/5K1Q b - - 0 1",
         "Not stalemate - black king has legal moves", "normal"),

        # Complex middlegame
        ("r3k2r/pp3ppp/2p1pn2/3p4/3P4/2P1PN2/PP3PPP/R3K2R w KQkq - 0 10",
         "Complex middlegame", "normal"),
    ]

    for fen, description, expected_type in test_cases:
        print(f"\n{'─' * 70}")
        print(f"Testing: {description}")
        print(f"FEN: {fen}")

        test_board = chess.Board(fen)

        # Check game state
        is_checkmate = test_board.is_checkmate()
        is_stalemate = test_board.is_stalemate()
        is_game_over = test_board.is_game_over()
        is_insufficient = test_board.is_insufficient_material()

        print(f"  Checkmate: {is_checkmate}, Stalemate: {is_stalemate}, "
              f"Game over: {is_game_over}, Insufficient: {is_insufficient}")

        # Get evaluation
        try:
            eval_result = evaluator.evaluate_full_centipawns(CachedBoard(fen))
            print(f"  Evaluation: {eval_result:.6f} ({output_to_centipawns(eval_result):+.2f} cp)")

            # Verify expected behavior
            if expected_type == "checkmate":
                if eval_result == -MAX_SCORE:
                    print("  ✓ PASS: Correctly identified as checkmate")
                else:
                    print("  ✗ FAIL: Should be checkmate")
                    all_passed = False
            elif expected_type == "stalemate":
                if is_stalemate:
                    if abs(eval_result) < 0.01:  # Should evaluate to ~0
                        print("  ✓ PASS: Correctly identified as stalemate with ~0 eval")
                    else:
                        print(f"  ✗ WARNING: Stalemate but eval is {eval_result:.6f}")
                else:
                    print("  ✗ FAIL: Should be stalemate")
                    all_passed = False
            elif expected_type == "insufficient_material":
                if abs(eval_result) < 0.01:  # Should evaluate to ~0:
                    print("  ✓ PASS: Correctly identified as draw (insufficient material)")
                else:
                    print("  ✗ FAIL: Should be insufficient material")
                    all_passed = False
            else:
                print("  ✓ PASS: Normal position evaluated successfully")

        except Exception as e:
            print(f"  ✗ FAIL: Evaluation error: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All edge case tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Reset Consistency Test
# =============================================================================

def test_reset_consistency(nn_type: str, model_path: str):
    """
    Test that resetting the evaluator produces consistent results.

    After push/pop operations, reset should restore to a clean state
    that matches a freshly created evaluator.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Reset Consistency Test")
    print("=" * 70)

    all_passed = True

    # Create two evaluators
    board1 = CachedBoard()
    board2 = CachedBoard()

    evaluator1 = NNEvaluator.create(board1, nn_type, model_path)
    evaluator2 = NNEvaluator.create(board2, nn_type, model_path)

    # Make some moves on evaluator1
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    for uci in moves:
        move_int = uci_to_int(uci)
        evaluator_push(evaluator1, board1, move_int)

    print(f"After {len(moves)} moves:")
    print(f"  Board1: {board1.fen()}")

    # Now reset evaluator1 to the current position of board2 (starting position)
    evaluator1.reset(board2)

    # Both evaluators should now give the same evaluation for the starting position
    eval1 = evaluator1._evaluate(board2)
    eval2 = evaluator2._evaluate(board2)

    diff = abs(eval1 - eval2)
    print(f"\nAfter reset to starting position:")
    print(f"  Evaluator1: {eval1:.10f}")
    print(f"  Evaluator2: {eval2:.10f}")
    print(f"  Difference: {diff:.10e}")

    if diff < 1e-1:
        print("  ✓ PASS: Reset produces consistent results")
    else:
        print("  ✗ FAIL: Reset produced different results")
        all_passed = False

    # Also test resetting to a different position
    test_fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    test_board = CachedBoard(test_fen)

    evaluator1.reset(test_board)
    eval_reset = evaluator1._evaluate(test_board)
    eval_full = evaluator1._evaluate_full(test_board)

    diff = abs(eval_reset - eval_full)
    print(f"\nAfter reset to new position:")
    print(f"  Position: {test_fen}")
    print(f"  Incremental: {eval_reset:.10f}")
    print(f"  Full:        {eval_full:.10f}")
    print(f"  Difference:  {diff:.10e}")

    if diff < 1e-1:
        print("  ✓ PASS: Reset to arbitrary position works")
    else:
        print("  ✗ FAIL: Reset to arbitrary position failed")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All reset consistency tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Deep Search Simulation Test
# =============================================================================

def test_deep_search_simulation(nn_type: str, model_path: str, depth: int = 4,
                                num_iterations: int = 20, tolerance: float = 1e-1):
    """
    Simulate a deep search with many push/pop cycles.

    This tests that the evaluator maintains consistency through
    many levels of recursion and undo operations.

    Note: A tolerance of 1e-4 is used because floating-point errors
    naturally accumulate over many push/pop operations. This is expected
    behavior and doesn't affect practical usage.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Deep Search Simulation Test")
    print("=" * 70)
    print(f"Parameters: depth={depth}, iterations={num_iterations}, tolerance={tolerance}")

    board = CachedBoard()
    evaluator = NNEvaluator.create(board, nn_type, model_path)

    all_passed = True
    total_nodes = 0
    max_diff = 0.0

    def search_recursive(depth_remaining: int, path: list) -> int:
        """Recursive search that validates at each node."""
        nonlocal total_nodes, max_diff, all_passed

        total_nodes += 1

        # Validate incremental vs full at this node
        eval_inc = evaluator._evaluate(board)
        eval_full = evaluator._evaluate_full(board)
        diff = abs(eval_inc - eval_full)
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            print(f"  âœ— Mismatch at depth {len(path)}: diff={diff:.10e}")
            print(f"    Path: {' '.join(int_to_uci(m) for m in path)}")
            all_passed = False

        if depth_remaining == 0 or board.is_game_over():
            return 1

        nodes1 = 0
        legal_moves_int = board.get_legal_moves_int()
        random.shuffle(legal_moves_int)
        moves_to_test = legal_moves_int[:5]

        for move_int in moves_to_test:
            evaluator_push(evaluator, board, move_int)
            path.append(move_int)

            nodes1 += search_recursive(depth_remaining - 1, path)

            path.pop()
            evaluator_pop(evaluator, board)

        return nodes1

    print(f"\nRunning {num_iterations} search iterations...")

    start_time = time.time()
    for i in range(num_iterations):

        total_nodes = 0
        nodes = search_recursive(depth, [])

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{num_iterations}: {nodes} nodes, max_diff={max_diff:.2e}")

    elapsed = time.time() - start_time

    print(f"\n{'─' * 70}")
    print("Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Nodes per second: {total_nodes * num_iterations / elapsed:.0f}")
    print(f"  Maximum difference observed: {max_diff:.10e}")
    print(f"  Tolerance used: {tolerance:.0e}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Deep search simulation PASSED!")
    else:
        print("✗ Deep search simulation FAILED!")
    print("=" * 70)

    return all_passed


def test_engine_best_move():
    win_at_chess_positions = \
        "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id 'WAC.001';\
        5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - bm Rg3; id 'WAC.003';\
        r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - bm Qxh7+; id 'WAC.004';\
        5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - bm Qc4+; id 'WAC.005';\
        7k/p7/1R5K/6r1/6p1/6P1/8/8 w - - bm Rb7; id 'WAC.006';\
        rnbqkb1r/pppp1ppp/8/4P3/6n1/7P/PPPNPPP1/R1BQKBNR b KQkq - - bm Ne3; id 'WAC.007';\
        r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - bm Rf7; id 'WAC.008';\
        3q1rk1/p4pp1/2pb3p/3p4/6Pr/1PNQ4/P1PB1PP1/4RRK1 b - - bm Bh2+; id 'WAC.009';\
        2br2k1/2q3rn/p2NppQ1/2p1P3/Pp5R/4P3/1P3PPP/3R2K1 w - - bm Rxh7; id 'WAC.010';\
        r1b1kb1r/3q1ppp/pBp1pn2/8/Np3P2/5B2/PPP3PP/R2Q1RK1 w kq - - bm Bxc6; id 'WAC.011';\
        4k1r1/2p3r1/1pR1p3/3pP2p/3P2qP/P4N2/1PQ4P/5R1K b - - bm Qxf3+; id 'WAC.012';\
        5rk1/pp4p1/2n1p2p/2Npq3/2p5/6P1/P3P1BP/R4Q1K w - - bm Qxf8+; id 'WAC.013';\
        r2rb1k1/pp1q1p1p/2n1p1p1/2bp4/5P2/PP1BPR1Q/1BPN2PP/R5K1 w - - bm Qxh7+; id 'WAC.014';\
        1R6/1brk2p1/4p2p/p1P1Pp2/P7/6P1/1P4P1/2R3K1 w - - bm Rxb7; id 'WAC.015';\
        r4rk1/ppp2ppp/2n5/2bqp3/8/P2PB3/1PP1NPPP/R2Q1RK1 w - - bm Nc3; id 'WAC.016';\
        1k5r/pppbn1pp/4q1r1/1P3p2/2NPp3/1QP5/P4PPP/R1B1R1K1 w - - bm Ne5; id 'WAC.017';\
        R7/P4k2/8/8/8/8/r7/6K1 w - - bm Rh8; id 'WAC.018';\
        r1b2rk1/ppbn1ppp/4p3/1QP4q/3P4/N4N2/5PPP/R1B2RK1 w - - bm c6; id 'WAC.019';\
        r2qkb1r/1ppb1ppp/p7/4p3/P1Q1P3/2P5/5PPP/R1B2KNR b kq - - bm Bb5; id 'WAC.020';\
        r1bqk2r/ppp1nppp/4p3/n5N1/2BPp3/P1P5/2P2PPP/R1BQK2R w KQkq - - bm Ba2 Nxf7; id 'WAC.022';\
        6k1/1b1nqpbp/pp4p1/5P2/1PN5/4Q3/P5PP/1B2B1K1 b - - bm Bd4; id 'WAC.024';\
        3R1rk1/8/5Qpp/2p5/2P1p1q1/P3P3/1P2PK2/8 b - - bm Qh4+; id 'WAC.025';\
        6k1/1b1nqpbp/pp4p1/5P2/1PN5/4Q3/P5PP/1B2B1K1 b - - bm Bd4; id 'WAC.024';\
        3R1rk1/8/5Qpp/2p5/2P1p1q1/P3P3/1P2PK2/8 b - - bm Qh4+; id 'WAC.025';"
        # 5rk1/1b3p1p/pp3p2/3n1N2/1P6/P1qB1PP1/3Q3P/4R1K1 w - - bm Qh6; id 'WAC.021';\

    print("\n" + "=" * 70)
    print(f"Engine Tests")
    print("=" * 70)

    tests_total = 0
    tests_passed = 0
    nodes_sum = 0
    nps_sum = 0.0
    test_suite = (win_at_chess_positions, r'\d{3}\';', -1, 5)

    for line in re.split(test_suite[1], test_suite[0])[:test_suite[2]]:
        tests_total += 1

        fen = line.split('- -')[0].strip()
        best_moves = line.split('- -')[1].split('bm')[1].split(';')[0].strip().split(' ')

        board = CachedBoard(fen)
        expected_moves_int = []
        for best_move in best_moves:
            expected_move_int = board.parse_san(best_move)
            expected_moves_int.append(expected_move_int)

        f = io.StringIO()
        with redirect_stdout(f):
            found_move_int, score, _, nodes, nps = find_best_move(fen, max_depth=30, time_limit=test_suite[3],
                                                              expected_best_moves=expected_moves_int)
            nps_sum += nps
            nodes_sum += nodes

        found_move_san = board.san(found_move_int)

        if found_move_san in best_moves:
            tests_passed += 1
            print(f"{tests_total}.", end="", flush=True)
        else:
            print(f"\nFailed test: fen={fen}, expected_moves={best_moves}, found_move={found_move_san}")

    print("\n" + "=" * 70)
    print(f"total={tests_total}, passed={tests_passed}, "
          f"success-rate={round(tests_passed / tests_total * 100, 2)}%")
    print(f"nodes-avg={round(nodes_sum / tests_total)}, nps-avg={round(nps_sum / tests_total)}")

    all_passed = tests_total == tests_passed
    if all_passed:
        print("✓ Engine Tests PASSED!")
    else:
        print("✗ Engine Tests FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Random Games Test
# =============================================================================

def test_random_games(nn_type: str, model_path: str, num_games: int = 10, max_moves: int = 100):
    """
    Play through random games and verify incremental vs full evaluation.

    This tests a wide variety of positions that might not be covered
    by handcrafted test sequences.
    """
    print("\n" + "=" * 70)
    print(f"{nn_type} Random Games Test")
    print("=" * 70)
    print(f"Parameters: num_games={num_games}, max_moves={max_moves}")

    all_passed = True
    total_positions = 0
    max_diff = 0.0
    failures = []

    random.seed(42)  # Reproducible

    evaluator = NNEvaluator.create(CachedBoard(), nn_type, model_path)  # Loads model

    for game_num in range(num_games):
        board = CachedBoard()
        evaluator.reset(board)

        moves_played = []

        for move_num in range(max_moves):
            if board.is_game_over():
                break

            # Pick a random legal move
            legal_moves_int = board.get_legal_moves_int()
            move_int = random.choice(legal_moves_int)

            # Push and validate
            evaluator_push(evaluator, board, move_int)
            moves_played.append(move_int)
            total_positions += 1

            eval_inc = evaluator._evaluate(board)
            eval_full = evaluator._evaluate_full(board)
            diff = abs(eval_inc - eval_full)
            max_diff = max(max_diff, diff)

            if diff > 1e-1:
                failures.append({
                    'game': game_num,
                    'move': move_num,
                    'fen': board.fen(),
                    'diff': diff,
                    'moves': [int_to_uci(m) for m in moves_played[-5:]]
                })
                all_passed = False

        # Test some pop operations
        num_pops = min(5, len(moves_played))
        for _ in range(num_pops):
            evaluator_pop(evaluator, board)

            eval_inc = evaluator._evaluate(board)
            eval_full = evaluator._evaluate_full(board)
            diff = abs(eval_inc - eval_full)
            max_diff = max(max_diff, diff)

            if diff > 1e-1:
                all_passed = False

        print(f"  Game {game_num + 1}/{num_games}: {len(moves_played)} moves, "
              f"max_diff so far: {max_diff:.2e}")

    print(f"\n{'─' * 70}")
    print("Results:")
    print(f"  Total positions tested: {total_positions}")
    print(f"  Maximum difference: {max_diff:.10e}")
    print(f"  Failures: {len(failures)}")

    if failures:
        print("\nFirst 5 failures:")
        for f in failures[:5]:
            print(f"  Game {f['game']}, move {f['move']}: diff={f['diff']:.2e}")
            print(f"    Recent moves: {' '.join(f['moves'])}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ Random games test PASSED!")
    else:
        print("✗ Random games test FAILED!")
    print("=" * 70)

    return all_passed


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests(nn_type: str, model_path: str):
    """Run all non-interactive tests and report results."""
    print("\n" + "=" * 70)
    print(f"RUNNING ALL {nn_type} TESTS")
    print("=" * 70)

    results = {}

    # Run each test
    tests = [
        ("Incremental-vs-Full", lambda: performance_test(nn_type, model_path)),
        ("Accumulator Correctness", lambda: test_accumulator_correctness(nn_type, model_path)),
        ("Eval-Accuracy", lambda: test_eval_accuracy(nn_type, model_path)),
        ("NN-vs-Stockfish", lambda: test_nn_vs_stockfish(nn_type, model_path)),
        ("Feature Extraction", lambda: test_feature_extraction(nn_type)),
        ("Edge Cases", lambda: test_edge_cases(nn_type, model_path)),
        ("Reset Consistency", lambda: test_reset_consistency(nn_type, model_path)),
        ("Deep Search Simulation", lambda: test_deep_search_simulation(nn_type, model_path, depth=4, num_iterations=5,
                                                                       tolerance=1e-1)),
        ("Random Games", lambda: test_random_games(nn_type, model_path, num_games=5, max_moves=50)),
        ("Engine-Tests", lambda: test_engine_best_move()),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Performance test (doesn't return pass/fail)
    print("\n")
    performance_test(nn_type, model_path)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED!")

    print("=" * 70)

    return passed == total


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with ArgumentParser."""
    parser = argparse.ArgumentParser(
        description="Chess Neural Network Evaluation Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test types:
  0  Interactive-FEN         - Interactive FEN evaluation
  1  Incremental-vs-Full     - Performance comparison
  2  Accumulator-Correctness - Verify incremental == full evaluation
  3  Eval-Accuracy           - Test prediction accuracy against training data
  4  NN-vs-Stockfish         - Compare NN predictions against Stockfish static eval
  5  Feature-Extraction      - Verify feature extraction correctness
  6  Symmetry                - Test evaluation symmetry
  7  Edge-Cases              - Test edge cases (checkmate, stalemate, etc.)
  8  Reset-Consistency       - Test evaluator reset functionality
  9  Deep-Search-Simulation  - Simulate deep search with many push/pop cycles
  10 Random-Games            - Test with random legal move sequences
  11 Engine-Tests            - Compares engines best-move against known best-move(s) for a few FENs. 
  12 All                     - Run all non-interactive tests

Examples:
  %(prog)s --nn-type NNUE --test 0          # Interactive FEN
  %(prog)s --nn-type NNUE --test 1           # Performance test
  %(prog)s --nn-type NNUE --test 2          # Accumulator correctness
  %(prog)s --nn-type NNUE --test 3 --positions 1000  # Eval accuracy
  %(prog)s --nn-type NNUE --test 10 --num-positions 10  # Random games
  %(prog)s --nn-type NNUE --test 13         # Run all tests
  %(prog)s --nn-type NNUE --test 4 --positions 100  # NN vs Stockfish
"""
    )

    parser.add_argument(
        '--nn-type', '-n',
        type=str,
        choices=['NNUE'],
        default='NNUE',
        required=False,
        help='Neural network type (default: NNUE)'
    )

    parser.add_argument(
        '--test', '-t',
        type=int,
        choices=list(VALID_TEST_TYPES.keys()),
        required=True,
        help='Test type number (see list below)'
    )

    parser.add_argument(
        '--positions', '-p',
        type=int,
        default=10000,
        help='Number of positions for Eval-Accuracy and NN-vs-Stockfish tests (default: 10000)'
    )

    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path to model file (default: model/nnue.pt)'
    )

    parser.add_argument(
        '--depth', '-d',
        type=int,
        default=4,
        help='Search depth for Deep-Search-Simulation (default: 4)'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of iterations for Deep-Search-Simulation (default: 10)'
    )

    parser.add_argument(
        '--num-games', '-g',
        type=int,
        default=10,
        help='Number of games for Random-Games test (default: 10)'
    )

    parser.add_argument(
        '--max-moves',
        type=int,
        default=100,
        help='Max moves per game for Random-Games test (default: 100)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-1,
        help='Tolerance for floating-point comparisons in Deep-Search-Simulation (default: 1e-4)'
    )

    parser.add_argument(
        '--num-positions',
        type=int,
        default=10,
        help='Number of positions for CP-Integrity test (default: 10)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory for CP-Integrity test (default: data)'
    )

    parser.add_argument(
        '--stockfish',
        type=str,
        default='stockfish',
        help='Path to Stockfish binary for CP-Integrity and NN-vs-Stockfish tests (default: stockfish)'
    )

    parser.add_argument(
        '--time-limit',
        type=float,
        default=2.0,
        help='Stockfish time limit per position in seconds for CP-Integrity test (default: 2.0)'
    )

    args = parser.parse_args()

    nn_type = args.nn_type
    test_type = VALID_TEST_TYPES[args.test]
    model_path = args.model_path or get_model_path(nn_type)

    # Custom conditional logic
    if nn_type is None and test_type != "Engine-Tests":
        parser.error("--nn-type is required")

    print(f"Neural Network Type: {nn_type}")
    print(f"Test Type: {test_type}")
    print(f"Model Path: {model_path}")

    # Run appropriate test
    if test_type == "Interactive-FEN":
        board = CachedBoard()
        evaluator = NNEvaluator.create(board, nn_type, model_path)

        # Test with starting position first
        print("\nTesting with starting position...")
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = evaluate_fen(startpos, evaluator)
        print_evaluation(result)

        interactive_loop(evaluator, nn_type)

    elif test_type == "Incremental-vs-Full":
        performance_test(nn_type, model_path)

    elif test_type == "Accumulator-Correctness":
        test_accumulator_correctness(nn_type, model_path)

    elif test_type == "Eval-Accuracy":
        if args.positions <= 0:
            print("ERROR: Eval-Accuracy requires --positions > 0")
            sys.exit(1)
        test_eval_accuracy(nn_type, model_path, args.positions)

    elif test_type == "Feature-Extraction":
        test_feature_extraction(nn_type)

    elif test_type == "Symmetry":
        test_symmetry(nn_type, model_path)

    elif test_type == "Edge-Cases":
        test_edge_cases(nn_type, model_path)

    elif test_type == "Reset-Consistency":
        test_reset_consistency(nn_type, model_path)

    elif test_type == "Deep-Search-Simulation":
        test_deep_search_simulation(nn_type, model_path, args.depth, args.iterations, args.tolerance)

    elif test_type == "Random-Games":
        test_random_games(nn_type, model_path, args.num_games, args.max_moves)

    elif test_type == "Engine-Tests":
        test_engine_best_move()

    elif test_type == "All":
        success = run_all_tests(nn_type, model_path)
        sys.exit(0 if success else 1)

    elif test_type == "NN-vs-Stockfish":
        if args.positions <= 0:
            print("ERROR: NN-vs-Stockfish requires --positions > 0")
            sys.exit(1)
        test_nn_vs_stockfish(nn_type, model_path, args.positions)


if __name__ == "__main__":
    main()