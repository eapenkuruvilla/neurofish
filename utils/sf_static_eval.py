#!/usr/bin/env python3
"""
build_sf_static_eval_file.py - Pre-compute Stockfish NNUE static evaluations.

This script reads FEN positions from training shard files, evaluates them with
Stockfish's static NNUE evaluation, and saves the results to a binary file.
This allows Test 4 (NN-vs-Stockfish) to run much faster by avoiding repeated
Stockfish subprocess calls.

Binary output format:
    Header:
        [num_records:uint32]
    Each record:
        [fen_length:uint8][fen_bytes:char[]][sf_eval_cp:int16][shard_cp:int16]

Usage:
    python build_sf_static_eval_file.py --num-positions 10000
    python build_sf_static_eval_file.py --num-positions 5000 --output data/sf_eval.bin
"""

import argparse
import os
import random
import struct
import subprocess
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

from nn_train.shard_io import ShardReader, discover_shards

# Import MAX_SCORE from nn_inference
try:
    from config import MAX_SCORE
except ImportError:
    MAX_SCORE = 10000  # Fallback value

# Constants
DEFAULT_OUTPUT = "data/sf_nnue_static_eval.bin"
DEFAULT_NUM_POSITIONS = 10000
DEFAULT_DATA_DIR = "data/dnn"
DEFAULT_SEED = 42
MATE_SCORE_CP = MAX_SCORE  # Use MAX_SCORE for mate scores


def get_stockfish_static_eval(fen: str, stockfish_path: str = "stockfish") -> Optional[int]:
    """
    Get Stockfish's static NNUE evaluation for a position.

    Returns evaluation in centipawns from WHITE's perspective,
    or None if evaluation fails.
    """
    try:
        commands = f"uci\nisready\nposition fen {fen}\neval\nquit\n"
        result = subprocess.run(
            [stockfish_path],
            input=commands,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Parse NNUE evaluation from output
        for line in result.stdout.split('\n'):
            if line.startswith("NNUE evaluation"):
                # Format: "NNUE evaluation        +0.26 (white side)"
                parts = line.split()
                if len(parts) >= 3:
                    score_str = parts[2]
                    # Convert to centipawns (score is in pawns)
                    score_cp = float(score_str) * 100
                    return int(round(score_cp))
        return None
    except Exception as e:
        return None


def collect_diagnostic_fens(data_dir: str, reader: ShardReader, target_count: int) -> List[Dict]:
    """
    Collect diagnostic records (with FEN) from shard files.

    Stops early once enough unique positions are collected (with 2x buffer for deduplication).

    Args:
        data_dir: Directory containing shard files
        reader: ShardReader instance
        target_count: Target number of positions needed

    Returns list of dicts with 'fen' and 'score_cp' keys.
    """
    shards = discover_shards(data_dir, "DNN")

    if not shards:
        # Try without nn_type subdirectory
        import glob
        pattern = os.path.join(data_dir, "*.bin.zst")
        shards = sorted(glob.glob(pattern), reverse=True)  # Use the last shards for testing

    if not shards:
        raise FileNotFoundError(f"No shard files found in {data_dir}")

    print(f"Found {len(shards)} shard files in {data_dir}")

    # Shuffle shards for random sampling across different shards
    # shards_reversed = shards.copy()
    # random.shuffle(shards_reversed)

    # Collect with early stopping - aim for 2x target to allow for deduplication
    collection_target = target_count * 2
    seen_fens = set()
    unique_records = []

    pbar = tqdm(shards, desc="Scanning shards for diagnostic records")
    for shard_path in pbar:
        records = reader.read_diagnostic_records(shard_path, max_records=10000)
        for rec in records:
            if 'fen' in rec and rec['fen'] not in seen_fens:
                seen_fens.add(rec['fen'])
                unique_records.append({
                    'fen': rec['fen'],
                    'score_cp': rec['score_cp']
                })

        pbar.set_postfix({'unique': len(unique_records)})

        # Early stopping if we have enough
        if len(unique_records) >= collection_target:
            print(f"\nCollected {len(unique_records)} unique positions, stopping early")
            break

    return unique_records


def deduplicate_by_fen(records: List[Dict]) -> List[Dict]:
    """Remove duplicate FENs, keeping the first occurrence."""
    seen = set()
    unique = []
    for rec in records:
        if rec['fen'] not in seen:
            seen.add(rec['fen'])
            unique.append(rec)
    return unique


def write_binary_file(output_path: str, records: List[Tuple[str, int, int]]):
    """
    Write records to binary file.

    Args:
        output_path: Path to output file
        records: List of (fen, sf_eval_cp, shard_cp) tuples
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        # Write header: number of records
        f.write(struct.pack('<I', len(records)))

        # Write each record
        for fen, sf_eval_cp, shard_cp in records:
            fen_bytes = fen.encode('utf-8')
            f.write(struct.pack('<B', len(fen_bytes)))
            f.write(fen_bytes)
            f.write(struct.pack('<h', sf_eval_cp))
            f.write(struct.pack('<h', shard_cp))

    print(f"Wrote {len(records)} records to {output_path}")


def read_sf_eval_file(filepath: str) -> List[Tuple[str, int, int]]:
    """
    Read the Stockfish evaluation binary file.

    Returns list of (fen, sf_eval_cp, shard_cp) tuples.
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute Stockfish NNUE static evaluations for test positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --num-positions 10000
    %(prog)s --num-positions 5000 --output data/sf_eval.bin
    %(prog)s --data-dir data/dnn --stockfish /usr/local/bin/stockfish
"""
    )

    parser.add_argument(
        '--num-positions', '-n',
        type=int,
        default=DEFAULT_NUM_POSITIONS,
        help=f'Number of positions to evaluate (default: {DEFAULT_NUM_POSITIONS})'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Output file path (default: {DEFAULT_OUTPUT})'
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Directory containing shard files (default: {DEFAULT_DATA_DIR})'
    )

    parser.add_argument(
        '--stockfish', '-s',
        type=str,
        default=os.environ.get('STOCKFISH_PATH', 'stockfish'),
        help='Path to Stockfish binary (default: $STOCKFISH_PATH or "stockfish")'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_SEED})'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 70)
    print("Stockfish Static Evaluation File Builder")
    print("=" * 70)
    print(f"Num positions:  {args.num_positions}")
    print(f"Output file:    {args.output}")
    print(f"Data directory: {args.data_dir}")
    print(f"Stockfish:      {args.stockfish}")
    print(f"Random seed:    {args.seed}")
    print("=" * 70)

    # Test Stockfish availability
    print("\nTesting Stockfish...")
    test_eval = get_stockfish_static_eval(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        args.stockfish
    )
    if test_eval is None:
        print(f"ERROR: Could not get evaluation from Stockfish at '{args.stockfish}'")
        return 1
    print(f"✓ Stockfish working (startpos eval: {test_eval:+d} cp)")

    # Collect diagnostic records from shards (with early stopping)
    print("\nCollecting positions from shards...")
    reader = ShardReader("DNN")
    unique_records = collect_diagnostic_fens(args.data_dir, reader, args.num_positions)
    print(f"Collected {len(unique_records)} unique positions")

    if len(unique_records) < args.num_positions:
        print(f"WARNING: Only {len(unique_records)} unique positions available, "
              f"requested {args.num_positions}")
        selected = unique_records
    else:
        # Random sample from collected positions
        selected = random.sample(unique_records, args.num_positions)
    print(f"Selected {len(selected)} positions for evaluation")

    # Evaluate with Stockfish
    print("\nEvaluating positions with Stockfish...")
    results = []
    errors = 0

    for rec in tqdm(selected, desc="Stockfish evaluation"):
        fen = rec['fen']
        shard_cp = rec['score_cp']

        sf_eval = get_stockfish_static_eval(fen, args.stockfish)

        if sf_eval is None:
            errors += 1
            continue

        # Determine side to move from FEN
        stm_is_white = ' w ' in fen

        # Stockfish eval is from white's perspective, convert to STM perspective
        if not stm_is_white:
            sf_eval_stm = -sf_eval
        else:
            sf_eval_stm = sf_eval

        # Cap at MAX_SCORE
        sf_eval_capped = max(-MAX_SCORE, min(MAX_SCORE, sf_eval_stm))

        results.append((fen, sf_eval_capped, shard_cp))

    if errors > 0:
        print(f"WARNING: {errors} positions failed Stockfish evaluation")

    print(f"\nSuccessfully evaluated {len(results)} positions")

    # Write output file
    write_binary_file(args.output, results)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
