#!/usr/bin/env python3
"""
prepare_data.py - Extract chess position features from PGN files into binary shards.

This script reads a .pgn.zst file and extracts features for both DNN and NNUE networks,
writing them to compressed binary shard files for efficient training.

Usage:
    python prepare_data.py --pgn-zst-file lichess_2023_01.pgn.zst --output-dir data/ --output-file-prefix jan2023

Output format:
    - DNN:  data/dnn/{prefix}_0001.bin.zst, data/dnn/{prefix}_0002.bin.zst, ...
    - NNUE: data/nnue/{prefix}_0001.bin.zst, data/nnue/{prefix}_0002.bin.zst, ...
    - Progress: data/{prefix}_progress.json

Binary shard format:
    DNN Normal record:
        [score:int16][num_features:uint8][features:uint16[num_features]]

    DNN Diagnostic record (every 1000 positions, first byte = 0xFF):
        [marker:uint8=0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[num_features]]
        [fen_length:uint8][fen_bytes:char[fen_length]]

    NNUE Normal record:
        [score:int16][stm:uint8][num_white:uint8][white_features:uint16[num_white]]
        [num_black:uint8][black_features:uint16[num_black]]

    NNUE Diagnostic record (every 1000 positions, first byte = 0xFF):
        [marker:uint8=0xFF][score:int16][stm:uint8][num_white:uint8][white_features:uint16[num_white]]
        [num_black:uint8][black_features:uint16[num_black]][fen_length:uint8][fen_bytes:char[fen_length]]

Feature encoding:
    - Piece order (shared by DNN and NNUE): P=0, N=1, B=2, R=3, Q=4, K=5 (piece_type - 1)
    - DNN: 768 features = 12 planes × 64 squares, feature_idx = piece_idx * 64 + square
    - NNUE: 40,960 features = king_sq * 640 + piece_sq * 10 + (type_idx + color_idx * 5)
"""

import argparse
import json
import os
import sys
import io
from typing import List, Optional, Dict, Any
from pathlib import Path

import chess.pgn
import zstandard as zstd

# Import from nn_inference to ensure consistency
from nn_inference import (
    NNUEFeatures, DNNFeatures, )
from config import MAX_SCORE

# Import ShardWriter from shared module
from nn_train.shard_io import ShardWriter

# Constants from nn_train.py
MATE_FACTOR = 100
MAX_MATE_DEPTH = 10
MAX_NON_MATE_SCORE = MAX_SCORE - MAX_MATE_DEPTH * MATE_FACTOR
OPENING_PLYS = 10
DEFAULT_POSITIONS_PER_SHARD = 1_000_000

def eval_to_cp_stm(ev, board_turn: bool) -> Optional[int]:
    """
    Convert a chess.engine evaluation to centipawns from side-to-move perspective.

    This matches the logic from nn_train.py ProcessGameWithValidation.eval_to_cp_stm

    Args:
        ev: The evaluation from node.eval() (can be None)
        board_turn: True if white to move, False if black to move

    Returns:
        Centipawn score from STM perspective, or None if ev is None
    """
    if ev is None:
        return None

    if ev.is_mate():
        mate_in = ev.white().mate()
        if mate_in < 0:  # -ve when black is winning
            mate_in = max(-MAX_MATE_DEPTH, mate_in)
            score_cp = -MAX_SCORE - mate_in * MATE_FACTOR
        else:
            mate_in = min(MAX_MATE_DEPTH, mate_in)
            score_cp = MAX_SCORE - mate_in * MATE_FACTOR
    else:
        score_cp = ev.white().score()
        score_cp = min(score_cp, MAX_NON_MATE_SCORE)
        score_cp = max(score_cp, -MAX_NON_MATE_SCORE)

    # Lichess eval is always from White's perspective - convert to STM
    if not board_turn:  # Black to move
        score_cp = -score_cp

    return score_cp


import chess


def process_game(game) -> List[Dict[str, Any]]:
    """
    Process a single game and extract positions with features.

    Filtering criteria (from ProcessGameWithValidation.__call__):
    - Skip variant games
    - Skip opening moves (first OPENING_PLYS moves)
    - Skip game-over positions
    - Skip positions where side-to-move is in check
    - Skip positions after captures (tactically unstable)
    - Only include positions with valid eval comments

    Returns:
        List of dicts with keys: 'score_cp', 'dnn_features', 'nnue_white', 'nnue_black', 'stm', 'fen'
    """
    if game is None:
        return []

    # Skip variant games
    if any("Variant" in key for key in game.headers.keys()):
        return []

    positions = []
    board = game.board()
    move_count = 0

    for node in game.mainline():
        move_count += 1
        current_move = node.move

        # Check if this move is a capture BEFORE pushing it
        was_last_move_capture = board.is_capture(current_move)

        if move_count <= OPENING_PLYS:
            board.push(current_move)
            continue

        board.push(current_move)

        if board.is_game_over():
            continue

        # Skip if side to move is in check
        if board.is_check():
            continue

        # Skip if this move was a capture (position after capture is tactically unstable)
        if was_last_move_capture:
            continue

        # Get evaluation from node
        score_cp = eval_to_cp_stm(node.eval(), board.turn == chess.WHITE)
        if score_cp is None:
            continue

        # Get FEN for diagnostic records
        fen = board.fen()

        # Extract DNN features (from perspective of side to move)
        dnn_features = DNNFeatures.extract_features(board, board.turn == chess.WHITE)

        # Extract NNUE features (white and black perspectives)
        nnue_white = NNUEFeatures.extract_features(board, chess.WHITE)
        nnue_black = NNUEFeatures.extract_features(board, chess.BLACK)

        # Side to move: 1 for white, 0 for black
        stm = 1 if board.turn == chess.WHITE else 0

        positions.append({
            'score_cp': score_cp,
            'dnn_features': dnn_features,
            'nnue_white': nnue_white,
            'nnue_black': nnue_black,
            'stm': stm,
            'fen': fen
        })

    return positions


class ProgressTracker:
    """Track progress for resume capability."""

    def __init__(self, output_dir: str, prefix: str):
        self.progress_file = Path(output_dir) / f"{prefix}_progress.json"
        self.progress = {
            'pgn_file': '',
            'games_processed': 0,
            'total_positions_written': 0,
            'dnn_shard_count': 0,
            'nnue_shard_count': 0,
            'completed': False
        }

    def load(self) -> bool:
        """Load existing progress. Returns True if progress file exists."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
            return True
        return False

    def save(self):
        """Save current progress."""
        # Ensure directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def update(self, games_processed: int, dnn_stats: Dict, nnue_stats: Dict):
        """Update progress stats."""
        self.progress['games_processed'] = games_processed
        self.progress['total_positions_written'] = dnn_stats['total_positions']
        self.progress['dnn_shard_count'] = dnn_stats['num_shards']
        self.progress['nnue_shard_count'] = nnue_stats['num_shards']

    def mark_completed(self):
        """Mark processing as completed."""
        self.progress['completed'] = True
        self.save()

    def is_completed(self) -> bool:
        return self.progress.get('completed', False)

    def get_games_processed(self) -> int:
        return self.progress.get('games_processed', 0)


def process_pgn_file(
        pgn_file: str,
        output_dir: str,
        prefix: str,
        positions_per_shard: int,
        resume: bool
) -> Dict[str, Any]:
    """
    Process a PGN file and write features to shard files.

    Args:
        pgn_file: Path to .pgn.zst file
        output_dir: Output directory for shards
        prefix: Prefix for output files
        positions_per_shard: Number of positions per shard
        resume: Whether to resume from existing progress

    Returns:
        Statistics dict
    """
    # Initialize progress tracker
    tracker = ProgressTracker(output_dir, prefix)

    # Check for existing progress
    if tracker.load():
        if tracker.is_completed():
            print(f"Processing already completed for {prefix}. Use different prefix or delete progress file.")
            return {'status': 'already_completed'}

        if not resume:
            print(f"Error: Progress file exists at {tracker.progress_file}")
            print("Use --resume to continue from where you left off, or delete the progress file to start fresh.")
            sys.exit(1)

        print(f"Resuming from game {tracker.get_games_processed()}")
        skip_games = tracker.get_games_processed()
    else:
        if resume:
            print("Warning: --resume specified but no progress file found. Starting fresh.")
        skip_games = 0

    # Store PGN file info
    tracker.progress['pgn_file'] = os.path.basename(pgn_file)

    # Initialize shard writers
    dnn_writer = ShardWriter(output_dir, prefix, "DNN", positions_per_shard)
    nnue_writer = ShardWriter(output_dir, prefix, "NNUE", positions_per_shard)

    # If resuming, we need to account for already-written shards
    if skip_games > 0:
        # Count existing shards
        dnn_dir = Path(output_dir) / "dnn"
        nnue_dir = Path(output_dir) / "nnue"

        existing_dnn_shards = len(list(dnn_dir.glob(f"{prefix}_*.bin.zst")))
        existing_nnue_shards = len(list(nnue_dir.glob(f"{prefix}_*.bin.zst")))

        dnn_writer.current_shard_num = existing_dnn_shards
        nnue_writer.current_shard_num = existing_nnue_shards
        dnn_writer.total_positions = tracker.progress.get('total_positions_written', 0)
        nnue_writer.total_positions = tracker.progress.get('total_positions_written', 0)

    games_processed = skip_games
    positions_extracted = dnn_writer.total_positions

    # Progress reporting interval
    report_interval = 1000
    save_interval = 10000  # Save progress every N games

    print(f"\nProcessing: {pgn_file}")
    print(f"Output prefix: {prefix}")
    print(f"Positions per shard: {positions_per_shard:,}")
    print("-" * 60)

    try:
        with open(pgn_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')

                game_num = 0
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break

                    game_num += 1

                    # Skip games if resuming
                    if game_num <= skip_games:
                        continue

                    # Process game
                    positions = process_game(game)

                    # Write positions to both DNN and NNUE shards
                    for pos in positions:
                        dnn_writer.add_position(pos)
                        nnue_writer.add_position(pos)

                    games_processed += 1
                    positions_extracted += len(positions)

                    # Progress reporting
                    if games_processed % report_interval == 0:
                        dnn_stats = dnn_writer.get_stats()
                        print(f"  Games: {games_processed:,} | Positions: {positions_extracted:,} | "
                              f"DNN shards: {dnn_stats['num_shards']} | "
                              f"NNUE shards: {nnue_writer.get_stats()['num_shards']}")

                    # Save progress periodically
                    if games_processed % save_interval == 0:
                        tracker.update(games_processed, dnn_writer.get_stats(), nnue_writer.get_stats())
                        tracker.save()

        # Finalize shards
        dnn_writer.finalize()
        nnue_writer.finalize()

        # Mark as completed
        tracker.update(games_processed, dnn_writer.get_stats(), nnue_writer.get_stats())
        tracker.mark_completed()

        # Final stats
        dnn_stats = dnn_writer.get_stats()
        nnue_stats = nnue_writer.get_stats()

        print("-" * 60)
        print(f"Completed!")
        print(f"  Total games processed: {games_processed:,}")
        print(f"  Total positions extracted: {positions_extracted:,}")
        print(f"  DNN shards written: {dnn_stats['num_shards']}")
        print(f"  NNUE shards written: {nnue_stats['num_shards']}")
        print(f"  Output directories:")
        print(f"    DNN:  {Path(output_dir) / 'dnn'}")
        print(f"    NNUE: {Path(output_dir) / 'nnue'}")

        return {
            'status': 'completed',
            'games_processed': games_processed,
            'positions_extracted': positions_extracted,
            'dnn_shards': dnn_stats['num_shards'],
            'nnue_shards': nnue_stats['num_shards']
        }

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        dnn_writer.finalize()
        nnue_writer.finalize()
        tracker.update(games_processed, dnn_writer.get_stats(), nnue_writer.get_stats())
        tracker.save()
        print(f"Progress saved. Resume with --resume flag.")
        return {'status': 'interrupted', 'games_processed': games_processed}

    except Exception as e:
        print(f"\nError: {e}")
        # Save progress before exiting
        tracker.update(games_processed, dnn_writer.get_stats(), nnue_writer.get_stats())
        tracker.save()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Extract chess position features from PGN files into binary shards.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single PGN file
    python prepare_data.py --pgn-zst-file pgn/lichess_2023_01.pgn.zst \\
                           --output-dir data/ --output-file-prefix jan2023

    # Resume interrupted processing
    python prepare_data.py --pgn-zst-file pgn/lichess_2023_01.pgn.zst \\
                           --output-dir data/ --output-file-prefix jan2023 --resume

Output files:
    data/dnn/jan2023_0001.bin.zst, data/dnn/jan2023_0002.bin.zst, ...
    data/nnue/jan2023_0001.bin.zst, data/nnue/jan2023_0002.bin.zst, ...
    data/jan2023_progress.json
"""
    )

    parser.add_argument(
        '--pgn-zst-file',
        type=str,
        required=True,
        help='Path to the .pgn.zst file to process'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/',
        help='Output directory for shard files (default: data/)'
    )

    parser.add_argument(
        '--output-file-prefix',
        type=str,
        required=True,
        help='Prefix for output shard files (e.g., "jan2023")'
    )

    parser.add_argument(
        '--positions-per-shard',
        type=int,
        default=DEFAULT_POSITIONS_PER_SHARD,
        help=f'Number of positions per shard file (default: {DEFAULT_POSITIONS_PER_SHARD:,})'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing progress file'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.pgn_zst_file):
        print(f"Error: PGN file not found: {args.pgn_zst_file}")
        sys.exit(1)

    if not args.pgn_zst_file.endswith('.pgn.zst'):
        print(f"Warning: File does not have .pgn.zst extension: {args.pgn_zst_file}")

    # Process the file
    stats = process_pgn_file(
        pgn_file=args.pgn_zst_file,
        output_dir=args.output_dir,
        prefix=args.output_file_prefix,
        positions_per_shard=args.positions_per_shard,
        resume=args.resume
    )

    if stats.get('status') == 'already_completed':
        sys.exit(0)
    elif stats.get('status') == 'interrupted':
        sys.exit(1)


if __name__ == "__main__":
    main()