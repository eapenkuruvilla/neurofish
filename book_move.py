import random
from dataclasses import dataclass
from typing import Optional, List

import chess
import chess.polyglot


@dataclass
class BookMove:
    move: chess.Move
    weight: int
    learn: int = 0


# Global book reader
_book_reader: Optional[chess.polyglot.MemoryMappedReader] = None


def init_opening_book(book_path: str) -> bool:
    """Initialize the opening book."""
    global _book_reader

    if _book_reader:
        _book_reader.close()
        _book_reader = None

    if not book_path:
        print("info string No book path specified", flush=True)
        return False

    try:
        _book_reader = chess.polyglot.open_reader(book_path)
        print(f"info string Loaded opening book: {book_path}", flush=True)
        return True
    except FileNotFoundError:
        print(f"info string Book file not found: {book_path}", flush=True)
        return False
    except Exception as e:
        print(f"info string Failed to load book: {e}", flush=True)
        return False


def is_book_loaded() -> bool:
    """Check if a book is currently loaded."""
    return _book_reader is not None


def get_book_move(
        board: chess.Board,
        min_weight: int = 1,
        temperature: float = 1.0
) -> Optional[chess.Move]:
    """
    Get a move from the opening book with weighted random selection.

    Args:
        board: Current board position
        min_weight: Minimum weight threshold (filters out rare moves)
        temperature: Controls randomness (0 = best only, 1 = proportional, >1 = more random)

    Returns:
        Selected move or None if not in book
    """
    global _book_reader

    if _book_reader is None:
        return None

    try:
        entries = list(_book_reader.find_all(board))

        if not entries:
            return None

        # Filter by minimum weight
        entries = [e for e in entries if e.weight >= min_weight]

        if not entries:
            return None

        # Single entry - no choice needed
        if len(entries) == 1:
            return entries[0].move

        # Temperature-based selection
        if temperature == 0:
            return max(entries, key=lambda e: e.weight).move

        # Apply temperature to weights
        weights = [e.weight ** (1.0 / temperature) for e in entries]
        total = sum(weights)
        weights = [w / total for w in weights]

        # Weighted random selection
        selected = random.choices(entries, weights=weights, k=1)[0]
        return selected.move

    except Exception as e:
        print(f"info string Book lookup error: {e}", flush=True)
        return None


def get_all_book_moves(board: chess.Board) -> List[BookMove]:
    """Get all book moves for a position with their weights."""
    global _book_reader

    if _book_reader is None:
        return []

    try:
        entries = list(_book_reader.find_all(board))
        return [BookMove(e.move, e.weight, e.learn) for e in entries]
    except Exception:
        return []


def close_book():
    """Close the opening book."""
    global _book_reader
    if _book_reader:
        _book_reader.close()
        _book_reader = None
