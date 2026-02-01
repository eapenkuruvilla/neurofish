"""
CachedBoard - Efficient chess board wrapper with intelligent caching.

OPTIMIZED VERSION

Optimizations:
- Inlined _cache property access (eliminates 2.8M property calls)
- Move conversion caching in MoveAdapter
- Integer-based move keys for fast hashing
- Pre-computed lookup tables
"""

import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

try:
    import libs.chess_cpp as chess_cpp

    HAS_CPP_BACKEND = True
    print("✓ Using fast C++ chess backend (chess_cpp)", file=sys.stderr)
except ImportError:
    HAS_CPP_BACKEND = False
    print("! C++ backend not available, using python-chess (slower)")

import chess
import chess.polyglot

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# OPTIMIZATION: Pre-computed MVV-LVA table as nested list (2x faster than dict)
# Index: [victim_type][attacker_type], piece types 1-6
# Value: 10 * victim_value - attacker_value
_MVV_LVA = [[0] * 7 for _ in range(7)]
for _v in range(1, 7):
    for _a in range(1, 7):
        _MVV_LVA[_v][_a] = 10 * PIECE_VALUES.get(_v, 0) - PIECE_VALUES.get(_a, 0)

# fmt: off
_PST_PAWN = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_PST_ROOK = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PST_QUEEN = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_PST_KING_MG = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_PST_KING_EG = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]
# fmt: on

_PST_TABLES = {
    chess.PAWN: _PST_PAWN,
    chess.KNIGHT: _PST_KNIGHT,
    chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK,
    chess.QUEEN: _PST_QUEEN,
    chess.KING: _PST_KING_MG,
}


def _get_pst_value(piece_type: int, square: int, color: bool, is_endgame: bool = False) -> int:
    if piece_type == chess.KING:
        table = _PST_KING_EG if is_endgame else _PST_KING_MG
    else:
        table = _PST_TABLES.get(piece_type)
        if table is None:
            return 0
    if color == chess.BLACK:
        square = chess.square_mirror(square)
    return table[square]


@dataclass(slots=True)
class _CacheState:
    zobrist_hash: Optional[int] = None
    legal_moves: Optional[List[chess.Move]] = None
    legal_moves_int: Optional[List[int]] = None  # Phase 1: Integer move list
    has_non_pawn_material: Optional[Dict[bool, bool]] = None
    is_check: Optional[bool] = None
    is_checkmate: Optional[bool] = None
    is_game_over: Optional[bool] = None
    material_evaluation: Optional[int] = None
    is_endgame: Optional[bool] = None
    move_is_capture: Optional[Dict[chess.Move, bool]] = None
    move_gives_check: Optional[Dict[chess.Move, bool]] = None
    move_victim_type: Optional[Dict[chess.Move, Optional[int]]] = None
    move_attacker_type: Optional[Dict[chess.Move, Optional[int]]] = None
    # Integer-keyed move info caches
    move_is_capture_int: Optional[Dict[int, bool]] = None
    move_gives_check_int: Optional[Dict[int, bool]] = None
    move_victim_type_int: Optional[Dict[int, Optional[int]]] = None
    move_attacker_type_int: Optional[Dict[int, Optional[int]]] = None
    # OPTIMIZATION: Additional caches to eliminate piece_at() calls
    move_is_en_passant_int: Optional[Dict[int, bool]] = None
    move_is_castling_int: Optional[Dict[int, bool]] = None
    move_piece_color_int: Optional[Dict[int, bool]] = None  # True = WHITE
    move_captured_piece_type_int: Optional[Dict[int, Optional[int]]] = None  # For non-ep captures
    move_captured_piece_color_int: Optional[Dict[int, Optional[bool]]] = None
    # OPTIMIZATION: Pre-computed MVV-LVA scores (from MVV-LVA optimization)
    move_mvv_lva_int: Optional[Dict[int, int]] = None


# OPTIMIZATION: Pre-computed bitmasks for fast capture detection
_SQUARE_MASKS = tuple(1 << sq for sq in range(64))


@dataclass(slots=True)
class _MoveInfo:
    move: chess.Move
    captured_piece: Optional[chess.Piece] = None
    was_en_passant: bool = False
    was_castling: bool = False
    previous_castling_rights: int = 0
    previous_ep_square: Optional[int] = None


def move_to_int(move: chess.Move) -> int:
    """Convert move to integer key for fast hashing."""
    promo = move.promotion if move.promotion else 0
    return move.from_square | (move.to_square << 6) | (promo << 12)


def int_to_move(key: int) -> chess.Move:
    """Convert integer key back to Move"""
    from_sq = key & 0x3F
    to_sq = (key >> 6) & 0x3F
    promo = (key >> 12) & 0xF
    return chess.Move(from_sq, to_sq, promo if promo else None)


class MoveAdapter:
    """OPTIMIZED: Adapter with move caching to reduce object creation."""

    _to_py_cache: Dict[tuple, chess.Move] = {}
    _to_cpp_cache: Dict[tuple, Any] = {}
    _MAX_CACHE_SIZE = 50000

    @classmethod
    def to_chess_move(cls, cpp_move: Any, is_castling: bool = False) -> chess.Move:
        """Convert C++ move to python-chess Move, normalizing castling representation.

        The C++ backend represents castling as king-to-rook-square:
        - Kingside: e1->h1 (white) or e8->h8 (black)
        - Queenside: e1->a1 (white) or e8->a8 (black)

        python-chess represents castling as king-to-destination:
        - Kingside: e1->g1 (white) or e8->g8 (black)
        - Queenside: e1->c1 (white) or e8->c8 (black)

        Args:
            cpp_move: The C++ move object
            is_castling: If True, apply castling conversion. This should be determined
                        by calling board.is_castling(cpp_move) before conversion.
        """
        if not HAS_CPP_BACKEND:
            return cpp_move

        from_sq = cpp_move.from_square
        to_sq = cpp_move.to_square
        promo = cpp_move.promotion if cpp_move.promotion > 0 else None

        # Only convert if explicitly told this is a castling move
        if is_castling and promo is None:
            # White castling (king starts on e1 = square 4)
            if from_sq == chess.E1:
                if to_sq == chess.H1:  # C++ kingside castling
                    to_sq = chess.G1  # Convert to python-chess convention
                elif to_sq == chess.A1:  # C++ queenside castling
                    to_sq = chess.C1  # Convert to python-chess convention

            # Black castling (king starts on e8 = square 60)
            elif from_sq == chess.E8:
                if to_sq == chess.H8:  # C++ kingside castling
                    to_sq = chess.G8  # Convert to python-chess convention
                elif to_sq == chess.A8:  # C++ queenside castling
                    to_sq = chess.C8  # Convert to python-chess convention

        # Use cache with the CONVERTED coordinates
        key = (from_sq, to_sq, promo if promo else 0)
        cached = cls._to_py_cache.get(key)
        if cached is not None:
            return cached

        py_move = chess.Move(from_sq, to_sq, promo)

        if len(cls._to_py_cache) < cls._MAX_CACHE_SIZE:
            cls._to_py_cache[key] = py_move

        return py_move

    @classmethod
    def from_chess_move(cls, py_move: chess.Move, is_castling: bool = None) -> Any:
        """Convert python-chess Move to C++ move.

        For castling moves, we need to convert from python-chess convention
        (king-to-destination) to C++ convention (king-to-rook-square).

        Args:
            py_move: The python-chess Move object
            is_castling: If True, apply castling conversion. If False, don't convert.
                        If None (default), auto-detect based on move pattern (legacy behavior,
                        but may incorrectly convert non-castling moves from e1/e8).
        """
        if not HAS_CPP_BACKEND:
            return py_move

        from_sq = py_move.from_square
        to_sq = py_move.to_square
        promo = py_move.promotion if py_move.promotion else 0

        # Only apply castling conversion if explicitly told it's castling,
        # or if is_castling is None and move pattern matches castling
        apply_conversion = is_castling if is_castling is not None else True

        if apply_conversion and promo == 0:
            # White castling (king starts on e1 = square 4)
            if from_sq == chess.E1:
                if to_sq == chess.G1:  # Python-chess kingside castling (e1->g1)
                    to_sq = chess.H1  # Convert to C++ convention (e1->h1)
                elif to_sq == chess.C1:  # Python-chess queenside castling (e1->c1)
                    to_sq = chess.A1  # Convert to C++ convention (e1->a1)

            # Black castling (king starts on e8 = square 60)
            elif from_sq == chess.E8:
                if to_sq == chess.G8:  # Python-chess kingside castling (e8->g8)
                    to_sq = chess.H8  # Convert to C++ convention (e8->h8)
                elif to_sq == chess.C8:  # Python-chess queenside castling (e8->c8)
                    to_sq = chess.A8  # Convert to C++ convention (e8->a8)

        key = (from_sq, to_sq, promo)

        cached = cls._to_cpp_cache.get(key)
        if cached is not None:
            return cached

        cpp_move = chess_cpp.Move(from_sq, to_sq, promo)

        if len(cls._to_cpp_cache) < cls._MAX_CACHE_SIZE:
            cls._to_cpp_cache[key] = cpp_move

        return cpp_move

    @classmethod
    def clear_cache(cls):
        cls._to_py_cache.clear()
        cls._to_cpp_cache.clear()


class CachedBoard:
    """
    OPTIMIZED: Chess board wrapper with inlined cache access.

    Key optimization: All _cache property access is now inlined as _cache_stack[-1]
    to eliminate 2.8M+ property call overhead.
    """

    __slots__ = ('_board', '_py_board', '_py_board_dirty', '_cache_stack',
                 '_move_info_stack', '_move_stack', '_use_cpp', '_initial_fen',
                 '_cpp_stack_dirty', '_hash_history')

    # OPTIMIZATION: Class-level pool for _CacheState objects to avoid allocation overhead
    _cache_pool: List[_CacheState] = []
    _POOL_MAX_SIZE = 128  # Limit pool size to avoid unbounded memory growth

    @classmethod
    def _get_pooled_cache(cls) -> _CacheState:
        """Get a _CacheState from pool or create new one."""
        if cls._cache_pool:
            cache = cls._cache_pool.pop()
            # Reset all fields to None (faster than creating new object)
            cache.zobrist_hash = None
            cache.legal_moves = None
            cache.legal_moves_int = None
            cache.has_non_pawn_material = None
            cache.is_check = None
            cache.is_checkmate = None
            cache.is_game_over = None
            cache.material_evaluation = None
            cache.is_endgame = None
            cache.move_is_capture = None
            cache.move_gives_check = None
            cache.move_victim_type = None
            cache.move_attacker_type = None
            cache.move_is_capture_int = None
            cache.move_gives_check_int = None
            cache.move_victim_type_int = None
            cache.move_attacker_type_int = None
            cache.move_is_en_passant_int = None
            cache.move_is_castling_int = None
            cache.move_piece_color_int = None
            cache.move_captured_piece_type_int = None
            cache.move_captured_piece_color_int = None
            cache.move_mvv_lva_int = None
            return cache
        return _CacheState()

    @classmethod
    def _return_to_pool(cls, cache: _CacheState) -> None:
        """Return a _CacheState to the pool for reuse."""
        if len(cls._cache_pool) < cls._POOL_MAX_SIZE:
            cls._cache_pool.append(cache)

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN):
        self._cache_stack: List[_CacheState] = [self._get_pooled_cache()]
        self._move_info_stack: List[_MoveInfo] = []
        self._move_stack: List[chess.Move] = []
        self._use_cpp = HAS_CPP_BACKEND
        self._py_board: Optional[chess.Board] = None
        self._py_board_dirty = True
        self._cpp_stack_dirty = False
        self._hash_history: List[int] = []
        self._initial_fen = fen if fen is not None else "8/8/8/8/8/8/8/8 w - - 0 1"

        if self._use_cpp:
            if fen is None:
                self._board = chess_cpp.Board("8/8/8/8/8/8/8/8 w - - 0 1")
            else:
                self._board = chess_cpp.Board(fen)
            self._hash_history.append(self._board.polyglot_hash())
        else:
            self._board = chess.Board(fen) if fen else chess.Board(None)
            self._py_board = self._board
            self._py_board_dirty = False
            self._hash_history.append(chess.polyglot.zobrist_hash(self._board))

    # Keep property for backward compatibility but inline in hot paths
    @property
    def _cache(self) -> _CacheState:
        return self._cache_stack[-1]

    @property
    def turn(self) -> bool:
        return self._board.turn

    @property
    def fullmove_number(self) -> int:
        return self._board.fullmove_number

    @property
    def occupied(self) -> int:
        return self._board.occupied

    @property
    def castling_rights(self) -> int:
        return int(self._board.castling_rights)

    @property
    def ep_square(self) -> Optional[int]:
        ep = self._board.ep_square
        # C++ backend returns -1 for no EP, python-chess returns None
        if ep is None or (isinstance(ep, int) and ep < 0):
            return None
        return ep

    @property
    def move_stack(self) -> List[chess.Move]:
        return self._move_stack

    @property
    def legal_moves(self):
        return iter(self.get_legal_moves_list())

    def push(self, move: chess.Move) -> None:
        is_null_move = (move.from_square == move.to_square == 0 and move.promotion is None)

        if is_null_move:
            move_info = _MoveInfo(
                move=move,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                parts = self._board.fen().split(' ')
                parts[1] = 'b' if parts[1] == 'w' else 'w'
                parts[3] = '-'
                self._board.set_fen(' '.join(parts))
                self._py_board_dirty = True
                self._cpp_stack_dirty = True
            else:
                self._board.push(move)
        else:
            move_info = _MoveInfo(
                move=move,
                captured_piece=self._get_captured_piece(move),
                was_en_passant=self.is_en_passant(move),
                was_castling=self.is_castling(move),
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                if self._cpp_stack_dirty:
                    temp_board = chess.Board(self._board.fen())
                    temp_board.push(move)
                    self._board.set_fen(temp_board.fen())
                else:
                    # Only apply castling conversion if this is actually a castling move
                    self._board.push(MoveAdapter.from_chess_move(move, is_castling=move_info.was_castling))
                self._py_board_dirty = True
            else:
                self._board.push(move)

        self._move_stack.append(move)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(self._get_pooled_cache())
        # OPTIMIZATION: Defer hash computation until actually needed
        self._hash_history.append(None)

    def push_with_info(self, move: chess.Move, move_int: int,
                       is_en_passant: bool, is_castling: bool,
                       captured_piece_type: Optional[int],
                       captured_piece_color: Optional[bool]) -> None:
        """
        OPTIMIZATION: Push move using pre-computed move info.

        This eliminates redundant calls to:
        - _get_captured_piece() (which calls is_en_passant())
        - is_en_passant()
        - is_castling() (which calls piece_at())

        Args:
            move: The chess.Move to push
            move_int: Integer representation of the move
            is_en_passant: Whether this move is en passant
            is_castling: Whether this move is castling
            captured_piece_type: Type of captured piece (1-6) or None
            captured_piece_color: Color of captured piece (True=WHITE) or None
        """
        is_null_move = (move.from_square == move.to_square == 0 and move.promotion is None)

        if is_null_move:
            move_info = _MoveInfo(
                move=move,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                parts = self._board.fen().split(' ')
                parts[1] = 'b' if parts[1] == 'w' else 'w'
                parts[3] = '-'
                self._board.set_fen(' '.join(parts))
                self._py_board_dirty = True
                self._cpp_stack_dirty = True
            else:
                self._board.push(move)
        else:
            # Build captured_piece from pre-computed info (no function calls!)
            captured_piece = None
            if captured_piece_type is not None and captured_piece_color is not None:
                captured_piece = chess.Piece(captured_piece_type, captured_piece_color)

            move_info = _MoveInfo(
                move=move,
                captured_piece=captured_piece,
                was_en_passant=is_en_passant,
                was_castling=is_castling,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            if self._use_cpp:
                if self._cpp_stack_dirty:
                    temp_board = chess.Board(self._board.fen())
                    temp_board.push(move)
                    self._board.set_fen(temp_board.fen())
                else:
                    # Only apply castling conversion if this is actually a castling move
                    self._board.push(MoveAdapter.from_chess_move(move, is_castling=is_castling))
                self._py_board_dirty = True
            else:
                self._board.push(move)

        self._move_stack.append(move)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(self._get_pooled_cache())
        # OPTIMIZATION: Defer hash computation until actually needed
        # Store None as placeholder - will be computed lazily in is_repetition or zobrist_hash
        self._hash_history.append(None)

    def pop(self) -> chess.Move:
        if not self._move_stack:
            raise IndexError("pop from empty move stack")

        move = self._move_stack[-1]
        is_null_move = (move.from_square == move.to_square == 0 and move.promotion is None)

        if self._hash_history:
            self._hash_history.pop()

        if self._use_cpp:
            if self._cpp_stack_dirty or is_null_move:
                self._move_stack.pop()
                if len(self._cache_stack) > 1:
                    self._return_to_pool(self._cache_stack.pop())
                if self._move_info_stack:
                    self._move_info_stack.pop()

                self._board.set_fen(self._initial_fen)
                for idx, m in enumerate(self._move_stack):
                    m_is_null = (m.from_square == m.to_square == 0 and m.promotion is None)
                    if m_is_null:
                        parts = self._board.fen().split(' ')
                        parts[1] = 'b' if parts[1] == 'w' else 'w'
                        parts[3] = '-'
                        self._board.set_fen(' '.join(parts))
                    else:
                        # Use the stored was_castling info if available
                        was_castling = False
                        if idx < len(self._move_info_stack):
                            was_castling = self._move_info_stack[idx].was_castling
                        self._board.push(MoveAdapter.from_chess_move(m, is_castling=was_castling))

                self._cpp_stack_dirty = any(
                    m.from_square == m.to_square == 0 and m.promotion is None
                    for m in self._move_stack
                )
                self._py_board_dirty = True
                return move
            else:
                self._board.pop()  # Pop from C++ board but don't use its return value
                # Use the move already stored in _move_stack (it's in python-chess format)
                self._py_board_dirty = True
        else:
            move = self._board.pop()

        self._move_stack.pop()
        if len(self._cache_stack) > 1:
            self._return_to_pool(self._cache_stack.pop())
        if self._move_info_stack:
            self._move_info_stack.pop()
        return move

    def _get_captured_piece(self, move: chess.Move) -> Optional[chess.Piece]:
        if self.is_en_passant(move):
            return chess.Piece(chess.PAWN, not self.turn)
        piece_opt = self._board.piece_at(move.to_square)
        if self._use_cpp:
            return chess.Piece(piece_opt.piece_type, piece_opt.color) if piece_opt else None
        return piece_opt

    def copy(self, stack: bool = True) -> "CachedBoard":
        if stack and self._move_stack:
            board = CachedBoard(self._initial_fen)
            for move in self._move_stack:
                board.push(move)
        else:
            board = CachedBoard(self.fen())
        return board

    def set_fen(self, fen: str) -> None:
        self._board.set_fen(fen)
        # OPTIMIZATION: Return old caches to pool before replacing
        for cache in self._cache_stack:
            self._return_to_pool(cache)
        self._cache_stack = [self._get_pooled_cache()]
        self._move_info_stack = []
        self._move_stack = []
        self._initial_fen = fen
        self._hash_history = [self.zobrist_hash()]
        if self._use_cpp:
            self._py_board_dirty = True
            self._cpp_stack_dirty = False

    def fen(self) -> str:
        return self._board.fen()

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        piece_opt = self._board.piece_at(square)
        if self._use_cpp:
            return chess.Piece(piece_opt.piece_type, piece_opt.color) if piece_opt else None
        return piece_opt

    def piece_type_at(self, square: int) -> Optional[int]:
        """
        Get piece type at square without creating chess.Piece object.

        This is faster than piece_at() when you only need the piece type,
        as it avoids chess.Piece object creation overhead.

        Returns:
            Piece type (1-6) or None if square is empty
        """
        piece_opt = self._board.piece_at(square)
        # Use same pattern as piece_at - C++ backend may return falsy non-None
        if self._use_cpp:
            return piece_opt.piece_type if piece_opt else None
        return piece_opt.piece_type if piece_opt else None

    def king(self, color: bool) -> Optional[int]:
        result = self._board.king(color)
        return result if (not self._use_cpp or result >= 0) else None

    def san(self, move: chess.Move) -> str:
        if self._use_cpp:
            is_castling_move = self.is_castling(move)
            return self._board.san(MoveAdapter.from_chess_move(move, is_castling=is_castling_move))
        return self._board.san(move)

    def parse_san(self, san: str) -> chess.Move:
        if self._use_cpp:
            cpp_move = self._board.parse_san(san)
            is_castling_move = self._board.is_castling(cpp_move)
            return MoveAdapter.to_chess_move(cpp_move, is_castling=is_castling_move)
        return self._board.parse_san(san)

    def is_en_passant(self, move: chess.Move) -> bool:
        if self._use_cpp:
            # En passant detection doesn't need castling conversion
            return self._board.is_en_passant(MoveAdapter.from_chess_move(move, is_castling=False))
        return self._board.is_en_passant(move)

    def is_castling(self, move: chess.Move) -> bool:
        """Check if a move is castling.

        For the C++ backend, we detect castling by checking if:
        1. A king is on the from_square
        2. The move goes from e1/e8 to c1/g1 or c8/g8 (python-chess format)

        This avoids the circular dependency with from_chess_move().
        """
        piece = self.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.KING:
            return False

        # Python-chess castling destinations
        from_sq = move.from_square
        to_sq = move.to_square

        # White castling: e1 -> c1 or g1
        if from_sq == chess.E1 and to_sq in (chess.C1, chess.G1):
            return True
        # Black castling: e8 -> c8 or g8
        if from_sq == chess.E8 and to_sq in (chess.C8, chess.G8):
            return True

        return False

    def is_capture(self, move: chess.Move) -> bool:
        if self._use_cpp:
            # Capture detection doesn't need castling conversion
            is_castling_move = self.is_castling(move)
            return self._board.is_capture(MoveAdapter.from_chess_move(move, is_castling=is_castling_move))
        return self._board.is_capture(move)

    def _ensure_current_hash(self) -> int:
        """Ensure current position's hash is computed and stored in history."""
        if self._hash_history and self._hash_history[-1] is None:
            h = self.zobrist_hash()  # This caches in _CacheState
            self._hash_history[-1] = h
            return h
        return self._hash_history[-1] if self._hash_history else self.zobrist_hash()

    def is_repetition(self, count: int = 3) -> bool:
        """OPTIMIZED: Early termination with lazy hash computation."""
        history_len = len(self._hash_history)
        if history_len < count:
            return False

        # Ensure current hash is computed
        current_hash = self._ensure_current_hash()

        match_count = 1  # Current position counts as 1
        # Check backwards through history (more likely to find recent repetitions)
        for i in range(history_len - 2, -1, -1):
            h = self._hash_history[i]
            # Skip None entries (positions we never needed to check)
            if h is not None and h == current_hash:
                match_count += 1
                if match_count >= count:
                    return True
        return False

    def ply(self) -> int:
        return len(self._move_stack)

    def halfmove_clock(self) -> int:
        return self._board.halfmove_clock

    def pieces_mask(self, piece_type: int, color: bool) -> int:
        return self._board.pieces_mask(piece_type, color)

    def can_claim_fifty_moves(self) -> bool:
        return self.halfmove_clock() >= 100

    # ==================== Inlined cache access ====================

    def zobrist_hash(self) -> int:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.zobrist_hash is None:
            if self._use_cpp:
                cache.zobrist_hash = self._board.polyglot_hash()
            else:
                cache.zobrist_hash = chess.polyglot.zobrist_hash(self._board)
        return cache.zobrist_hash

    def get_legal_moves_list(self) -> List[chess.Move]:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.legal_moves is None:
            if self._use_cpp:
                # Check each move with is_castling before converting
                cache.legal_moves = [
                    MoveAdapter.to_chess_move(m, is_castling=self._board.is_castling(m))
                    for m in self._board.legal_moves()
                ]
            else:
                cache.legal_moves = list(self._board.legal_moves)
        return cache.legal_moves

    def get_legal_moves_int(self) -> List[int]:
        """
        OPTIMIZATION: Return legal moves as integers.

        Also computes gives_check during generation to avoid
        redundant int→C++ move conversion in precompute_move_info_int.

        This avoids chess.Move object creation, providing significant speedup:
        - No object allocation (~56 bytes saved per move)
        - Faster hashing (int hash vs tuple creation)
        - Faster comparison (int vs object __eq__)

        Returns:
            List of integer move representations where:
            - bits 0-5: from_square (0-63)
            - bits 6-11: to_square (0-63)
            - bits 12-15: promotion piece type (0 if none)
        """
        cache = self._cache_stack[-1]  # Inlined
        if cache.legal_moves_int is None:
            if self._use_cpp:
                # Compute gives_check while we have C++ moves
                # This avoids redundant int→C++ conversion later
                cpp_moves = self._board.legal_moves()
                result = []
                gives_check_cache = {}
                board = self._board

                for m in cpp_moves:
                    from_sq = m.from_square
                    to_sq = m.to_square
                    promo = m.promotion if m.promotion > 0 else 0

                    # Only check castling for potential king moves
                    if promo == 0 and (from_sq == 4 or from_sq == 60):
                        if board.is_castling(m):
                            if from_sq == 4:  # E1
                                if to_sq == 7:
                                    to_sq = 6  # H1 -> G1
                                elif to_sq == 0:
                                    to_sq = 2  # A1 -> C1
                            else:  # E8
                                if to_sq == 63:
                                    to_sq = 62  # H8 -> G8
                                elif to_sq == 56:
                                    to_sq = 58  # A8 -> C8

                    move_int = from_sq | (to_sq << 6) | (promo << 12)
                    result.append(move_int)

                    # Compute gives_check while we have the C++ move
                    gives_check_cache[move_int] = board.gives_check(m)

                cache.legal_moves_int = result
                # Pre-populate move_gives_check_int to avoid redundant conversion later
                cache.move_gives_check_int = gives_check_cache
            else:
                cache.legal_moves_int = [
                    move_to_int(m) for m in self._board.legal_moves
                ]
        return cache.legal_moves_int

    def _cpp_move_to_int(self, cpp_move) -> int:
        """
        Convert C++ move directly to integer, handling castling conversion.

        The C++ backend uses king-to-rook for castling, but we use
        king-to-destination (python-chess convention) in our integer format.
        """
        from_sq = cpp_move.from_square
        to_sq = cpp_move.to_square
        promo = cpp_move.promotion if cpp_move.promotion > 0 else 0

        # Only check castling for potential king moves (optimization)
        # E1=4, E8=60 are the only squares where castling can originate
        if promo == 0 and (from_sq == 4 or from_sq == 60):
            if self._board.is_castling(cpp_move):
                # White castling (from E1=4)
                if from_sq == 4:
                    if to_sq == 7:  # H1 -> G1 (kingside)
                        to_sq = 6
                    elif to_sq == 0:  # A1 -> C1 (queenside)
                        to_sq = 2
                # Black castling (from E8=60)
                else:
                    if to_sq == 63:  # H8 -> G8 (kingside)
                        to_sq = 62
                    elif to_sq == 56:  # A8 -> C8 (queenside)
                        to_sq = 58

        return from_sq | (to_sq << 6) | (promo << 12)

    def is_castling_int(self, move_int: int) -> bool:
        """
        Check if an integer move represents castling.

        Castling moves in our integer format have specific patterns:
        - White kingside: e1(4)->g1(6)
        - White queenside: e1(4)->c1(2)
        - Black kingside: e8(60)->g8(62)
        - Black queenside: e8(60)->c8(58)
        """
        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF

        if promo != 0:
            return False

        # Check standard castling patterns
        if from_sq == 4 and to_sq in (6, 2):  # White: E1 to G1 or C1
            piece = self.piece_at(4)
            return piece is not None and piece.piece_type == chess.KING

        if from_sq == 60 and to_sq in (62, 58):  # Black: E8 to G8 or C8
            piece = self.piece_at(60)
            return piece is not None and piece.piece_type == chess.KING

        return False

    def precompute_move_info_int(self) -> None:
        """
        OPTIMIZATION: Precompute move info using integer keys.

        This is significantly faster than the object-based version because:
        - No chess.Move object creation or hashing
        - Integer dictionary keys are faster to hash and compare
        - Combines with get_legal_moves_int() for zero-object move generation

        gives_check is computed during get_legal_moves_int() for
        C++ backend. We MUST call get_legal_moves_int() BEFORE checking if
        move_gives_check_int is populated.

        Uses piece_type_at() instead of piece_at() to avoid
        chess.Piece object creation overhead.

        OPTIMIZATION: Also caches en passant, castling, piece colors
        to eliminate piece_at() calls in nn_inference.update_pre_push().
        """
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_is_capture_int is not None:
            return

        cache.move_is_capture_int = {}
        cache.move_victim_type_int = {}
        cache.move_attacker_type_int = {}
        # New caches
        cache.move_is_en_passant_int = {}
        cache.move_is_castling_int = {}
        cache.move_piece_color_int = {}
        cache.move_captured_piece_type_int = {}
        cache.move_captured_piece_color_int = {}
        # OPTIMIZATION: Pre-computed MVV-LVA scores
        cache.move_mvv_lva_int = {}

        # Call get_legal_moves_int FIRST - this populates
        # move_gives_check_int for C++ backend, avoiding redundant conversion
        legal_moves_int = self.get_legal_moves_int()

        # NOW check if gives_check was already computed during move generation
        if cache.move_gives_check_int is None:
            cache.move_gives_check_int = {}
            needs_gives_check = True
        else:
            needs_gives_check = False

        occupied = self.occupied
        ep_square = self.ep_square
        stm = self.turn  # Side to move

        # Use local reference to avoid repeated attribute lookup
        piece_type_at = self.piece_type_at

        # Build piece map for fast color lookup (avoiding piece_at)
        # piece_color_map[sq] = True if white piece, False if black, None if empty
        piece_color_map = {}
        # Handle both C++ backend (method) and python-chess (dict-like)
        if self._use_cpp:
            white_pieces = self._board.occupied_co(chess.WHITE)
        elif hasattr(self._board, 'occupied_co'):
            white_pieces = self._board.occupied_co[chess.WHITE]
        else:
            white_pieces = 0
        for sq in range(64):
            if occupied & (1 << sq):
                piece_color_map[sq] = bool(white_pieces & (1 << sq))

        for move_int in legal_moves_int:
            from_sq = move_int & 0x3F
            to_sq = (move_int >> 6) & 0x3F
            promo = (move_int >> 12) & 0xF

            # Cache piece color (side to move)
            cache.move_piece_color_int[move_int] = stm

            # Check en passant (pawn moving to ep_square diagonally)
            attacker_type = piece_type_at(from_sq)
            is_ep = (ep_square is not None and
                     to_sq == ep_square and
                     attacker_type == chess.PAWN and
                     abs(from_sq - to_sq) in (7, 9))  # Diagonal pawn move
            cache.move_is_en_passant_int[move_int] = is_ep

            # Check capture
            is_cap = bool(occupied & (1 << to_sq)) or is_ep
            cache.move_is_capture_int[move_int] = is_cap

            # Check castling
            is_castling = False
            if attacker_type == chess.KING and promo == 0:
                # White: e1->g1 or e1->c1, Black: e8->g8 or e8->c8
                if from_sq == 4 and to_sq in (6, 2):  # White castling
                    is_castling = True
                elif from_sq == 60 and to_sq in (62, 58):  # Black castling
                    is_castling = True
            cache.move_is_castling_int[move_int] = is_castling

            # Only compute gives_check if not already done
            if needs_gives_check:
                if self._use_cpp:
                    # Convert int to C++ move for gives_check call
                    cpp_move = self._int_to_cpp_move(move_int)
                    cache.move_gives_check_int[move_int] = self._board.gives_check(cpp_move)
                else:
                    py_move = int_to_move(move_int)
                    cache.move_gives_check_int[move_int] = self._board.gives_check(py_move)

            # Get victim type using piece_type_at (no object creation)
            # Also cache captured piece color
            # OPTIMIZATION: Pre-compute MVV-LVA score
            if is_cap:
                if is_ep:
                    cache.move_victim_type_int[move_int] = chess.PAWN
                    cache.move_captured_piece_type_int[move_int] = chess.PAWN
                    cache.move_captured_piece_color_int[move_int] = not stm  # Opposite of moving piece
                    cache.move_mvv_lva_int[move_int] = _MVV_LVA[chess.PAWN][attacker_type]
                else:
                    victim_type = piece_type_at(to_sq)
                    cache.move_victim_type_int[move_int] = victim_type
                    cache.move_captured_piece_type_int[move_int] = victim_type
                    cache.move_captured_piece_color_int[move_int] = piece_color_map.get(to_sq)
                    if victim_type and attacker_type:
                        cache.move_mvv_lva_int[move_int] = _MVV_LVA[victim_type][attacker_type]
                    else:
                        cache.move_mvv_lva_int[move_int] = 0
            else:
                cache.move_victim_type_int[move_int] = None
                cache.move_captured_piece_type_int[move_int] = None
                cache.move_captured_piece_color_int[move_int] = None
                cache.move_mvv_lva_int[move_int] = 0  # Non-captures have score 0

            # Get attacker type using piece_type_at (no object creation)
            cache.move_attacker_type_int[move_int] = attacker_type

    def _int_to_cpp_move(self, move_int: int):
        """
        Convert integer move to C++ move for board operations.
        Handles castling coordinate conversion (python-chess -> C++ convention).

        Note: Only call this when self._use_cpp is True.
        """
        if not self._use_cpp:
            raise RuntimeError("_int_to_cpp_move called without C++ backend")

        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF

        # Convert castling from python-chess convention to C++ convention
        if promo == 0 and (from_sq == 4 or from_sq == 60):
            # Check if this looks like castling
            if from_sq == 4:  # E1
                if to_sq == 6:  # G1 -> H1 (kingside)
                    to_sq = 7
                elif to_sq == 2:  # C1 -> A1 (queenside)
                    to_sq = 0
            elif from_sq == 60:  # E8
                if to_sq == 62:  # G8 -> H8 (kingside)
                    to_sq = 63
                elif to_sq == 58:  # C8 -> A8 (queenside)
                    to_sq = 56

        return chess_cpp.Move(from_sq, to_sq, promo)

    def is_capture_int(self, move_int: int) -> bool:
        """
        OPTIMIZED: Check if move is a capture using integer key.
        Uses fast fallback to avoid triggering expensive precompute.
        """
        cache = self._cache_stack[-1]
        # Try cache first if available
        if cache.move_is_capture_int is not None:
            result = cache.move_is_capture_int.get(move_int)
            if result is not None:
                return result
        # Fast fallback: direct bitboard check (avoids precompute)
        to_sq = (move_int >> 6) & 0x3F
        if self.occupied & _SQUARE_MASKS[to_sq]:
            return True
        # Check en passant
        ep = self.ep_square
        if ep is not None and to_sq == ep:
            from_sq = move_int & 0x3F
            diff = from_sq - to_sq
            return diff == 7 or diff == -7 or diff == 9 or diff == -9
        return False

    def gives_check_int(self, move_int: int) -> bool:
        """Check if move gives check using integer key."""
        cache = self._cache_stack[-1]
        if cache.move_gives_check_int is None:
            self.precompute_move_info_int()
        result = cache.move_gives_check_int.get(move_int)
        if result is not None:
            return result
        # Fallback: compute directly
        if self._use_cpp:
            cpp_move = self._int_to_cpp_move(move_int)
            return self._board.gives_check(cpp_move)
        else:
            return self._board.gives_check(int_to_move(move_int))

    def get_victim_type_int(self, move_int: int) -> Optional[int]:
        """Get victim piece type using integer key."""
        cache = self._cache_stack[-1]
        if cache.move_victim_type_int is None:
            self.precompute_move_info_int()
        return cache.move_victim_type_int.get(move_int)

    def get_attacker_type_int(self, move_int: int) -> Optional[int]:
        """Get attacker piece type using integer key."""
        cache = self._cache_stack[-1]
        if cache.move_attacker_type_int is None:
            self.precompute_move_info_int()
        return cache.move_attacker_type_int.get(move_int)

    def get_mvv_lva_int(self, move_int: int) -> int:
        """
        OPTIMIZATION: Get pre-computed MVV-LVA score for a move.

        Returns the MVV-LVA score (10*victim_value - attacker_value) for captures,
        or 0 for non-captures. This eliminates 3 dict lookups in move_score_q_search_int.
        """
        cache = self._cache_stack[-1]
        if cache.move_mvv_lva_int is None:
            self.precompute_move_info_int()
        return cache.move_mvv_lva_int.get(move_int, 0)

    # ==================== OPTIMIZATION: New getters ====================

    def is_en_passant_int(self, move_int: int) -> bool:
        """
        OPTIMIZATION: Check if move is en passant using integer key.
        Eliminates need for move conversion and piece_at() calls.
        """
        cache = self._cache_stack[-1]
        if cache.move_is_en_passant_int is None:
            self.precompute_move_info_int()
        result = cache.move_is_en_passant_int.get(move_int)
        if result is not None:
            return result
        # Fallback: compute directly
        ep_square = self.ep_square
        if ep_square is None:
            return False
        to_sq = (move_int >> 6) & 0x3F
        if to_sq != ep_square:
            return False
        from_sq = move_int & 0x3F
        piece_type = self.piece_type_at(from_sq)
        return piece_type == chess.PAWN and abs(from_sq - to_sq) in (7, 9)

    def is_castling_int(self, move_int: int) -> bool:
        """
        OPTIMIZATION: Check if move is castling using integer key.
        Uses cached value when available, avoiding piece_at() calls.
        """
        cache = self._cache_stack[-1]
        if cache.move_is_castling_int is None:
            self.precompute_move_info_int()
        result = cache.move_is_castling_int.get(move_int)
        if result is not None:
            return result
        # Fallback: compute directly
        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF
        if promo != 0:
            return False
        # Check standard castling patterns
        if from_sq == 4 and to_sq in (6, 2):  # White: E1 to G1 or C1
            piece_type = self.piece_type_at(4)
            return piece_type == chess.KING
        if from_sq == 60 and to_sq in (62, 58):  # Black: E8 to G8 or C8
            piece_type = self.piece_type_at(60)
            return piece_type == chess.KING
        return False

    def get_move_piece_color_int(self, move_int: int) -> Optional[bool]:
        """
        OPTIMIZATION: Get the color of the moving piece.
        Returns True for WHITE, False for BLACK, None if not cached.
        """
        cache = self._cache_stack[-1]
        if cache.move_piece_color_int is None:
            self.precompute_move_info_int()
        return cache.move_piece_color_int.get(move_int, self.turn)

    def get_captured_piece_info_int(self, move_int: int) -> Tuple[Optional[int], Optional[bool]]:
        """
        OPTIMIZATION: Get captured piece type and color using integer key.
        Returns (piece_type, is_white) or (None, None) if not a capture.
        Eliminates piece_at() calls in nn_inference.update_pre_push().
        """
        cache = self._cache_stack[-1]
        if cache.move_captured_piece_type_int is None:
            self.precompute_move_info_int()
        return (
            cache.move_captured_piece_type_int.get(move_int),
            cache.move_captured_piece_color_int.get(move_int)
        )

    def get_move_info_for_nn_int(self, move_int: int) -> Tuple[int, bool, bool, bool, Optional[int], Optional[bool]]:
        """
        OPTIMIZATION: Get all move info needed for NN updates in one call.

        Returns:
            (attacker_type, attacker_color, is_en_passant, is_castling,
             captured_type, captured_color)

        This eliminates multiple piece_at() calls in nn_inference.update_pre_push().
        """
        cache = self._cache_stack[-1]
        if cache.move_attacker_type_int is None:
            self.precompute_move_info_int()

        return (
            cache.move_attacker_type_int.get(move_int),
            cache.move_piece_color_int.get(move_int, self.turn),
            cache.move_is_en_passant_int.get(move_int, False),
            cache.move_is_castling_int.get(move_int, False),
            cache.move_captured_piece_type_int.get(move_int),
            cache.move_captured_piece_color_int.get(move_int)
        )

    def is_check(self) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.is_check is None:
            cache.is_check = self._board.is_check()
        return cache.is_check

    def is_checkmate(self) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.is_checkmate is None:
            cache.is_checkmate = self._board.is_checkmate()
        return cache.is_checkmate

    def is_game_over(self) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.is_game_over is None:
            cache.is_game_over = self._board.is_game_over()
        return cache.is_game_over

    def has_non_pawn_material(self, color: Optional[bool] = None) -> bool:
        """Inlined cache access"""
        if color is None:
            color = self.turn
        cache = self._cache_stack[-1]  # Inlined
        if cache.has_non_pawn_material is None:
            cache.has_non_pawn_material = {}
        if color not in cache.has_non_pawn_material:
            cache.has_non_pawn_material[color] = any(
                self.pieces_mask(pt, color) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        return cache.has_non_pawn_material[color]

    def precompute_move_info(self) -> None:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_is_capture is not None:
            return

        cache.move_is_capture = {}
        cache.move_gives_check = {}
        cache.move_victim_type = {}
        cache.move_attacker_type = {}

        legal_moves = self.get_legal_moves_list()
        occupied = self.occupied

        for move in legal_moves:
            is_ep = self.is_en_passant(move)
            is_cap = bool(occupied & chess.BB_SQUARES[move.to_square]) or is_ep
            cache.move_is_capture[move] = is_cap

            if self._use_cpp:
                is_castling_move = self.is_castling(move)
                cache.move_gives_check[move] = self._board.gives_check(
                    MoveAdapter.from_chess_move(move, is_castling=is_castling_move))
            else:
                cache.move_gives_check[move] = self._board.gives_check(move)

            if is_cap:
                cache.move_victim_type[move] = chess.PAWN if is_ep else (
                    self.piece_at(move.to_square).piece_type if self.piece_at(move.to_square) else None)
            else:
                cache.move_victim_type[move] = None

            attacker = self.piece_at(move.from_square)
            cache.move_attacker_type[move] = attacker.piece_type if attacker else None

    def is_capture_cached(self, move: chess.Move) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_is_capture is None:
            self.precompute_move_info()
        result = cache.move_is_capture.get(move)
        return result if result is not None else self.is_capture(move)

    def gives_check_cached(self, move: chess.Move) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_gives_check is None:
            self.precompute_move_info()
        result = cache.move_gives_check.get(move)
        if result is not None:
            return result
        if self._use_cpp:
            is_castling_move = self.is_castling(move)
            return self._board.gives_check(MoveAdapter.from_chess_move(move, is_castling=is_castling_move))
        return self._board.gives_check(move)

    def get_victim_type(self, move: chess.Move) -> Optional[int]:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_victim_type is None:
            self.precompute_move_info()
        return cache.move_victim_type.get(move)

    def get_attacker_type(self, move: chess.Move) -> Optional[int]:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.move_attacker_type is None:
            self.precompute_move_info()
        return cache.move_attacker_type.get(move)

    def material_evaluation(self) -> int:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.material_evaluation is None:
            if len(self._cache_stack) > 1 and self._move_info_stack:
                parent_cache = self._cache_stack[-2]
                if parent_cache.material_evaluation is not None:
                    cache.material_evaluation = self._compute_incremental_material(
                        parent_cache.material_evaluation, self._move_info_stack[-1], parent_cache.is_endgame)
            if cache.material_evaluation is None:
                cache.material_evaluation = self._compute_material_evaluation()
        return cache.material_evaluation

    def _is_endgame(self) -> bool:
        """Inlined cache access"""
        cache = self._cache_stack[-1]  # Inlined
        if cache.is_endgame is None:
            popcount = chess_cpp.popcount if self._use_cpp else chess.popcount
            wq = popcount(self.pieces_mask(chess.QUEEN, chess.WHITE))
            bq = popcount(self.pieces_mask(chess.QUEEN, chess.BLACK))

            if wq == 0 and bq == 0:
                cache.is_endgame = True
            else:
                cache.is_endgame = True
                for color in (chess.WHITE, chess.BLACK):
                    if popcount(self.pieces_mask(chess.QUEEN, color)) > 0:
                        minors = popcount(self.pieces_mask(chess.KNIGHT, color) | self.pieces_mask(chess.BISHOP, color))
                        rooks = popcount(self.pieces_mask(chess.ROOK, color))
                        if rooks > 0 or minors > 1:
                            cache.is_endgame = False
                            break
        return cache.is_endgame

    def _compute_material_evaluation(self) -> int:
        our_mat, their_mat = 0, 0
        our_color = self.turn
        is_eg = self._is_endgame()

        for sq in chess.SQUARES:
            piece = self.piece_at(sq)
            if piece is None:
                continue
            val = PIECE_VALUES[piece.piece_type] + _get_pst_value(piece.piece_type, sq, piece.color, is_eg)
            if piece.color == our_color:
                our_mat += val
            else:
                their_mat += val
        return our_mat - their_mat

    def _compute_incremental_material(self, parent_eval: int, move_info: _MoveInfo,
                                      parent_is_endgame: Optional[bool]) -> int:
        move = move_info.move
        is_eg = self._is_endgame()

        if parent_is_endgame is not None and parent_is_endgame != is_eg:
            return self._compute_material_evaluation()

        new_eval = -parent_eval
        piece = self.piece_at(move.to_square)
        if piece is None:
            return self._compute_material_evaluation()

        moving_color = piece.color
        original_type = chess.PAWN if move.promotion else piece.piece_type

        old_pst = _get_pst_value(original_type, move.from_square, moving_color, is_eg)
        new_pst = _get_pst_value(piece.piece_type, move.to_square, moving_color, is_eg)
        new_eval += old_pst - new_pst

        if move.promotion:
            new_eval -= PIECE_VALUES[piece.piece_type] - PIECE_VALUES[chess.PAWN]

        if move_info.captured_piece is not None:
            cap = move_info.captured_piece
            if move_info.was_en_passant:
                ep_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                cap_pst = _get_pst_value(cap.piece_type, ep_sq, not moving_color, is_eg)
            else:
                cap_pst = _get_pst_value(cap.piece_type, move.to_square, not moving_color, is_eg)
            new_eval -= PIECE_VALUES[cap.piece_type] + cap_pst

        return new_eval
