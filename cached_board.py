"""
CachedBoard - Efficient chess board wrapper with intelligent caching.

OPTIMIZED VERSION - Pure C++ backend with integer-based methods for multi-threading

Optimizations:
- Pure C++ backend for all chess operations
- Integer-based move representation throughout
- Uses C++ integer methods to avoid Python object creation overhead
- Inlined _cache property access
- Pre-computed lookup tables
- Instance-level cache pool (thread-safe without locks)
"""

import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, NamedTuple

# C++ backend is required
import libs.chess_cpp as chess_cpp

print("✓ Using fast C++ chess backend (chess_cpp) with integer optimizations", file=sys.stderr)

# Import constants from chess_cpp
from libs.chess_cpp import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE, BLACK,
    A1, B1, C1, D1, E1, F1, G1, H1,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQUARES, BB_SQUARES, STARTING_FEN,
)


# =============================================================================
# Piece representation (replaces chess.Piece)
# =============================================================================

class Piece(NamedTuple):
    """Simple piece representation: (piece_type, color)"""
    piece_type: int  # 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
    color: bool  # True=WHITE, False=BLACK


# =============================================================================
# Move utilities (integer-based)
# =============================================================================

def move_to_int(from_sq: int, to_sq: int, promo: int = 0) -> int:
    """Encode move as integer. Also accepts cpp_move or tuple."""
    return from_sq | (to_sq << 6) | (promo << 12)


def move_to_int_from_obj(move) -> int:
    """Convert move object (with from_square, to_square, promotion attrs) to int."""
    promo = move.promotion if move.promotion else 0
    return move.from_square | (move.to_square << 6) | (promo << 12)


def int_to_tuple(key: int) -> Tuple[int, int, int]:
    """Extract (from_sq, to_sq, promo) from integer move."""
    return (key & 0x3F, (key >> 6) & 0x3F, (key >> 12) & 0xF)


def int_to_uci(key: int) -> str:
    """Convert integer move to UCI string (e.g., 'e2e4', 'e7e8q')."""
    from_sq, to_sq, promo = int_to_tuple(key)
    uci = chr(ord('a') + from_sq % 8) + str(from_sq // 8 + 1)
    uci += chr(ord('a') + to_sq % 8) + str(to_sq // 8 + 1)
    if promo:
        uci += "nbrq"[promo - 2]  # 2=N, 3=B, 4=R, 5=Q
    return uci


def uci_to_int(uci: str) -> int:
    """Convert UCI string to integer move."""
    from_sq = (ord(uci[0]) - ord('a')) + (int(uci[1]) - 1) * 8
    to_sq = (ord(uci[2]) - ord('a')) + (int(uci[3]) - 1) * 8
    promo = 0
    if len(uci) == 5:
        promo = "nbrq".index(uci[4].lower()) + 2
    return move_to_int(from_sq, to_sq, promo)


def square_mirror(square: int) -> int:
    """Mirror square vertically (a1 <-> a8)."""
    return square ^ 56


def square_name(square: int) -> str:
    """Get algebraic name of square (e.g., 0 -> 'a1')."""
    return chr(ord('a') + square % 8) + str(square // 8 + 1)


# =============================================================================
# Piece values and tables
# =============================================================================

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000,
}

# Piece-Square Tables for evaluation (from white's perspective)
# Values in centipawns, applied to material score

PST_PAWN = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

PST_ROOK = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MIDDLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PST_KING_END = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]

# Combine into dict for easy lookup
PST = {
    PAWN: PST_PAWN,
    KNIGHT: PST_KNIGHT,
    BISHOP: PST_BISHOP,
    ROOK: PST_ROOK,
    QUEEN: PST_QUEEN,
    KING: PST_KING_MIDDLE,  # Default to middle game
}

PST_KING_ENDGAME = PST_KING_END


def get_pst_value(piece_type: int, square: int, color: bool, is_endgame: bool = False) -> int:
    """Get piece-square table value for a piece."""
    if piece_type == KING:
        table = PST_KING_ENDGAME if is_endgame else PST_KING_MIDDLE
    else:
        table = PST.get(piece_type)
        if table is None:
            return 0
    if color == BLACK:
        square = square_mirror(square)
    return table[square]


# =============================================================================
# Cache state dataclass
# =============================================================================

@dataclass
class _CacheState:
    """Cache state for a single position."""
    zobrist_hash: Optional[int] = None
    legal_moves_int: Optional[List[int]] = None
    has_non_pawn_material: Optional[Dict[bool, bool]] = None
    is_check: Optional[bool] = None
    is_checkmate: Optional[bool] = None
    is_game_over: Optional[bool] = None
    material_evaluation: Optional[int] = None
    is_endgame: Optional[bool] = None
    # Per-move caches (keyed by move_int)
    move_is_capture_int: Optional[Dict[int, bool]] = None
    move_gives_check_int: Optional[Dict[int, bool]] = None
    move_victim_type_int: Optional[Dict[int, Optional[int]]] = None
    move_attacker_type_int: Optional[Dict[int, int]] = None
    move_is_en_passant_int: Optional[Dict[int, bool]] = None
    move_is_castling_int: Optional[Dict[int, bool]] = None
    move_piece_color_int: Optional[Dict[int, bool]] = None
    move_captured_piece_type_int: Optional[Dict[int, Optional[int]]] = None
    move_captured_piece_color_int: Optional[Dict[int, Optional[bool]]] = None
    move_mvv_lva_int: Optional[Dict[int, int]] = None


@dataclass
class _MoveInfo:
    """Information about a move, stored for efficient unmake."""
    move_int: int = 0
    captured_piece_type: Optional[int] = None
    captured_piece_color: Optional[bool] = None
    was_en_passant: bool = False
    was_castling: bool = False
    previous_castling_rights: int = 0
    previous_ep_square: Optional[int] = None


# =============================================================================
# CachedBoard class
# =============================================================================

class CachedBoard:
    """
    OPTIMIZED: Chess board wrapper with pure C++ backend.

    All moves are represented as integers internally.
    Uses instance-level cache pool for thread-safe operation without locks.
    Uses C++ integer methods to minimize Python object creation overhead.
    """

    __slots__ = ('_board', '_cache_stack', '_move_info_stack', '_move_stack',
                 '_initial_fen', '_hash_history', '_cache_pool')

    # Maximum size for instance-level cache pool
    _POOL_MAX_SIZE = 128

    def __init__(self, fen: Optional[str] = STARTING_FEN):
        # Initialize instance-level pool FIRST (before _get_pooled_cache is called)
        self._cache_pool: List[_CacheState] = []
        self._cache_stack: List[_CacheState] = [self._get_pooled_cache()]
        self._move_info_stack: List[_MoveInfo] = []
        self._move_stack: List[int] = []  # Integer moves
        self._initial_fen = fen if fen is not None else "8/8/8/8/8/8/8/8 w - - 0 1"

        if fen is None:
            self._board = chess_cpp.Board("8/8/8/8/8/8/8/8 w - - 0 1")
        else:
            self._board = chess_cpp.Board(fen)
        self._hash_history: List[int] = [self._board.polyglot_hash()]

    def _get_pooled_cache(self) -> _CacheState:
        """Get a _CacheState from instance pool or create new one."""
        if self._cache_pool:
            cache = self._cache_pool.pop()
            # Reset all fields
            cache.zobrist_hash = None
            cache.legal_moves_int = None
            cache.has_non_pawn_material = None
            cache.is_check = None
            cache.is_checkmate = None
            cache.is_game_over = None
            cache.material_evaluation = None
            cache.is_endgame = None
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

    def _return_to_pool(self, cache: _CacheState) -> None:
        """Return a _CacheState to the instance pool for reuse."""
        if len(self._cache_pool) < self._POOL_MAX_SIZE:
            self._cache_pool.append(cache)

    @property
    def _cache(self) -> _CacheState:
        return self._cache_stack[-1]

    # =========================================================================
    # Castling conversion helpers
    # =========================================================================
    # C++ library uses king-to-rook notation (e1h1), we use standard UCI (e1g1)

    def _cpp_int_to_std_int(self, cpp_int: int) -> int:
        """Convert C++ move int (king-to-rook) to standard UCI int (king-to-dest)."""
        from_sq = cpp_int & 0x3F
        to_sq = (cpp_int >> 6) & 0x3F
        promo = (cpp_int >> 12) & 0xF

        # Only convert castling moves (king moving to rook square)
        # MUST check that there's actually a king on the from square!
        if promo == 0:
            if from_sq == E1 and self._board.piece_type_at(E1) == KING:
                if to_sq == H1:
                    return E1 | (G1 << 6)
                elif to_sq == A1:
                    return E1 | (C1 << 6)
            elif from_sq == E8 and self._board.piece_type_at(E8) == KING:
                if to_sq == H8:
                    return E8 | (G8 << 6)
                elif to_sq == A8:
                    return E8 | (C8 << 6)

        return cpp_int

    def _std_int_to_cpp_int(self, std_int: int) -> int:
        """Convert standard UCI int (king-to-dest) to C++ move int (king-to-rook)."""
        from_sq = std_int & 0x3F
        to_sq = (std_int >> 6) & 0x3F
        promo = (std_int >> 12) & 0xF

        # Only convert if it looks like castling (king moving 2 squares)
        if promo == 0:
            if from_sq == E1:
                # Check if there's a king on e1
                if self._board.piece_type_at(E1) == KING:
                    if to_sq == G1:
                        return E1 | (H1 << 6)
                    elif to_sq == C1:
                        return E1 | (A1 << 6)
            elif from_sq == E8:
                if self._board.piece_type_at(E8) == KING:
                    if to_sq == G8:
                        return E8 | (H8 << 6)
                    elif to_sq == C8:
                        return E8 | (A8 << 6)

        return std_int

    # =========================================================================
    # Legacy conversion methods (for compatibility)
    # =========================================================================

    def _cpp_move_to_int(self, cpp_move) -> int:
        """Convert C++ move to integer, handling castling conversion."""
        from_sq = cpp_move.from_square
        to_sq = cpp_move.to_square
        promo = cpp_move.promotion if cpp_move.promotion > 0 else 0

        # Handle castling: C++ uses king-to-rook, we use king-to-destination
        if promo == 0 and (from_sq == E1 or from_sq == E8):
            if self._board.is_castling(cpp_move):
                if from_sq == E1:
                    if to_sq == H1:
                        to_sq = G1
                    elif to_sq == A1:
                        to_sq = C1
                else:  # E8
                    if to_sq == H8:
                        to_sq = G8
                    elif to_sq == A8:
                        to_sq = C8

        return from_sq | (to_sq << 6) | (promo << 12)

    def _int_to_cpp_move(self, move_int: int):
        """Convert integer move to C++ move, handling castling conversion."""
        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF

        # Handle castling: convert king-to-destination to king-to-rook
        # Only convert if there's actually a king on the from-square
        if promo == 0 and (from_sq == E1 or from_sq == E8):
            piece = self._board.piece_at(from_sq)
            if piece is not None and piece.piece_type == KING:
                if from_sq == E1:
                    if to_sq == G1:
                        to_sq = H1
                    elif to_sq == C1:
                        to_sq = A1
                elif from_sq == E8:
                    if to_sq == G8:
                        to_sq = H8
                    elif to_sq == C8:
                        to_sq = A8

        return chess_cpp.Move(from_sq, to_sq, promo)

    # =========================================================================
    # Basic properties
    # =========================================================================

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
        if ep is None or (isinstance(ep, int) and ep < 0):
            return None
        return ep

    @property
    def move_stack(self) -> List[int]:
        """Return move stack as list of integer moves."""
        return self._move_stack

    @property
    def legal_moves(self):
        """Iterator over legal moves (as integers)."""
        return iter(self.get_legal_moves_int())

    # =========================================================================
    # Move making
    # =========================================================================

    def push(self, move_int: int) -> None:
        """Push a move (as integer) onto the board."""
        is_null_move = (move_int == 0)

        if is_null_move:
            move_info = _MoveInfo(
                move_int=move_int,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )
            # Handle null move via FEN manipulation
            parts = self._board.fen().split(' ')
            parts[1] = 'b' if parts[1] == 'w' else 'w'
            parts[3] = '-'
            self._board.set_fen(' '.join(parts))
        else:
            # Get move info before push
            from_sq = move_int & 0x3F
            to_sq = (move_int >> 6) & 0x3F

            # Convert to C++ format for checking en passant and castling
            cpp_int = self._std_int_to_cpp_int(move_int)
            is_ep = self._board.is_en_passant_int(cpp_int)
            is_castling = self._board.is_castling_int(cpp_int)

            # Get captured piece info using C++ integer methods
            captured_type = None
            captured_color = None
            if is_ep:
                captured_type = PAWN
                captured_color = not self.turn
            else:
                # Use C++ integer method for piece lookup
                actual_to_sq = (cpp_int >> 6) & 0x3F  # Use C++ to_sq for capture check
                pt = self._board.piece_type_at(actual_to_sq)
                if pt != 0:
                    captured_type = pt
                    captured_color = self._board.piece_color_at(actual_to_sq)

            move_info = _MoveInfo(
                move_int=move_int,
                captured_piece_type=captured_type,
                captured_piece_color=captured_color,
                was_en_passant=is_ep,
                was_castling=is_castling,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            # Use C++ integer push (faster, no PyMove object creation)
            self._board.push_int(cpp_int)

        self._move_stack.append(move_int)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(self._get_pooled_cache())
        self._hash_history.append(None)  # Lazy hash computation

    def pop(self) -> int:
        """Pop a move from the board, return the integer move."""
        if not self._move_stack:
            raise IndexError("pop from empty move stack")

        move_int = self._move_stack[-1]
        is_null_move = (move_int == 0)

        if self._hash_history:
            self._hash_history.pop()

        if is_null_move:
            # Rebuild board from initial FEN
            self._move_stack.pop()
            if len(self._cache_stack) > 1:
                self._return_to_pool(self._cache_stack.pop())
            if self._move_info_stack:
                self._move_info_stack.pop()

            self._board.set_fen(self._initial_fen)
            for m_int in self._move_stack:
                if m_int == 0:
                    parts = self._board.fen().split(' ')
                    parts[1] = 'b' if parts[1] == 'w' else 'w'
                    parts[3] = '-'
                    self._board.set_fen(' '.join(parts))
                else:
                    cpp_int = self._std_int_to_cpp_int(m_int)
                    self._board.push_int(cpp_int)
            return move_int

        self._board.pop()
        self._move_stack.pop()
        if len(self._cache_stack) > 1:
            self._return_to_pool(self._cache_stack.pop())
        if self._move_info_stack:
            self._move_info_stack.pop()
        return move_int

    def copy(self, stack: bool = True) -> "CachedBoard":
        """Create a copy of the board."""
        if stack and self._move_stack:
            board = CachedBoard(self._initial_fen)
            for move_int in self._move_stack:
                board.push(move_int)
        else:
            board = CachedBoard(self.fen())
        return board

    def set_fen(self, fen: str) -> None:
        """Set position from FEN string."""
        self._board.set_fen(fen)
        for cache in self._cache_stack:
            self._return_to_pool(cache)
        self._cache_stack = [self._get_pooled_cache()]
        self._move_info_stack = []
        self._move_stack = []
        self._initial_fen = fen
        self._hash_history = [self.zobrist_hash()]

    def fen(self) -> str:
        """Get FEN string of current position."""
        return self._board.fen()

    # =========================================================================
    # Piece queries
    # =========================================================================

    def piece_at(self, square: int) -> Optional[Piece]:
        """Get piece at square, or None if empty."""
        piece = self._board.piece_at(square)
        if piece:
            return Piece(piece.piece_type, piece.color)
        return None

    def piece_type_at(self, square: int) -> Optional[int]:
        """Get piece type at square (faster than piece_at if you only need type)."""
        pt = self._board.piece_type_at(square)
        return pt if pt != 0 else None

    def king(self, color: bool) -> Optional[int]:
        """Get square of king for given color."""
        result = self._board.king(color)
        return result if result >= 0 else None

    def pieces_mask(self, piece_type: int, color: bool) -> int:
        """Get bitboard of pieces of given type and color."""
        return self._board.pieces_mask(piece_type, color)

    def occupied_co(self, color: bool) -> int:
        """Get bitboard of all pieces of given color."""
        return self._board.occupied_co(color)

    # =========================================================================
    # Game state queries
    # =========================================================================

    def is_check(self) -> bool:
        """Check if current side is in check."""
        cache = self._cache_stack[-1]
        if cache.is_check is None:
            cache.is_check = self._board.is_check()
        return cache.is_check

    def is_checkmate(self) -> bool:
        """Check if current position is checkmate."""
        cache = self._cache_stack[-1]
        if cache.is_checkmate is None:
            cache.is_checkmate = self._board.is_checkmate()
        return cache.is_checkmate

    def is_stalemate(self) -> bool:
        """Check if current position is stalemate."""
        return self._board.is_stalemate()

    def is_game_over(self) -> bool:
        """Check if game is over."""
        cache = self._cache_stack[-1]
        if cache.is_game_over is None:
            cache.is_game_over = self._board.is_game_over()
        return cache.is_game_over

    def is_insufficient_material(self) -> bool:
        """Check for insufficient material."""
        return self._board.is_insufficient_material()

    def can_claim_fifty_moves(self) -> bool:
        """Check if fifty-move rule can be claimed."""
        return self._board.halfmove_clock >= 100

    def halfmove_clock(self) -> int:
        """Get halfmove clock (for fifty-move rule)."""
        return self._board.halfmove_clock

    def ply(self) -> int:
        """Get number of half-moves played."""
        return len(self._move_stack)

    # =========================================================================
    # Hash and repetition
    # =========================================================================

    def zobrist_hash(self) -> int:
        """Get Zobrist hash of current position."""
        cache = self._cache_stack[-1]
        if cache.zobrist_hash is None:
            cache.zobrist_hash = self._board.polyglot_hash()
        return cache.zobrist_hash

    def _ensure_current_hash(self) -> int:
        """Ensure current position's hash is computed."""
        if self._hash_history and self._hash_history[-1] is None:
            h = self.zobrist_hash()
            self._hash_history[-1] = h
            return h
        return self._hash_history[-1] if self._hash_history else self.zobrist_hash()

    def is_repetition(self, count: int = 3) -> bool:
        """Check for position repetition."""
        history_len = len(self._hash_history)
        if history_len < count:
            return False

        current_hash = self._ensure_current_hash()
        match_count = 1
        for i in range(history_len - 2, -1, -1):
            h = self._hash_history[i]
            if h is not None and h == current_hash:
                match_count += 1
                if match_count >= count:
                    return True
        return False

    # =========================================================================
    # Legal moves (OPTIMIZED: uses C++ integer method)
    # =========================================================================

    def get_legal_moves_int(self) -> List[int]:
        """Get list of legal moves as integers."""
        cache = self._cache_stack[-1]
        if cache.legal_moves_int is None:
            # Use C++ integer method - much faster, no PyMove object creation
            cpp_moves_int = self._board.legal_moves_int()
            result = []
            gives_check_cache = {}

            for cpp_int in cpp_moves_int:
                # Convert from C++ notation to standard UCI notation
                std_int = self._cpp_int_to_std_int(cpp_int)
                result.append(std_int)
                # Pre-compute gives_check using C++ integer method
                gives_check_cache[std_int] = self._board.gives_check_int(cpp_int)

            cache.legal_moves_int = result
            cache.move_gives_check_int = gives_check_cache

        return cache.legal_moves_int

    def is_legal_int(self, move_int: int) -> bool:
        """Check if integer move is legal."""
        legal = self.get_legal_moves_int()
        return move_int in legal

    # =========================================================================
    # Move classification (OPTIMIZED: uses C++ integer methods where possible)
    # =========================================================================

    def is_capture_int(self, move_int: int) -> bool:
        """Check if move is a capture."""
        cache = self._cache_stack[-1]
        if cache.move_is_capture_int is None:
            cache.move_is_capture_int = {}
        if move_int not in cache.move_is_capture_int:
            # Convert to C++ format and use integer method
            cpp_int = self._std_int_to_cpp_int(move_int)
            cache.move_is_capture_int[move_int] = self._board.is_capture_int(cpp_int)
        return cache.move_is_capture_int[move_int]

    def is_en_passant_int(self, move_int: int) -> bool:
        """Check if move is en passant."""
        cache = self._cache_stack[-1]
        if cache.move_is_en_passant_int is None:
            cache.move_is_en_passant_int = {}
        if move_int not in cache.move_is_en_passant_int:
            # Convert to C++ format and use integer method
            cpp_int = self._std_int_to_cpp_int(move_int)
            cache.move_is_en_passant_int[move_int] = self._board.is_en_passant_int(cpp_int)
        return cache.move_is_en_passant_int[move_int]

    def is_castling_int(self, move_int: int) -> bool:
        """Check if move is castling."""
        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF

        if promo != 0:
            return False

        # Standard castling patterns (using standard UCI notation)
        if from_sq == E1 and to_sq in (G1, C1):
            pt = self._board.piece_type_at(E1)
            return pt == KING
        if from_sq == E8 and to_sq in (G8, C8):
            pt = self._board.piece_type_at(E8)
            return pt == KING
        return False

    def gives_check_int(self, move_int: int) -> bool:
        """Check if move gives check."""
        cache = self._cache_stack[-1]
        if cache.move_gives_check_int is None:
            cache.move_gives_check_int = {}
        if move_int not in cache.move_gives_check_int:
            # Convert to C++ format and use integer method
            cpp_int = self._std_int_to_cpp_int(move_int)
            cache.move_gives_check_int[move_int] = self._board.gives_check_int(cpp_int)
        return cache.move_gives_check_int[move_int]

    def san(self, move_int: int) -> str:
        """Get SAN string for move."""
        cpp_move = self._int_to_cpp_move(move_int)
        return self._board.san(cpp_move)

    def parse_san(self, san: str) -> int:
        """Parse SAN string to integer move."""
        cpp_move = self._board.parse_san(san)
        return self._cpp_move_to_int(cpp_move)

    # =========================================================================
    # Material and evaluation helpers
    # =========================================================================

    def has_non_pawn_material(self, color: bool) -> bool:
        """Check if side has non-pawn material."""
        cache = self._cache_stack[-1]
        if cache.has_non_pawn_material is None:
            cache.has_non_pawn_material = {}
        if color not in cache.has_non_pawn_material:
            for pt in (KNIGHT, BISHOP, ROOK, QUEEN):
                if self._board.pieces_mask(pt, color):
                    cache.has_non_pawn_material[color] = True
                    break
            else:
                cache.has_non_pawn_material[color] = False
        return cache.has_non_pawn_material[color]

    def is_endgame(self) -> bool:
        """Check if position is an endgame."""
        cache = self._cache_stack[-1]
        if cache.is_endgame is None:
            # Simple heuristic: endgame if both sides have little material
            w_material = sum(
                bin(self._board.pieces_mask(pt, WHITE)).count('1') * PIECE_VALUES[pt]
                for pt in (KNIGHT, BISHOP, ROOK, QUEEN)
            )
            b_material = sum(
                bin(self._board.pieces_mask(pt, BLACK)).count('1') * PIECE_VALUES[pt]
                for pt in (KNIGHT, BISHOP, ROOK, QUEEN)
            )
            cache.is_endgame = (w_material + b_material) < 2600
        return cache.is_endgame

    def material_count(self, color: bool) -> int:
        """Get material count for a color."""
        total = 0
        for pt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
            count = bin(self._board.pieces_mask(pt, color)).count('1')
            total += count * PIECE_VALUES[pt]
        return total

    def evaluate_material(self) -> int:
        """Evaluate material balance (positive = white advantage)."""
        cache = self._cache_stack[-1]
        if cache.material_evaluation is None:
            cache.material_evaluation = self.material_count(WHITE) - self.material_count(BLACK)
        return cache.material_evaluation

    def _compute_material_evaluation(self) -> int:
        """Compute full material evaluation with PST."""
        our_mat, their_mat = 0, 0
        our_color = self.turn
        is_eg = self.is_endgame()

        for sq in SQUARES:
            piece = self._board.piece_at(sq)
            if piece is None:
                continue
            val = PIECE_VALUES[piece.piece_type] + get_pst_value(piece.piece_type, sq, piece.color, is_eg)
            if piece.color == our_color:
                our_mat += val
            else:
                their_mat += val
        return our_mat - their_mat

    def _compute_incremental_material(self, parent_eval: int, move_info: _MoveInfo,
                                      parent_is_endgame: Optional[bool]) -> int:
        """Compute material evaluation incrementally from parent position."""
        from_sq, to_sq, promo = int_to_tuple(move_info.move_int)
        is_eg = self.is_endgame()

        if parent_is_endgame is not None and parent_is_endgame != is_eg:
            return self._compute_material_evaluation()

        new_eval = -parent_eval
        piece = self._board.piece_at(to_sq)
        if piece is None:
            return self._compute_material_evaluation()

        moving_color = piece.color
        original_type = PAWN if promo else piece.piece_type

        old_pst = get_pst_value(original_type, from_sq, moving_color, is_eg)
        new_pst = get_pst_value(piece.piece_type, to_sq, moving_color, is_eg)
        new_eval += old_pst - new_pst

        if promo:
            new_eval -= PIECE_VALUES[piece.piece_type] - PIECE_VALUES[PAWN]

        if move_info.captured_piece_type is not None:
            cap_type = move_info.captured_piece_type
            if move_info.was_en_passant:
                # EP capture square is on the from rank
                ep_sq = (from_sq // 8) * 8 + (to_sq % 8)
                cap_pst = get_pst_value(cap_type, ep_sq, not moving_color, is_eg)
            else:
                cap_pst = get_pst_value(cap_type, to_sq, not moving_color, is_eg)
            new_eval -= PIECE_VALUES[cap_type] + cap_pst

        return new_eval

    def material_evaluation_full(self) -> int:
        """Get material evaluation with incremental computation when possible."""
        cache = self._cache_stack[-1]
        if cache.material_evaluation is None:
            if len(self._cache_stack) > 1 and self._move_info_stack:
                parent_cache = self._cache_stack[-2]
                if parent_cache.material_evaluation is not None:
                    cache.material_evaluation = self._compute_incremental_material(
                        parent_cache.material_evaluation, self._move_info_stack[-1], parent_cache.is_endgame)
            if cache.material_evaluation is None:
                cache.material_evaluation = self._compute_material_evaluation()
        return cache.material_evaluation

    # =========================================================================
    # Move ordering helpers
    # =========================================================================

    def get_victim_type_int(self, move_int: int) -> Optional[int]:
        """Get piece type of captured piece (for MVV-LVA)."""
        cache = self._cache_stack[-1]
        if cache.move_victim_type_int is None:
            cache.move_victim_type_int = {}
        if move_int not in cache.move_victim_type_int:
            if not self.is_capture_int(move_int):
                cache.move_victim_type_int[move_int] = None
            elif self.is_en_passant_int(move_int):
                cache.move_victim_type_int[move_int] = PAWN
            else:
                to_sq = (move_int >> 6) & 0x3F
                pt = self._board.piece_type_at(to_sq)
                cache.move_victim_type_int[move_int] = pt if pt != 0 else None
        return cache.move_victim_type_int[move_int]

    def get_attacker_type_int(self, move_int: int) -> int:
        """Get piece type of attacking piece."""
        cache = self._cache_stack[-1]
        if cache.move_attacker_type_int is None:
            cache.move_attacker_type_int = {}
        if move_int not in cache.move_attacker_type_int:
            from_sq = move_int & 0x3F
            pt = self._board.piece_type_at(from_sq)
            cache.move_attacker_type_int[move_int] = pt if pt != 0 else PAWN
        return cache.move_attacker_type_int[move_int]

    def get_mvv_lva_score_int(self, move_int: int) -> int:
        """Get MVV-LVA score for move ordering."""
        cache = self._cache_stack[-1]
        if cache.move_mvv_lva_int is None:
            cache.move_mvv_lva_int = {}
        if move_int not in cache.move_mvv_lva_int:
            victim = self.get_victim_type_int(move_int)
            if victim is None:
                cache.move_mvv_lva_int[move_int] = 0
            else:
                attacker = self.get_attacker_type_int(move_int)
                # Higher score for capturing valuable pieces with cheap pieces
                cache.move_mvv_lva_int[move_int] = PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker]
        return cache.move_mvv_lva_int[move_int]

    def get_captured_piece_type_int(self, move_int: int) -> Optional[int]:
        """Get piece type of captured piece."""
        return self.get_victim_type_int(move_int)

    def get_captured_piece_color_int(self, move_int: int) -> Optional[bool]:
        """Get color of captured piece."""
        cache = self._cache_stack[-1]
        if cache.move_captured_piece_color_int is None:
            cache.move_captured_piece_color_int = {}
        if move_int not in cache.move_captured_piece_color_int:
            if not self.is_capture_int(move_int):
                cache.move_captured_piece_color_int[move_int] = None
            else:
                # Captured piece is opposite color of side to move
                cache.move_captured_piece_color_int[move_int] = not self.turn
        return cache.move_captured_piece_color_int[move_int]

    def get_piece_color_int(self, move_int: int) -> bool:
        """Get color of moving piece."""
        cache = self._cache_stack[-1]
        if cache.move_piece_color_int is None:
            cache.move_piece_color_int = {}
        if move_int not in cache.move_piece_color_int:
            cache.move_piece_color_int[move_int] = self.turn
        return cache.move_piece_color_int[move_int]

    # =========================================================================
    # Batch precomputation for move ordering
    # =========================================================================

    def precompute_move_info_int(self, moves_int: List[int]) -> None:
        """Precompute move info for a batch of moves (for efficient move ordering)."""
        cache = self._cache_stack[-1]

        # Initialize ALL caches that nn_evaluator's fast path needs
        if cache.move_is_capture_int is None:
            cache.move_is_capture_int = {}
        if cache.move_victim_type_int is None:
            cache.move_victim_type_int = {}
        if cache.move_attacker_type_int is None:
            cache.move_attacker_type_int = {}
        if cache.move_mvv_lva_int is None:
            cache.move_mvv_lva_int = {}
        if cache.move_is_en_passant_int is None:
            cache.move_is_en_passant_int = {}
        if cache.move_is_castling_int is None:
            cache.move_is_castling_int = {}
        if cache.move_piece_color_int is None:
            cache.move_piece_color_int = {}
        if cache.move_captured_piece_type_int is None:
            cache.move_captured_piece_type_int = {}
        if cache.move_captured_piece_color_int is None:
            cache.move_captured_piece_color_int = {}

        ep_square = self.ep_square
        turn = self.turn

        for move_int in moves_int:
            if move_int in cache.move_mvv_lva_int:
                continue

            from_sq = move_int & 0x3F
            to_sq = (move_int >> 6) & 0x3F

            # Use C++ integer methods for piece lookup
            attacker_type = self._board.piece_type_at(from_sq)
            if attacker_type == 0:
                attacker_type = PAWN

            target_type = self._board.piece_type_at(to_sq)

            # Check for en passant
            is_ep = (attacker_type == PAWN and to_sq == ep_square and ep_square is not None)

            # Check for castling (king moving 2 squares)
            is_castling = False
            if attacker_type == KING:
                if from_sq == E1 and to_sq in (G1, C1):
                    is_castling = True
                elif from_sq == E8 and to_sq in (G8, C8):
                    is_castling = True

            if target_type != 0:
                is_capture = True
                victim_type = target_type
                captured_color = self._board.piece_color_at(to_sq)
            elif is_ep:
                is_capture = True
                victim_type = PAWN
                captured_color = not turn
            else:
                is_capture = False
                victim_type = None
                captured_color = None

            # Populate ALL caches needed by nn_evaluator fast path
            cache.move_is_capture_int[move_int] = is_capture
            cache.move_is_en_passant_int[move_int] = is_ep
            cache.move_is_castling_int[move_int] = is_castling
            cache.move_piece_color_int[move_int] = turn
            cache.move_victim_type_int[move_int] = victim_type
            cache.move_attacker_type_int[move_int] = attacker_type
            cache.move_captured_piece_type_int[move_int] = victim_type  # Same as victim for captures
            cache.move_captured_piece_color_int[move_int] = captured_color

            if victim_type is None:
                cache.move_mvv_lva_int[move_int] = 0
            else:
                cache.move_mvv_lva_int[move_int] = PIECE_VALUES[victim_type] * 10 - PIECE_VALUES[attacker_type]

    def get_move_info_for_nn_int(self, move_int: int) -> Tuple[int, bool, bool, bool, Optional[int], Optional[bool]]:
        """
        Get all move info needed for NN updates in one call.

        Returns:
            (attacker_type, attacker_color, is_en_passant, is_castling,
             captured_type, captured_color)
        """
        cache = self._cache_stack[-1]
        if cache.move_attacker_type_int is None:
            self.precompute_move_info_int([move_int])

        return (
            cache.move_attacker_type_int.get(move_int),
            cache.move_piece_color_int.get(move_int, self.turn),
            cache.move_is_en_passant_int.get(move_int, False),
            cache.move_is_castling_int.get(move_int, False),
            cache.move_captured_piece_type_int.get(move_int),
            cache.move_captured_piece_color_int.get(move_int)
        )

    def get_captured_piece_info_int(self, move_int: int) -> Tuple[Optional[int], Optional[bool]]:
        """Get (piece_type, color) of captured piece, or (None, None) if not capture."""
        cache = self._cache_stack[-1]
        if cache.move_captured_piece_type_int is None:
            self.precompute_move_info_int([move_int])
        return (
            cache.move_captured_piece_type_int.get(move_int),
            cache.move_captured_piece_color_int.get(move_int)
        )

    def push_with_info_int(self, move_int: int, is_en_passant: bool, is_castling: bool,
                           captured_piece_type: Optional[int], captured_piece_color: Optional[bool]) -> None:
        """
        Push move using pre-computed move info (avoids redundant lookups).
        Used by nn_evaluator fast path.
        """
        is_null_move = (move_int == 0)

        if is_null_move:
            move_info = _MoveInfo(
                move_int=move_int,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )
            parts = self._board.fen().split(' ')
            parts[1] = 'b' if parts[1] == 'w' else 'w'
            parts[3] = '-'
            self._board.set_fen(' '.join(parts))
        else:
            move_info = _MoveInfo(
                move_int=move_int,
                captured_piece_type=captured_piece_type,
                captured_piece_color=captured_piece_color,
                was_en_passant=is_en_passant,
                was_castling=is_castling,
                previous_castling_rights=self.castling_rights,
                previous_ep_square=self.ep_square,
            )

            # Use _int_to_cpp_move which handles castling conversion safely
            cpp_move = self._int_to_cpp_move(move_int)
            self._board.push(cpp_move)

        self._move_stack.append(move_int)
        self._move_info_stack.append(move_info)
        self._cache_stack.append(self._get_pooled_cache())
        self._hash_history.append(None)


# =============================================================================
# Module-level conversion helpers (for UCI interface)
# =============================================================================

def move_to_int_from_chess_move(move) -> int:
    """
    Convert a chess.Move to integer format.
    Only use this at UCI boundary!
    """
    promo = move.promotion if move.promotion else 0
    return move.from_square | (move.to_square << 6) | (promo << 12)


def int_to_chess_move(move_int: int):
    """
    Convert integer move to python-chess Move object.
    Only use this at UCI boundary!
    """
    import chess
    from_sq, to_sq, promo = int_to_tuple(move_int)
    return chess.Move(from_sq, to_sq, promo if promo else None)


def chess_move_to_int(move) -> int:
    """
    Convert python-chess Move to integer.
    Only use this at UCI boundary!
    """
    promo = move.promotion if move.promotion else 0
    return move.from_square | (move.to_square << 6) | (promo << 12)