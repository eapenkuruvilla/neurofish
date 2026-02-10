from abc import ABC, abstractmethod
from typing import Tuple
import chess

from cached_board import CachedBoard, move_to_int
from config import MAX_SCORE
from nn_inference import DNNIncrementalUpdater
from nn_inference import NNUEIncrementalUpdater
from nn_inference import load_model


class NNEvaluator(ABC):
    """
    Abstract evaluator for neural network position evaluation.
    Handles both NNUE and DNN with unified interface.

    IMPORTANT: This class does NOT manage board state. The caller (e.g., chess_engine.py)
    is responsible for calling board.push() and board.pop(). This class only
    maintains the incremental evaluation state (accumulators, feature trackers).

    Supports two evaluation modes:
    - Incremental: Efficient for search (push/pop updates accumulators)
    - Full: Standalone evaluation without incremental state

    Recommended usage pattern (unified interface):
        evaluator.push_with_board(board, move)  # Updates both evaluator AND board
        score = evaluator.evaluate(board)
        board.pop()                             # Caller restores board
        evaluator.pop()                         # Restore evaluator state

    Alternative usage pattern (separate updates):
        evaluator.push(board, move)  # Update evaluator state only
        board.push(move)             # Caller updates board separately
        score = evaluator.evaluate(board)
        board.pop()                  # Caller restores board
        evaluator.pop()              # Restore evaluator state
    """

    @abstractmethod
    def push(self, board_before_push: CachedBoard, move: chess.Move):
        """
        Update internal state for a move. Does NOT modify the board.

        Args:
            board_before_push: Board state BEFORE the move
            move: Move being made

        Note: Caller must call board.push(move) separately after this.
        """
        pass

    @abstractmethod
    def pop(self):
        """
        Restore internal state to before the last push. Does NOT modify the board.

        Note: Caller must call board.pop() separately (typically before this).
        """
        pass

    def push_with_board(self, board: CachedBoard, move: chess.Move):
        """
        Update both evaluator state and board for a move.
        This is the preferred method for making moves during search.

        Subclasses may override this for more efficient implementations
        (e.g., NNUE's two-phase update).

        Args:
            board: Board state BEFORE the move (will be modified)
            move: Move being made
            :param move:
            :param board:
        """
        self.push(board, move)
        board.push(move)

    @abstractmethod
    def reset(self, board: CachedBoard):
        """
        Reset incremental state to match a new board position.
        Use when jumping to a position not reachable via push/pop.

        Args:
            board: New board position to sync evaluator state with
        """
        pass

    @abstractmethod
    def _evaluate(self, board: CachedBoard) -> float:
        """
        Evaluate current position using incremental evaluation.
        Requires proper push/pop state management.

        Args:
            board: Current board state (must match internal state)

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        pass

    @abstractmethod
    def _evaluate_full(self, board: CachedBoard) -> float:
        """
        Evaluate position using full matrix multiplication.
        Does not use or affect incremental state.

        Useful for:
        - One-off evaluations
        - Debugging/validation against incremental results
        - Positions not reachable via push/pop from initial position

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        pass

    def evaluate_centipawns(self, board: CachedBoard) -> int:
        """Evaluate using incremental method and convert to centipawns."""
        # Handle game over positions - mate scores are already in centipawn scale
        if board.is_game_over():
            if board.is_checkmate():
                return int(-MAX_SCORE + board.ply())
            return 0
        # Normal positions: convert raw NN output to centipawns
        return int(self._evaluate(board) * 400)

    def evaluate_full_centipawns(self, board: CachedBoard) -> int:
        """Evaluate using full method and convert to centipawns."""
        # Handle game over positions - mate scores are already in centipawn scale
        if board.is_game_over():
            if board.is_checkmate():
                return int(-MAX_SCORE + board.ply())
            return 0
        # Normal positions: convert raw NN output to centipawns
        return int(self._evaluate_full(board) * 400)

    def validate_incremental(self, board: CachedBoard, tolerance: float = 1e-5) -> bool:
        """
        Validate that incremental and full evaluation match.
        Useful for debugging.

        Returns:
            True if evaluations match within tolerance
        """
        inc_eval = self._evaluate(board)
        full_eval = self._evaluate_full(board)
        return abs(inc_eval - full_eval) < tolerance

    @staticmethod
    def create(board: CachedBoard, nn_type: str, model_path: str) -> 'NNEvaluator':
        """Factory method to create appropriate evaluator."""
        if nn_type.upper() == "DNN":
            return DNNEvaluator(board, model_path)
        elif nn_type.upper() == "NNUE":
            return NNUEEvaluator(board, model_path)
        else:
            raise ValueError(f"Unknown NN type: {nn_type}")

    @staticmethod
    def evaluate_position(board: CachedBoard, nn_type: str, model_path: str) -> float:
        """
        Convenience method for one-off position evaluation.
        Does not create incremental state - just evaluates and returns.

        Args:
            board: Position to evaluate
            nn_type: "NNUE" or "DNN"
            model_path: Path to model file

        Returns:
            Raw NN output (approximately in [-1, 1])
        """
        inference = load_model(model_path, nn_type.upper())
        return inference.evaluate_board(board)

    @staticmethod
    def evaluate_position_centipawns(board: CachedBoard, nn_type: str, model_path: str) -> int:
        """
        Convenience method for one-off position evaluation in centipawns.
        """
        return int(NNEvaluator.evaluate_position(board, nn_type, model_path) * 400)


class DNNEvaluator(NNEvaluator):
    """DNN-based evaluator with incremental updates."""

    def __init__(self, board: CachedBoard, model_path: str):
        self.inference = load_model(model_path, "DNN")
        self.updater = DNNIncrementalUpdater(board)

        # Initialize accumulators for both perspectives
        white_feat, black_feat = self.updater.get_features_both()
        self.inference.refresh_accumulator(white_feat, True)
        self.inference.refresh_accumulator(black_feat, False)

    def push(self, board_before_push: CachedBoard, move: chess.Move):
        """Update internal state for a move. Does NOT modify the board."""
        old_white = set(self.updater.white_features)
        old_black = set(self.updater.black_features)

        # Update feature tracker
        self.updater.push(board_before_push, move)

        new_white = set(self.updater.white_features)
        new_black = set(self.updater.black_features)

        # Update both accumulators
        self.inference.update_accumulator(
            new_white - old_white, old_white - new_white, True
        )
        self.inference.update_accumulator(
            new_black - old_black, old_black - new_black, False
        )

    def pop(self):
        """Restore internal state to before the last push. Does NOT modify the board."""
        change_record = self.updater.pop()

        # Reverse the accumulator changes
        self.inference.update_accumulator(
            change_record['white_removed'],
            change_record['white_added'],
            True
        )
        self.inference.update_accumulator(
            change_record['black_removed'],
            change_record['black_added'],
            False
        )

    def _evaluate(self, board: CachedBoard) -> float:
        """Evaluate using incremental accumulators."""
        # if board.is_game_over():
        #    if board.is_checkmate():
        #        return -MAX_SCORE + board.ply()
        #    return 0.0

        perspective = board.turn == chess.WHITE
        return self.inference.evaluate_incremental(perspective)

    def _evaluate_full(self, board: CachedBoard) -> float:
        """Evaluate using full matrix multiplication (no incremental state)."""
        # if board.is_game_over():
        #    if board.is_checkmate():
        #        return -MAX_SCORE + board.ply()
        #    return 0.0

        return self.inference.evaluate_board(board)

    def reset(self, board: CachedBoard):
        """
        Reset incremental state to match a new board position.
        Use when jumping to a position not reachable via push/pop.
        """
        self.updater = DNNIncrementalUpdater(board)
        white_feat, black_feat = self.updater.get_features_both()
        self.inference.refresh_accumulator(white_feat, True)
        self.inference.refresh_accumulator(black_feat, False)


class NNUEEvaluator(NNEvaluator):
    """NNUE-based evaluator with incremental updates."""

    def __init__(self, board: CachedBoard, model_path: str):
        self.inference = load_model(model_path, "NNUE")
        self.updater = NNUEIncrementalUpdater(board)

        # Initialize accumulators
        white_feat, black_feat = self.updater.get_features_unsorted()
        self.inference.refresh_accumulator(white_feat, black_feat)

    def push(self, board_before_push: CachedBoard, move: chess.Move):
        """
        Update internal state for a move. Does NOT modify the board.

        Note: For NNUE, this is less efficient than push_with_board() because
        it requires creating a temporary board copy. Use push_with_board() when possible.

        Uses lazy accumulator refresh for king moves.
        """
        # Get pre-push data
        is_white_king_move, is_black_king_move, change_record = self.updater.update_pre_push(board_before_push, move)

        # Create temporary board to get post-push state
        temp_board = board_before_push.copy()
        temp_board.push(move)

        # Complete the update
        self.updater.update_post_push(temp_board, is_white_king_move, is_black_king_move, change_record)

        # Update accumulators - lazy refresh for king moves
        if is_white_king_move or is_black_king_move:
            self.inference.mark_dirty()
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def push_with_board(self, board: CachedBoard, move: chess.Move):
        """
        Efficiently update both evaluator state and board for a move.
        Uses NNUE's two-phase update for optimal performance.

        Uses lazy accumulator refresh - king moves just mark dirty
        instead of immediately refreshing.

        OPTIMIZATION: Uses cached move info when available to eliminate
        piece_at() calls in update_pre_push.

        OPTIMIZATION: Also uses push_with_info() to eliminate redundant
        is_en_passant(), is_castling(), and _get_captured_piece() calls in board.push().

        Args:
            board: Board state BEFORE the move (will be modified)
            move: Move being made
            :param move:
            :param board:
        """
        # Try to get cached move info to avoid piece_at() calls
        move_int = move_to_int(move)

        # Check if we have cached move info
        cache = board._cache_stack[-1] if board._cache_stack else None
        use_fast_path = (cache is not None and
                         cache.move_attacker_type_int is not None and
                         move_int in cache.move_attacker_type_int)

        if use_fast_path:
            # Fast path: use cached move info
            attacker_type = cache.move_attacker_type_int.get(move_int)
            piece_color = cache.move_piece_color_int.get(move_int, board.turn)
            is_en_passant = cache.move_is_en_passant_int.get(move_int, False)
            is_castling = cache.move_is_castling_int.get(move_int, False)
            captured_type = cache.move_captured_piece_type_int.get(move_int)
            captured_color = cache.move_captured_piece_color_int.get(move_int)

            # Before board.push() - use fast version
            is_white_king_move, is_black_king_move, change_record = self.updater.update_pre_push_fast(move,
                                                                                                      attacker_type,
                                                                                                      piece_color,
                                                                                                      is_en_passant,
                                                                                                      is_castling,
                                                                                                      captured_type,
                                                                                                      captured_color)

            # Push the board using cached info (no redundant lookups!)
            board.push_with_info(move, move_int, is_en_passant, is_castling,
                                 captured_type, captured_color)
        else:
            # Slow path: compute piece info via piece_at() calls
            is_white_king_move, is_black_king_move, change_record = self.updater.update_pre_push(board, move)
            # Slow path: regular push (computes is_en_passant, is_castling, etc.)
            board.push(move)

        # After board.push()
        self.updater.update_post_push(board, is_white_king_move, is_black_king_move, change_record)

        # Update accumulators - lazy refresh for king moves
        if is_white_king_move or is_black_king_move:
            # Mark dirty instead of immediate refresh
            self.inference.mark_dirty()
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def push_with_board_int(self, board: CachedBoard, move: chess.Move, move_int: int):
        """
        OPTIMIZATION: Variant of push_with_board that accepts pre-computed
        move_int to avoid redundant conversion.

        OPTIMIZATION: Also uses push_with_info() to eliminate redundant
        is_en_passant(), is_castling(), and _get_captured_piece() calls.

        Use this when you already have the integer move from move ordering.

        Args:
            board: Board state BEFORE the move (will be modified)
            move: The chess.Move object
            move_int: Pre-computed integer representation of the move
        """
        # Check if we have cached move info
        cache = board._cache_stack[-1] if board._cache_stack else None
        use_fast_path = (cache is not None and
                         cache.move_attacker_type_int is not None and
                         move_int in cache.move_attacker_type_int)

        if use_fast_path:
            # Fast path: use cached move info
            attacker_type = cache.move_attacker_type_int.get(move_int)
            piece_color = cache.move_piece_color_int.get(move_int, board.turn)
            is_en_passant = cache.move_is_en_passant_int.get(move_int, False)
            is_castling = cache.move_is_castling_int.get(move_int, False)
            captured_type = cache.move_captured_piece_type_int.get(move_int)
            captured_color = cache.move_captured_piece_color_int.get(move_int)

            # Before board.push() - use fast version
            is_white_king_move, is_black_king_move, change_record = self.updater.update_pre_push_fast(move,
                                                                                                      attacker_type,
                                                                                                      piece_color,
                                                                                                      is_en_passant,
                                                                                                      is_castling,
                                                                                                      captured_type,
                                                                                                      captured_color)

            # Push the board using cached info (no redundant lookups!)
            board.push_with_info(move, move_int, is_en_passant, is_castling,
                                 captured_type, captured_color)
        else:
            # Slow path: compute piece info via piece_at() calls
            is_white_king_move, is_black_king_move, change_record = self.updater.update_pre_push(board, move)
            # Slow path: regular push
            board.push(move)

        # After board.push()
        self.updater.update_post_push(board, is_white_king_move, is_black_king_move, change_record)

        # Update accumulators - lazy refresh for king moves
        if is_white_king_move or is_black_king_move:
            self.inference.mark_dirty()
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def push_with_board_int_only(self, board: CachedBoard, move_int: int):
        """
        OPTIMIZED: Integer-only push path - no chess.Move object created in fast path.

        This is the fastest push method, avoiding chess.Move object creation when
        cached move info is available (which is the common case in search).

        Args:
            board: Board state BEFORE the move (will be modified)
            move_int: Integer representation of the move (from_sq | to_sq<<6 | promo<<12)
        """
        from_sq = move_int & 0x3F
        to_sq = (move_int >> 6) & 0x3F
        promo = (move_int >> 12) & 0xF

        # Check if we have cached move info
        cache = board._cache_stack[-1] if board._cache_stack else None
        use_fast_path = (cache is not None and
                         cache.move_attacker_type_int is not None and
                         move_int in cache.move_attacker_type_int)

        if use_fast_path:
            # FAST PATH: Use cached move info, no chess.Move created
            attacker_type = cache.move_attacker_type_int.get(move_int)
            piece_color = cache.move_piece_color_int.get(move_int, board.turn)
            is_en_passant = cache.move_is_en_passant_int.get(move_int, False)
            is_castling = cache.move_is_castling_int.get(move_int, False)
            captured_type = cache.move_captured_piece_type_int.get(move_int)
            captured_color = cache.move_captured_piece_color_int.get(move_int)

            # Use integer-only NNUE update
            is_white_king_move, is_black_king_move, change_record = \
                self._update_pre_push_int(
                    from_sq, to_sq, promo,
                    attacker_type, piece_color,
                    is_en_passant, is_castling,
                    captured_type, captured_color)

            # Push the board using integer-only method
            board.push_int(move_int, is_en_passant, is_castling, captured_type, captured_color)
        else:
            # SLOW PATH: Need to compute piece info, create chess.Move
            from cached_board import int_to_move
            move = int_to_move(move_int)
            is_white_king_move, is_black_king_move, change_record = \
                self.updater.update_pre_push(board, move)
            board.push(move)

        # After board.push()
        self.updater.update_post_push(board, is_white_king_move, is_black_king_move, change_record)

        # Update accumulators - lazy refresh for king moves
        if is_white_king_move or is_black_king_move:
            self.inference.mark_dirty()
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def _update_pre_push_int(self, from_sq: int, to_sq: int, promo: int,
                             moving_piece_type: int, moving_piece_color: bool,
                             is_en_passant: bool, is_castling: bool,
                             captured_piece_type, captured_piece_color):
        """
        OPTIMIZED: Integer-only version of NNUE accumulator pre-push update.

        Avoids chess.Move object and uses integer constants directly.
        """
        # Constants (avoid attribute lookups in hot path)
        KING = 6
        PAWN = 1
        ROOK = 4
        WHITE = True

        updater = self.updater

        change_record = {
            'white_added': set(), 'white_removed': set(),
            'black_added': set(), 'black_removed': set(),
            'white_king_moved': False, 'black_king_moved': False,
            'prev_white_king_sq': updater.white_king_sq,
            'prev_black_king_sq': updater.black_king_sq,
            'prev_white_features': None, 'prev_black_features': None,
        }

        if moving_piece_type is None:
            return False, False, change_record

        is_white_king_move = (moving_piece_type == KING and moving_piece_color == WHITE)
        is_black_king_move = (moving_piece_type == KING and moving_piece_color != WHITE)

        # King move handling
        if is_white_king_move:
            change_record['white_king_moved'] = True
            change_record['prev_white_features'] = updater.white_features.copy()
            updater.white_king_sq = to_sq
        if is_black_king_move:
            change_record['black_king_moved'] = True
            change_record['prev_black_features'] = updater.black_features.copy()
            updater.black_king_sq = to_sq

        # Handle captures
        if is_en_passant:
            ep_sq = to_sq + (-8 if moving_piece_color else 8)
            updater._remove_piece_features(ep_sq, PAWN, not moving_piece_color, change_record)
        elif captured_piece_type is not None and captured_piece_color is not None:
            updater._remove_piece_features(to_sq, captured_piece_type, captured_piece_color, change_record)

        # Handle castling rook movement
        rook_to = None
        if is_castling:
            if to_sq > from_sq:  # Kingside
                rook_from = 7 if moving_piece_color else 63  # H1 or H8
                rook_to = 5 if moving_piece_color else 61  # F1 or F8
            else:  # Queenside
                rook_from = 0 if moving_piece_color else 56  # A1 or A8
                rook_to = 3 if moving_piece_color else 59  # D1 or D8
            updater._remove_piece_features(rook_from, ROOK, moving_piece_color, change_record)

        # For non-king moves, handle the moving piece's feature changes
        if not is_white_king_move and not is_black_king_move:
            updater._remove_piece_features(from_sq, moving_piece_type, moving_piece_color, change_record)

        # Handle promotion or regular move to destination
        final_piece_type = promo if promo > 0 else moving_piece_type
        if not is_white_king_move and not is_black_king_move:
            updater._add_piece_features(to_sq, final_piece_type, moving_piece_color, change_record)

        # Add rook features for castling
        if rook_to is not None:
            updater._add_piece_features(rook_to, ROOK, moving_piece_color, change_record)

        return is_white_king_move, is_black_king_move, change_record

    def update_pre_push(self, board_before_push: CachedBoard, move: chess.Move) -> Tuple:
        """
        Returns:
            Tuple of (is_white_king_move, is_black_king_move, change_record)
            Pass these to update_post_push() after calling board.push(move).
        """
        return self.updater.update_pre_push(board_before_push, move)

    def update_post_push(self, board_after_push: CachedBoard,
                         is_white_king_move: bool,
                         is_black_king_move: bool,
                         change_record: dict):
        """
        Uses lazy accumulator refresh for king moves.

        Args:
            board_after_push: Board state after the move was pushed
            is_white_king_move: From update_pre_push return value
            is_black_king_move: From update_pre_push return value
            change_record: From update_pre_push return value
        """
        self.updater.update_post_push(board_after_push, is_white_king_move,
                                      is_black_king_move, change_record)

        # Update accumulators - lazy refresh for king moves
        if is_white_king_move or is_black_king_move:
            self.inference.mark_dirty()
        else:
            self.inference.update_accumulator(
                change_record['white_added'],
                change_record['white_removed'],
                change_record['black_added'],
                change_record['black_removed']
            )

    def pop(self):
        """Restore internal state to before the last push. Does NOT modify the board.

        Uses lazy accumulator handling - king move pops either decrement
        dirty counter (if never evaluated) or refresh (if evaluation happened).
        """
        change_record = self.updater.pop()

        # Restore accumulators - lazy handling for king moves
        if change_record['white_king_moved'] or change_record['black_king_moved']:
            if self.inference.is_dirty():
                # Never evaluated after this king move - just decrement dirty counter
                # Accumulators are still at the pre-push state
                self.inference.unmark_dirty()
            else:
                # We evaluated after this king move (accumulator was refreshed)
                # Need to refresh to the restored features (updater already popped)
                white_feat, black_feat = self.updater.get_features_unsorted()
                self.inference.refresh_accumulator(white_feat, black_feat)
        else:
            # Regular move: reverse incremental updates (only if not dirty)
            # If dirty, the update was skipped on push, so skip the reverse too
            if not self.inference.is_dirty():
                self.inference.update_accumulator(
                    change_record['white_removed'],
                    change_record['white_added'],
                    change_record['black_removed'],
                    change_record['black_added']
                )

    def _evaluate(self, board: CachedBoard) -> float:
        """Evaluate using incremental accumulators.

        Provides feature_getter for lazy refresh when needed.
        """
        stm = board.turn == chess.WHITE
        # Provide feature getter for lazy refresh
        return self.inference.evaluate_incremental(stm, self.updater.get_features_unsorted)

    def _evaluate_full(self, board: CachedBoard) -> float:
        """Evaluate using full matrix multiplication (no incremental state)."""
        # if board.is_game_over():
        #    if board.is_checkmate():
        #        return -MAX_SCORE + board.ply()
        #    return 0.0

        return self.inference.evaluate_board(board)

    def reset(self, board: CachedBoard):
        """
        Reset incremental state to match a new board position.
        Use when jumping to a position not reachable via push/pop.
        """
        self.updater = NNUEIncrementalUpdater(board)
        white_feat, black_feat = self.updater.get_features_unsorted()
        self.inference.refresh_accumulator(white_feat, black_feat)
        self.inference.force_clean()  # Clear any dirty state


# =============================================================================
# Example Usage
# =============================================================================
"""
# Recommended: Use push_with_board() for unified interface (works for both DNN and NNUE):

board = CachedBoard()
evaluator = NNEvaluator.create(board, "NNUE", "model/nnue.pt")

# Evaluate initial position
score = evaluator.evaluate_centipawns(board)
print(f"Initial: {score} cp")

# Make moves using unified interface (updates both evaluator and board)
move = chess.Move.from_uci("e2e4")
evaluator.push_with_board(board, move)  # Updates evaluator AND board
score = evaluator.evaluate_centipawns(board)
print(f"After e4: {score} cp")

# Undo moves - board first, then evaluator
board.pop()
evaluator.pop()
score = evaluator.evaluate_centipawns(board)
print(f"After undo: {score} cp")


# Alternative: Separate push() and board.push() (if you need to do something between):

move = chess.Move.from_uci("d2d4")
evaluator.push(board, move)  # Update evaluator state only
board.push(move)             # Update board state separately
score = evaluator.evaluate_centipawns(board)
print(f"After d4: {score} cp")


# Full evaluation (standalone, no incremental state needed):

board = CachedBoard("r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
score = NNEvaluator.evaluate_position_centipawns(board, "NNUE", "model/nnue.pt")
print(f"Position score: {score} cp")

# Or using an existing evaluator (doesn't affect incremental state):
score = evaluator.evaluate_full_centipawns(board)


# In a search function (chess_engine.py pattern):

class ChessEngine:
    def __init__(self, nn_type: str = "NNUE", model_path: str = "model/nnue.pt"):
        self.nn_type = nn_type
        self.model_path = model_path
        self.evaluator = None

    def search(self, board: CachedBoard, depth: int):
        # Create evaluator for this search
        self.evaluator = NNEvaluator.create(board, self.nn_type, self.model_path)
        return self._negamax(board, depth, -float('inf'), float('inf'))

    def _negamax(self, board: CachedBoard, depth: int, alpha: float, beta: float):
        if depth == 0 or board.is_game_over():
            return self.evaluator.evaluate_centipawns(board), []

        best_score = -float('inf')
        best_pv = []

        for move in board.legal_moves:
            # Unified push - updates both evaluator and board
            self.evaluator.push_with_board(board, move)

            score, pv = self._negamax(board, depth - 1, -beta, -alpha)
            score = -score

            # Restore board, then evaluator state
            board.pop()
            self.evaluator.pop()

            if score > best_score:
                best_score = score
                best_pv = [move] + pv

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_pv
"""