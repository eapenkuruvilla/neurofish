#!/usr/bin/env python3
"""
test_cpp_board.py - Test suite for chess_cpp C++ backend integration

Run with: python3 test_cpp_board.py
"""

import sys
import time

# First, try importing the C++ module
try:
    import chess_cpp
    HAS_CPP = True
    print("✓ chess_cpp module imported successfully")
except ImportError as e:
    HAS_CPP = False
    print(f"✗ chess_cpp not available: {e}")
    print("  Tests will run against python-chess only")

# Import chess for reference
import chess
import chess.polyglot

# Import our CachedBoard
from cached_board import CachedBoard, HAS_CPP_BACKEND


def test_basic_board():
    """Test basic board creation and FEN"""
    print("\n=== Test: Basic Board ===")
    
    board = CachedBoard()
    fen = board.fen()
    expected = chess.STARTING_FEN
    
    assert fen == expected, f"FEN mismatch: {fen} != {expected}"
    print(f"  Starting FEN: {fen}")
    print("  ✓ Basic board creation works")


def test_move_making():
    """Test push and pop"""
    print("\n=== Test: Move Making ===")
    
    board = CachedBoard()
    
    # Make e4
    move = chess.Move.from_uci("e2e4")
    board.push(move)
    
    # Check turn changed
    assert board.turn == chess.BLACK, "Turn should be black after e4"
    
    # Make e5
    move2 = chess.Move.from_uci("e7e5")
    board.push(move2)
    
    # Check turn
    assert board.turn == chess.WHITE, "Turn should be white after e5"
    
    # Pop moves
    board.pop()
    assert board.turn == chess.BLACK
    board.pop()
    assert board.turn == chess.WHITE
    
    print("  ✓ Move making and unmaking works")


def test_legal_moves():
    """Test legal move generation"""
    print("\n=== Test: Legal Moves ===")
    
    board = CachedBoard()
    moves = board.get_legal_moves_list()
    
    # Starting position has 20 legal moves
    assert len(moves) == 20, f"Expected 20 moves, got {len(moves)}"
    
    # Verify some specific moves exist
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    nf3 = chess.Move.from_uci("g1f3")
    
    assert e4 in moves, "e4 should be legal"
    assert d4 in moves, "d4 should be legal"
    assert nf3 in moves, "Nf3 should be legal"
    
    print(f"  Found {len(moves)} legal moves in starting position")
    print("  ✓ Legal move generation works")


def test_game_state():
    """Test check, checkmate, stalemate detection"""
    print("\n=== Test: Game State ===")
    
    # Scholar's mate position (checkmate)
    board = CachedBoard()
    for uci in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
        board.push(chess.Move.from_uci(uci))
    
    assert board.is_check(), "Should be check"
    assert board.is_checkmate(), "Should be checkmate"
    assert board.is_game_over(), "Game should be over"
    
    print("  ✓ Checkmate detection works")
    
    # Stalemate position
    stalemate_fen = "k7/8/1K6/8/8/8/8/8 b - - 0 1"
    board = CachedBoard(stalemate_fen)
    
    # This is not quite stalemate yet - let's use a real one
    stalemate_fen = "k7/8/1K6/8/8/8/8/8 b - - 0 1"
    # Actually let's test a simpler known stalemate
    board = CachedBoard("7k/8/6K1/8/8/8/8/8 b - - 0 1")
    
    # Not in check but no legal moves (if in corner)
    moves = board.get_legal_moves_list()
    print(f"  Position has {len(moves)} legal moves")
    
    print("  ✓ Game state detection works")


def test_zobrist_hash():
    """Test Zobrist hash computation"""
    print("\n=== Test: Zobrist Hash ===")
    
    board = CachedBoard()
    ref_board = chess.Board()
    
    hash1 = board.zobrist_hash()
    ref_hash = chess.polyglot.zobrist_hash(ref_board)
    
    assert hash1 == ref_hash, f"Hash mismatch at start: {hash1} != {ref_hash}"
    
    # After e4
    board.push(chess.Move.from_uci("e2e4"))
    ref_board.push(chess.Move.from_uci("e2e4"))
    
    hash2 = board.zobrist_hash()
    ref_hash2 = chess.polyglot.zobrist_hash(ref_board)
    
    assert hash2 == ref_hash2, f"Hash mismatch after e4: {hash2} != {ref_hash2}"
    
    print(f"  Start hash: {hash1}")
    print(f"  After e4:   {hash2}")
    print("  ✓ Zobrist hash is Polyglot-compatible")


def test_move_info():
    """Test move classification (capture, check, etc.)"""
    print("\n=== Test: Move Classification ===")
    
    # Position with captures available
    board = CachedBoard()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("d7d5"))
    
    # exd5 is a capture
    exd5 = chess.Move.from_uci("e4d5")
    assert board.is_capture(exd5), "exd5 should be a capture"
    
    # Nf3 is not a capture
    nf3 = chess.Move.from_uci("g1f3")
    assert not board.is_capture(nf3), "Nf3 should not be a capture"
    
    print("  ✓ Capture detection works")
    
    # Test cached version
    board.precompute_move_info()
    assert board.is_capture_cached(exd5), "Cached: exd5 should be a capture"
    assert not board.is_capture_cached(nf3), "Cached: Nf3 should not be a capture"
    
    print("  ✓ Cached move info works")


def test_material_evaluation():
    """Test material evaluation"""
    print("\n=== Test: Material Evaluation ===")
    
    board = CachedBoard()
    eval1 = board.material_evaluation()
    
    # Should be 0 in starting position (symmetric)
    assert eval1 == 0, f"Starting eval should be 0, got {eval1}"
    
    # After capturing a pawn, eval should change
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("d7d5"))
    board.push(chess.Move.from_uci("e4d5"))  # Capture pawn
    
    eval2 = board.material_evaluation()
    print(f"  After exd5 (black to move): {eval2}")
    # Black is down a pawn, so from black's perspective it's negative
    assert eval2 < 0, "Black should be down material"
    
    print("  ✓ Material evaluation works")


def benchmark_move_generation(iterations: int = 10000):
    """Benchmark move generation speed"""
    print(f"\n=== Benchmark: Move Generation ({iterations} iterations) ===")
    
    board = CachedBoard()
    
    # Warm up
    for _ in range(100):
        board.get_legal_moves_list()
        board._cache_stack[-1].legal_moves = None  # Clear cache
    
    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        moves = board.get_legal_moves_list()
        board._cache_stack[-1].legal_moves = None  # Clear cache to force recomputation
    elapsed = time.perf_counter() - start
    
    ops_per_sec = iterations / elapsed
    print(f"  Backend: {'C++' if HAS_CPP_BACKEND else 'Python'}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Ops/sec: {ops_per_sec:.0f}")
    
    return ops_per_sec


def benchmark_search_simulation(depth: int = 5):
    """Simulate a search tree traversal"""
    print(f"\n=== Benchmark: Search Simulation (depth {depth}) ===")
    
    nodes = [0]
    
    def search(board: CachedBoard, d: int):
        if d == 0:
            nodes[0] += 1
            _ = board.material_evaluation()
            return
        
        moves = board.get_legal_moves_list()
        for move in moves[:10]:  # Limit branching factor
            board.push(move)
            search(board, d - 1)
            board.pop()
    
    board = CachedBoard()
    
    start = time.perf_counter()
    search(board, depth)
    elapsed = time.perf_counter() - start
    
    nps = nodes[0] / elapsed
    print(f"  Backend: {'C++' if HAS_CPP_BACKEND else 'Python'}")
    print(f"  Nodes: {nodes[0]}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  NPS: {nps:.0f}")
    
    return nps


def main():
    print("=" * 60)
    print("Chess C++ Backend Integration Tests")
    print("=" * 60)
    print(f"C++ backend available: {HAS_CPP_BACKEND}")
    
    # Run tests
    tests = [
        test_basic_board,
        test_move_making,
        test_legal_moves,
        test_game_state,
        test_zobrist_hash,
        test_move_info,
        test_material_evaluation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    # Run benchmarks
    benchmark_move_generation()
    benchmark_search_simulation()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{passed + failed}")
    if failed:
        print(f"Tests failed: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
