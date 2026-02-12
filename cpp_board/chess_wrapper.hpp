/**
 * chess_wrapper.hpp - C++ Chess Library Wrapper for Python Integration
 * 
 * This wrapper provides a fast C++ backend for chess move generation, legal move
 * checks, and position queries using Disservin's chess-library.
 * 
 * The wrapper is designed to be a drop-in replacement for python-chess's Board class
 * for the performance-critical operations while keeping caching logic in Python.
 *
 * OPTIMIZATION: Integer-based move methods added to reduce Python object creation
 * overhead in multi-threaded scenarios. Move format: from | (to << 6) | (promo << 12)
 */

#ifndef CHESS_WRAPPER_HPP
#define CHESS_WRAPPER_HPP

#include "chess.hpp"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace chess_wrapper {

/**
 * Move representation for Python binding
 */
struct PyMove {
    int from_square;
    int to_square;
    int promotion;  // 0 = none, 2=knight, 3=bishop, 4=rook, 5=queen

    PyMove() : from_square(0), to_square(0), promotion(0) {}
    PyMove(int from, int to, int promo = 0)
        : from_square(from), to_square(to), promotion(promo) {}

    std::string uci() const;
    bool operator==(const PyMove& other) const {
        return from_square == other.from_square &&
               to_square == other.to_square &&
               promotion == other.promotion;
    }
};

/**
 * Piece representation for Python binding
 */
struct PyPiece {
    int piece_type;  // 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
    bool color;      // true=white, false=black

    PyPiece() : piece_type(0), color(true) {}
    PyPiece(int pt, bool c) : piece_type(pt), color(c) {}
};

/**
 * FastBoard - High-performance chess board implementation
 *
 * Wraps chess-library's Board class and provides Python-compatible interface.
 */
class FastBoard {
public:
    // Constructors
    FastBoard();
    explicit FastBoard(const std::string& fen);

    // Copy
    FastBoard copy() const;

    // Position setup
    void set_fen(const std::string& fen);
    std::string fen() const;

    // Move making/unmaking - PyMove versions
    void push(const PyMove& move);
    void push_uci(const std::string& uci);
    PyMove pop();

    // ========== INTEGER-BASED METHODS (optimized for multi-threading) ==========
    // Move format: from_square | (to_square << 6) | (promotion << 12)
    // These avoid Python object creation overhead

    /// Generate legal moves as packed integers
    std::vector<int> legal_moves_int() const;

    /// Push a move given as packed integer
    void push_int(int move_int);

    /// Check if integer move is a capture
    bool is_capture_int(int move_int) const;

    /// Check if integer move is en passant
    bool is_en_passant_int(int move_int) const;

    /// Check if integer move is castling
    bool is_castling_int(int move_int) const;

    /// Check if integer move gives check
    bool gives_check_int(int move_int) const;

    /// Check if integer move is legal
    bool is_legal_int(int move_int) const;

    /// Get piece type at square (0 if empty, 1-6 for piece types)
    int piece_type_at(int square) const;

    /// Get piece color at square (true=white, false=black, undefined if empty)
    bool piece_color_at(int square) const;

    /// Check if square is occupied
    bool is_occupied(int square) const;

    // ========== END INTEGER-BASED METHODS ==========

    // Legal move generation - PyMove versions
    std::vector<PyMove> legal_moves() const;
    int legal_moves_count() const;
    bool is_legal(const PyMove& move) const;

    // Position queries
    bool turn() const;  // true=white, false=black
    int fullmove_number() const;
    int halfmove_clock() const;
    int ply() const;

    // Castling rights (as bitmask of rook squares)
    uint64_t castling_rights() const;

    // En passant square (-1 if none)
    int ep_square() const;

    // Piece queries
    std::optional<PyPiece> piece_at(int square) const;
    int king(bool color) const;  // Returns king square

    // Bitboard queries
    uint64_t occupied() const;
    uint64_t occupied_co(bool color) const;
    uint64_t pieces_mask(int piece_type, bool color) const;

    // Game state
    bool is_check() const;
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_game_over() const;
    bool is_insufficient_material() const;
    bool can_claim_fifty_moves() const;
    bool is_repetition(int count = 3) const;

    // Move classification - PyMove versions
    bool is_capture(const PyMove& move) const;
    bool is_en_passant(const PyMove& move) const;
    bool is_castling(const PyMove& move) const;
    bool gives_check(const PyMove& move) const;

    // Move stack
    std::vector<PyMove> move_stack() const;
    size_t move_stack_size() const;

    // Zobrist hash
    uint64_t zobrist_hash() const;

    // Polyglot-compatible Zobrist hash (matches python-chess)
    uint64_t polyglot_hash() const;

    // SAN parsing/formatting
    std::string san(const PyMove& move) const;
    PyMove parse_san(const std::string& san) const;
    PyMove parse_uci(const std::string& uci) const;

    // Utility
    static int popcount(uint64_t bb);
    static int square_file(int square);
    static int square_rank(int square);
    static int square_mirror(int square);
    static int make_square(int file, int rank);

    // Integer move conversion utilities
    static int move_to_int(const PyMove& move);
    static PyMove int_to_move(int move_int);

private:
    chess::Board board_;
    std::vector<chess::Move> move_stack_;  // Track moves for pop()

    // Conversion helpers
    static chess::Move to_chess_move(const PyMove& move);
    static PyMove from_chess_move(const chess::Move& move);
    static chess::PieceType to_chess_piece_type(int pt);
    static int from_chess_piece_type(chess::PieceType pt);

    // Find the internal chess::Move for a PyMove
    chess::Move find_move(const PyMove& move) const;

    // Find the internal chess::Move for an integer move
    chess::Move find_move_int(int move_int) const;
};

// Constants matching python-chess
namespace PieceType {
    constexpr int PAWN = 1;
    constexpr int KNIGHT = 2;
    constexpr int BISHOP = 3;
    constexpr int ROOK = 4;
    constexpr int QUEEN = 5;
    constexpr int KING = 6;
}

namespace Color {
    constexpr bool WHITE = true;
    constexpr bool BLACK = false;
}

// Standard squares (matching python-chess)
namespace Square {
    constexpr int A1 = 0, B1 = 1, C1 = 2, D1 = 3, E1 = 4, F1 = 5, G1 = 6, H1 = 7;
    constexpr int A8 = 56, B8 = 57, C8 = 58, D8 = 59, E8 = 60, F8 = 61, G8 = 62, H8 = 63;
}

// Bitboards for squares (matching python-chess BB_SQUARES)
namespace BB {
    constexpr uint64_t square(int sq) { return 1ULL << sq; }
    constexpr uint64_t A1 = square(0);
    constexpr uint64_t H1 = square(7);
    constexpr uint64_t A8 = square(56);
    constexpr uint64_t H8 = square(63);
}

} // namespace chess_wrapper

#endif // CHESS_WRAPPER_HPP