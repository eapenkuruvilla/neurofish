/**
 * chess_wrapper.cpp - Implementation of FastBoard wrapper
 * 
 * Fixed to match Disservin/chess-library actual API
 */

#include "chess_wrapper.hpp"
#include "polyglot_zobrist.hpp"
#include <sstream>
#include <stdexcept>
#include <bit>  // for std::popcount

namespace chess_wrapper {

// ============== PyMove ==============

std::string PyMove::uci() const {
    std::string result;
    result += static_cast<char>('a' + (from_square % 8));
    result += static_cast<char>('1' + (from_square / 8));
    result += static_cast<char>('a' + (to_square % 8));
    result += static_cast<char>('1' + (to_square / 8));
    
    if (promotion > 0) {
        const char promo_chars[] = {'\0', '\0', 'n', 'b', 'r', 'q'};
        if (promotion >= 2 && promotion <= 5) {
            result += promo_chars[promotion];
        }
    }
    return result;
}

// ============== FastBoard ==============

FastBoard::FastBoard() : board_(chess::constants::STARTPOS) {}

FastBoard::FastBoard(const std::string& fen) {
    if (fen.empty() || fen == "startpos") {
        board_ = chess::Board(chess::constants::STARTPOS);
    } else {
        board_ = chess::Board(fen);
    }
}

FastBoard FastBoard::copy() const {
    FastBoard new_board;
    new_board.board_ = board_;
    new_board.move_stack_ = move_stack_;
    return new_board;
}

void FastBoard::set_fen(const std::string& fen) {
    board_.setFen(fen);
    move_stack_.clear();
}

std::string FastBoard::fen() const {
    return board_.getFen();
}

chess::Move FastBoard::to_chess_move(const PyMove& move) {
    chess::Square from = chess::Square(move.from_square);
    chess::Square to = chess::Square(move.to_square);
    
    if (move.promotion > 0) {
        chess::PieceType promo = to_chess_piece_type(move.promotion);
        return chess::Move::make<chess::Move::PROMOTION>(from, to, promo);
    }
    
    return chess::Move::make<chess::Move::NORMAL>(from, to);
}

PyMove FastBoard::from_chess_move(const chess::Move& move) {
    PyMove result;
    result.from_square = move.from().index();
    result.to_square = move.to().index();
    
    if (move.typeOf() == chess::Move::PROMOTION) {
        result.promotion = from_chess_piece_type(move.promotionType());
    } else {
        result.promotion = 0;
    }
    
    return result;
}

chess::PieceType FastBoard::to_chess_piece_type(int pt) {
    switch (pt) {
        case PieceType::PAWN:   return chess::PieceType::PAWN;
        case PieceType::KNIGHT: return chess::PieceType::KNIGHT;
        case PieceType::BISHOP: return chess::PieceType::BISHOP;
        case PieceType::ROOK:   return chess::PieceType::ROOK;
        case PieceType::QUEEN:  return chess::PieceType::QUEEN;
        case PieceType::KING:   return chess::PieceType::KING;
        default: return chess::PieceType::NONE;
    }
}

int FastBoard::from_chess_piece_type(chess::PieceType pt) {
    if (pt == chess::PieceType::PAWN) return PieceType::PAWN;
    if (pt == chess::PieceType::KNIGHT) return PieceType::KNIGHT;
    if (pt == chess::PieceType::BISHOP) return PieceType::BISHOP;
    if (pt == chess::PieceType::ROOK) return PieceType::ROOK;
    if (pt == chess::PieceType::QUEEN) return PieceType::QUEEN;
    if (pt == chess::PieceType::KING) return PieceType::KING;
    return 0;
}

chess::Move FastBoard::find_move(const PyMove& move) const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    
    for (const auto& m : moves) {
        if (m.from().index() == move.from_square &&
            m.to().index() == move.to_square) {
            if (move.promotion > 0) {
                if (m.typeOf() == chess::Move::PROMOTION &&
                    from_chess_piece_type(m.promotionType()) == move.promotion) {
                    return m;
                }
            } else if (m.typeOf() != chess::Move::PROMOTION) {
                return m;
            }
        }
    }
    return chess::Move::NO_MOVE;
}

void FastBoard::push(const PyMove& move) {
    chess::Move m = find_move(move);
    if (m == chess::Move::NO_MOVE) {
        throw std::runtime_error("Illegal move: " + move.uci());
    }
    board_.makeMove(m);
    move_stack_.push_back(m);
}

void FastBoard::push_uci(const std::string& uci) {
    chess::Move move = chess::uci::uciToMove(board_, uci);
    if (move == chess::Move::NO_MOVE) {
        throw std::runtime_error("Invalid UCI move: " + uci);
    }
    board_.makeMove(move);
    move_stack_.push_back(move);
}

PyMove FastBoard::pop() {
    if (move_stack_.empty()) {
        throw std::runtime_error("No moves to pop");
    }
    
    chess::Move last_move = move_stack_.back();
    move_stack_.pop_back();
    board_.unmakeMove(last_move);
    
    return from_chess_move(last_move);
}

std::vector<PyMove> FastBoard::legal_moves() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    
    std::vector<PyMove> result;
    result.reserve(moves.size());
    
    for (const auto& m : moves) {
        result.push_back(from_chess_move(m));
    }
    
    return result;
}

int FastBoard::legal_moves_count() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    return static_cast<int>(moves.size());
}

bool FastBoard::is_legal(const PyMove& move) const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    
    for (const auto& m : moves) {
        if (m.from().index() == move.from_square &&
            m.to().index() == move.to_square) {
            if (move.promotion > 0) {
                if (m.typeOf() == chess::Move::PROMOTION &&
                    from_chess_piece_type(m.promotionType()) == move.promotion) {
                    return true;
                }
            } else if (m.typeOf() != chess::Move::PROMOTION) {
                return true;
            }
        }
    }
    return false;
}

bool FastBoard::turn() const {
    return board_.sideToMove() == chess::Color::WHITE;
}

int FastBoard::fullmove_number() const {
    return board_.fullMoveNumber();
}

int FastBoard::halfmove_clock() const {
    return board_.halfMoveClock();
}

int FastBoard::ply() const {
    // ply = (fullmove - 1) * 2 + (1 if black to move else 0)
    int fm = board_.fullMoveNumber();
    bool white_turn = (board_.sideToMove() == chess::Color::WHITE);
    return (fm - 1) * 2 + (white_turn ? 0 : 1);
}

uint64_t FastBoard::castling_rights() const {
    // Convert to python-chess format (bitfield of rook squares)
    uint64_t rights = 0;
    auto cr = board_.castlingRights();
    
    // Use the nested Side enum from Board::CastlingRights
    using Side = chess::Board::CastlingRights::Side;
    
    if (cr.has(chess::Color::WHITE, Side::KING_SIDE)) {
        rights |= (1ULL << 7);  // H1
    }
    if (cr.has(chess::Color::WHITE, Side::QUEEN_SIDE)) {
        rights |= (1ULL << 0);  // A1
    }
    if (cr.has(chess::Color::BLACK, Side::KING_SIDE)) {
        rights |= (1ULL << 63);  // H8
    }
    if (cr.has(chess::Color::BLACK, Side::QUEEN_SIDE)) {
        rights |= (1ULL << 56);  // A8
    }
    
    return rights;
}

int FastBoard::ep_square() const {
    auto ep = board_.enpassantSq();
    if (ep == chess::Square::underlying::NO_SQ) {
        return -1;
    }
    return ep.index();
}

std::optional<PyPiece> FastBoard::piece_at(int square) const {
    chess::Square sq(square);
    chess::Piece piece = board_.at(sq);
    
    if (piece == chess::Piece::NONE) {
        return std::nullopt;
    }
    
    PyPiece result;
    result.piece_type = from_chess_piece_type(piece.type());
    result.color = (piece.color() == chess::Color::WHITE);
    return result;
}

int FastBoard::king(bool color) const {
    chess::Color c = color ? chess::Color::WHITE : chess::Color::BLACK;
    chess::Bitboard king_bb = board_.pieces(chess::PieceType::KING, c);
    
    if (king_bb.empty()) {
        return -1;  // No king found
    }
    
    // lsb() returns the square index directly as int
    return king_bb.lsb();
}

uint64_t FastBoard::occupied() const {
    return board_.occ().getBits();
}

uint64_t FastBoard::occupied_co(bool color) const {
    chess::Color c = color ? chess::Color::WHITE : chess::Color::BLACK;
    return board_.us(c).getBits();
}

uint64_t FastBoard::pieces_mask(int piece_type, bool color) const {
    chess::PieceType pt = to_chess_piece_type(piece_type);
    chess::Color c = color ? chess::Color::WHITE : chess::Color::BLACK;
    return board_.pieces(pt, c).getBits();
}

bool FastBoard::is_check() const {
    return board_.inCheck();
}

bool FastBoard::is_checkmate() const {
    if (!board_.inCheck()) return false;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    return moves.empty();
}

bool FastBoard::is_stalemate() const {
    if (board_.inCheck()) return false;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    return moves.empty();
}

bool FastBoard::is_game_over() const {
    auto [reason, result] = board_.isGameOver();
    return reason != chess::GameResultReason::NONE;
}

bool FastBoard::is_insufficient_material() const {
    auto [reason, result] = board_.isGameOver();
    return reason == chess::GameResultReason::INSUFFICIENT_MATERIAL;
}

bool FastBoard::can_claim_fifty_moves() const {
    return board_.halfMoveClock() >= 100;
}

bool FastBoard::is_repetition(int count) const {
    return board_.isRepetition(count);
}

bool FastBoard::is_capture(const PyMove& move) const {
    chess::Square to(move.to_square);
    
    // Check if there's a piece on the target square
    if (board_.at(to) != chess::Piece::NONE) {
        return true;
    }
    
    // Check for en passant
    auto ep = board_.enpassantSq();
    if (ep != chess::Square::underlying::NO_SQ && to == ep) {
        chess::Square from(move.from_square);
        chess::Piece moving = board_.at(from);
        if (moving != chess::Piece::NONE && moving.type() == chess::PieceType::PAWN) {
            return true;
        }
    }
    
    return false;
}

bool FastBoard::is_en_passant(const PyMove& move) const {
    chess::Square from(move.from_square);
    chess::Square to(move.to_square);
    
    chess::Piece moving = board_.at(from);
    if (moving == chess::Piece::NONE || moving.type() != chess::PieceType::PAWN) {
        return false;
    }
    
    auto ep = board_.enpassantSq();
    return (ep != chess::Square::underlying::NO_SQ && to == ep);
}

bool FastBoard::is_castling(const PyMove& move) const {
    chess::Square from(move.from_square);
    chess::Piece moving = board_.at(from);
    
    if (moving == chess::Piece::NONE || moving.type() != chess::PieceType::KING) {
        return false;
    }
    
    // Check if king moves more than one square horizontally
    int from_file = move.from_square % 8;
    int to_file = move.to_square % 8;
    return std::abs(to_file - from_file) > 1;
}

bool FastBoard::gives_check(const PyMove& move) const {
    // Make a copy and test the move
    chess::Board test_board = board_;
    
    // Find and make the move
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, test_board);
    
    for (const auto& m : moves) {
        if (m.from().index() == move.from_square &&
            m.to().index() == move.to_square) {
            if (move.promotion > 0) {
                if (m.typeOf() == chess::Move::PROMOTION &&
                    from_chess_piece_type(m.promotionType()) == move.promotion) {
                    test_board.makeMove(m);
                    return test_board.inCheck();
                }
            } else if (m.typeOf() != chess::Move::PROMOTION) {
                test_board.makeMove(m);
                return test_board.inCheck();
            }
        }
    }
    
    return false;
}

std::vector<PyMove> FastBoard::move_stack() const {
    std::vector<PyMove> result;
    result.reserve(move_stack_.size());
    for (const auto& m : move_stack_) {
        result.push_back(from_chess_move(m));
    }
    return result;
}

size_t FastBoard::move_stack_size() const {
    return move_stack_.size();
}

uint64_t FastBoard::zobrist_hash() const {
    return board_.hash();
}

uint64_t FastBoard::polyglot_hash() const {
    uint64_t h = 0;
    
    // Hash all pieces
    for (int sq = 0; sq < 64; sq++) {
        chess::Square square(sq);
        chess::Piece piece = board_.at(square);
        if (piece != chess::Piece::NONE) {
            int pt = from_chess_piece_type(piece.type());
            bool color = (piece.color() == chess::Color::WHITE);
            h ^= polyglot::piece_key(pt, color, sq);
        }
    }
    
    // Hash castling rights
    auto cr = board_.castlingRights();
    using Side = chess::Board::CastlingRights::Side;
    
    if (cr.has(chess::Color::WHITE, Side::KING_SIDE)) {
        h ^= polyglot::RANDOM_ARRAY[polyglot::CASTLING_BASE + 0];
    }
    if (cr.has(chess::Color::WHITE, Side::QUEEN_SIDE)) {
        h ^= polyglot::RANDOM_ARRAY[polyglot::CASTLING_BASE + 1];
    }
    if (cr.has(chess::Color::BLACK, Side::KING_SIDE)) {
        h ^= polyglot::RANDOM_ARRAY[polyglot::CASTLING_BASE + 2];
    }
    if (cr.has(chess::Color::BLACK, Side::QUEEN_SIDE)) {
        h ^= polyglot::RANDOM_ARRAY[polyglot::CASTLING_BASE + 3];
    }
    
    // Hash en passant (only if there's a legal ep capture)
    auto ep = board_.enpassantSq();
    if (ep != chess::Square::underlying::NO_SQ) {
        int ep_file = ep.index() % 8;
        int ep_rank = ep.index() / 8;
        
        // Determine which color can capture and where their pawns would be
        int pawn_rank;
        chess::Color capturing_color;
        if (ep_rank == 5) {  // White can capture (ep target on rank 6)
            pawn_rank = 4;
            capturing_color = chess::Color::WHITE;
        } else {  // ep_rank == 2, Black can capture (ep target on rank 3)
            pawn_rank = 3;
            capturing_color = chess::Color::BLACK;
        }
        
        // Check for adjacent pawns of the capturing color
        bool has_legal_ep = false;
        for (int file_offset : {-1, 1}) {
            int pawn_file = ep_file + file_offset;
            if (pawn_file >= 0 && pawn_file <= 7) {
                int pawn_sq = pawn_rank * 8 + pawn_file;
                chess::Square psq(pawn_sq);
                chess::Piece p = board_.at(psq);
                if (p != chess::Piece::NONE && 
                    p.type() == chess::PieceType::PAWN && 
                    p.color() == capturing_color) {
                    has_legal_ep = true;
                    break;
                }
            }
        }
        
        if (has_legal_ep) {
            h ^= polyglot::RANDOM_ARRAY[polyglot::EP_BASE + ep_file];
        }
    }
    
    // Hash side to move (Polyglot only hashes when white to move)
    if (board_.sideToMove() == chess::Color::WHITE) {
        h ^= polyglot::RANDOM_ARRAY[polyglot::TURN_KEY];
    }
    
    return h;
}

std::string FastBoard::san(const PyMove& move) const {
    // Find the internal move representation
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    
    for (const auto& m : moves) {
        if (m.from().index() == move.from_square &&
            m.to().index() == move.to_square) {
            if (move.promotion > 0) {
                if (m.typeOf() == chess::Move::PROMOTION &&
                    from_chess_piece_type(m.promotionType()) == move.promotion) {
                    return chess::uci::moveToSan(board_, m);
                }
            } else if (m.typeOf() != chess::Move::PROMOTION) {
                return chess::uci::moveToSan(board_, m);
            }
        }
    }
    
    return move.uci();  // Fallback to UCI
}

PyMove FastBoard::parse_san(const std::string& san) const {
    chess::Move m = chess::uci::parseSan(board_, san);
    if (m == chess::Move::NO_MOVE) {
        throw std::runtime_error("Invalid SAN move: " + san);
    }
    return from_chess_move(m);
}

PyMove FastBoard::parse_uci(const std::string& uci) const {
    chess::Move m = chess::uci::uciToMove(board_, uci);
    if (m == chess::Move::NO_MOVE) {
        throw std::runtime_error("Invalid UCI move: " + uci);
    }
    return from_chess_move(m);
}

int FastBoard::popcount(uint64_t bb) {
    return std::popcount(bb);
}

int FastBoard::square_file(int square) {
    return square % 8;
}

int FastBoard::square_rank(int square) {
    return square / 8;
}

int FastBoard::square_mirror(int square) {
    // Flip vertically: rank = 7 - rank
    int rank = square / 8;
    int file = square % 8;
    return (7 - rank) * 8 + file;
}

int FastBoard::make_square(int file, int rank) {
    return rank * 8 + file;
}

} // namespace chess_wrapper
