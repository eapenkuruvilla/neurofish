/**
 * bindings.cpp - pybind11 Python bindings for chess_wrapper
 *
 * Updated for Python 3.13+ free-threading (no-GIL) support
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "chess_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chess_cpp, m, py::mod_gil_not_used()) {
    m.doc() = "Fast C++ chess library bindings for Python (free-threading safe)";

    // ============== PyMove ==============
    py::class_<chess_wrapper::PyMove>(m, "Move")
        .def(py::init<>())
        .def(py::init<int, int, int>(),
             py::arg("from_square"),
             py::arg("to_square"),
             py::arg("promotion") = 0)
        .def_readwrite("from_square", &chess_wrapper::PyMove::from_square)
        .def_readwrite("to_square", &chess_wrapper::PyMove::to_square)
        .def_readwrite("promotion", &chess_wrapper::PyMove::promotion)
        .def("uci", &chess_wrapper::PyMove::uci)
        .def("__eq__", &chess_wrapper::PyMove::operator==)
        .def("__hash__", [](const chess_wrapper::PyMove& m) {
            return std::hash<int>{}(m.from_square) ^
                   (std::hash<int>{}(m.to_square) << 1) ^
                   (std::hash<int>{}(m.promotion) << 2);
        })
        .def("__repr__", [](const chess_wrapper::PyMove& m) {
            return "Move(" + m.uci() + ")";
        })
        .def("__str__", &chess_wrapper::PyMove::uci);

    // Static factory for null move
    m.def("null_move", []() {
        return chess_wrapper::PyMove(0, 0, 0);
    });

    // ============== PyPiece ==============
    py::class_<chess_wrapper::PyPiece>(m, "Piece")
        .def(py::init<>())
        .def(py::init<int, bool>(),
             py::arg("piece_type"),
             py::arg("color"))
        .def_readwrite("piece_type", &chess_wrapper::PyPiece::piece_type)
        .def_readwrite("color", &chess_wrapper::PyPiece::color)
        .def("__repr__", [](const chess_wrapper::PyPiece& p) {
            const char* names[] = {"", "Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
            std::string color = p.color ? "White" : "Black";
            std::string type = (p.piece_type >= 1 && p.piece_type <= 6) ? names[p.piece_type] : "Unknown";
            return color + " " + type;
        });

    // ============== FastBoard ==============
    py::class_<chess_wrapper::FastBoard>(m, "Board")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("fen"))

        // Copy
        .def("copy", &chess_wrapper::FastBoard::copy)

        // Position setup
        .def("set_fen", &chess_wrapper::FastBoard::set_fen)
        .def("fen", &chess_wrapper::FastBoard::fen)

        // Move making
        .def("push", &chess_wrapper::FastBoard::push)
        .def("push_uci", &chess_wrapper::FastBoard::push_uci)
        .def("pop", &chess_wrapper::FastBoard::pop)

        // Legal moves
        .def("legal_moves", &chess_wrapper::FastBoard::legal_moves)
        .def("legal_moves_count", &chess_wrapper::FastBoard::legal_moves_count)
        .def("is_legal", &chess_wrapper::FastBoard::is_legal)

        // Position queries
        .def_property_readonly("turn", &chess_wrapper::FastBoard::turn)
        .def_property_readonly("fullmove_number", &chess_wrapper::FastBoard::fullmove_number)
        .def_property_readonly("halfmove_clock", &chess_wrapper::FastBoard::halfmove_clock)
        .def("ply", &chess_wrapper::FastBoard::ply)
        .def_property_readonly("castling_rights", &chess_wrapper::FastBoard::castling_rights)
        .def_property_readonly("ep_square", &chess_wrapper::FastBoard::ep_square)
        .def_property_readonly("occupied", &chess_wrapper::FastBoard::occupied)

        // Piece queries
        .def("piece_at", &chess_wrapper::FastBoard::piece_at)
        .def("king", &chess_wrapper::FastBoard::king)
        .def("occupied_co", &chess_wrapper::FastBoard::occupied_co)
        .def("pieces_mask", &chess_wrapper::FastBoard::pieces_mask)

        // Game state
        .def("is_check", &chess_wrapper::FastBoard::is_check)
        .def("is_checkmate", &chess_wrapper::FastBoard::is_checkmate)
        .def("is_stalemate", &chess_wrapper::FastBoard::is_stalemate)
        .def("is_game_over", &chess_wrapper::FastBoard::is_game_over)
        .def("is_insufficient_material", &chess_wrapper::FastBoard::is_insufficient_material)
        .def("can_claim_fifty_moves", &chess_wrapper::FastBoard::can_claim_fifty_moves)
        .def("is_repetition", &chess_wrapper::FastBoard::is_repetition,
             py::arg("count") = 3)

        // Move classification
        .def("is_capture", &chess_wrapper::FastBoard::is_capture)
        .def("is_en_passant", &chess_wrapper::FastBoard::is_en_passant)
        .def("is_castling", &chess_wrapper::FastBoard::is_castling)
        .def("gives_check", &chess_wrapper::FastBoard::gives_check)

        // Hash
        .def("zobrist_hash", &chess_wrapper::FastBoard::zobrist_hash)
        .def("polyglot_hash", &chess_wrapper::FastBoard::polyglot_hash)

        // SAN
        .def("san", &chess_wrapper::FastBoard::san)
        .def("parse_san", &chess_wrapper::FastBoard::parse_san)
        .def("parse_uci", &chess_wrapper::FastBoard::parse_uci)

        .def("__repr__", [](const chess_wrapper::FastBoard& b) {
            return "Board(\"" + b.fen() + "\")";
        });

    // ============== Static utilities ==============
    m.def("popcount", &chess_wrapper::FastBoard::popcount);
    m.def("square_file", &chess_wrapper::FastBoard::square_file);
    m.def("square_rank", &chess_wrapper::FastBoard::square_rank);
    m.def("square_mirror", &chess_wrapper::FastBoard::square_mirror);
    m.def("make_square", &chess_wrapper::FastBoard::make_square,
          py::arg("file"), py::arg("rank"));

    // ============== Constants ==============
    // Piece types
    m.attr("PAWN") = chess_wrapper::PieceType::PAWN;
    m.attr("KNIGHT") = chess_wrapper::PieceType::KNIGHT;
    m.attr("BISHOP") = chess_wrapper::PieceType::BISHOP;
    m.attr("ROOK") = chess_wrapper::PieceType::ROOK;
    m.attr("QUEEN") = chess_wrapper::PieceType::QUEEN;
    m.attr("KING") = chess_wrapper::PieceType::KING;

    // Colors
    m.attr("WHITE") = true;
    m.attr("BLACK") = false;

    // Standard squares
    m.attr("A1") = 0;
    m.attr("B1") = 1;
    m.attr("C1") = 2;
    m.attr("D1") = 3;
    m.attr("E1") = 4;
    m.attr("F1") = 5;
    m.attr("G1") = 6;
    m.attr("H1") = 7;
    m.attr("A8") = 56;
    m.attr("H8") = 63;

    // Bitboard constants for squares (useful for castling rights checks)
    m.attr("BB_A1") = chess_wrapper::BB::A1;
    m.attr("BB_H1") = chess_wrapper::BB::H1;
    m.attr("BB_A8") = chess_wrapper::BB::A8;
    m.attr("BB_H8") = chess_wrapper::BB::H8;

    // Squares array for BB_SQUARES[sq]
    py::list bb_squares;
    for (int i = 0; i < 64; i++) {
        bb_squares.append(1ULL << i);
    }
    m.attr("BB_SQUARES") = bb_squares;

    // SQUARES array
    py::list squares;
    for (int i = 0; i < 64; i++) {
        squares.append(i);
    }
    m.attr("SQUARES") = squares;

    // Starting FEN
    m.attr("STARTING_FEN") = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}