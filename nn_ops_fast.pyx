# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Fast NN Inference Operations - Cython Implementation

This module provides Cython-optimized versions of the hot paths in NN inference:
- evaluate_incremental (DNN and NNUE)
- evaluate_incremental_int8 (NNUE with INT8 quantized L1)
- evaluate_incremental_int16 (NNUE with INT16 quantized L1)
- update_accumulator
- clipped_relu_inplace

Compile with: cythonize -i nn_ops_fast.pyx
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fmax, fmin

# Type definitions
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.int8_t INT8_t
ctypedef np.int16_t INT16_t
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void clipped_relu_inplace(DTYPE_t[:] x) noexcept nogil:
    """In-place clipped ReLU: x = clip(x, 0, 1)"""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]

    for i in range(n):
        if x[i] < 0.0:
            x[i] = 0.0
        elif x[i] > 1.0:
            x[i] = 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void clipped_relu_copy(DTYPE_t[:] src, DTYPE_t[:] dst) noexcept nogil:
    """Copy with clipped ReLU: dst = clip(src, 0, 1)"""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = src.shape[0]

    for i in range(n):
        if src[i] < 0.0:
            dst[i] = 0.0
        elif src[i] > 1.0:
            dst[i] = 1.0
        else:
            dst[i] = src[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float dnn_evaluate_incremental(
    DTYPE_t[:] accumulator,
    DTYPE_t[:, :] l2_weight,
    DTYPE_t[:] l2_bias,
    DTYPE_t[:, :] l3_weight,
    DTYPE_t[:] l3_bias,
    DTYPE_t[:, :] l4_weight,
    DTYPE_t[:] l4_bias,
    DTYPE_t[:] l2_buf,
    DTYPE_t[:] l3_buf,
    DTYPE_t[:] acc_clipped
) noexcept:
    """
    Fast DNN incremental evaluation.

    Performs: accumulator -> clipped_relu -> L2 -> clipped_relu -> L3 -> clipped_relu -> L4

    All intermediate buffers must be pre-allocated.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t acc_size = accumulator.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef Py_ssize_t l3_size = l3_bias.shape[0]
    cdef float sum_val, output

    # Clipped ReLU on accumulator
    with nogil:
        for i in range(acc_size):
            if accumulator[i] < 0.0:
                acc_clipped[i] = 0.0
            elif accumulator[i] > 1.0:
                acc_clipped[i] = 1.0
            else:
                acc_clipped[i] = accumulator[i]

        # L2: acc_clipped @ l2_weight.T + l2_bias
        for i in range(l2_size):
            sum_val = l2_bias[i]
            for j in range(acc_size):
                sum_val = sum_val + acc_clipped[j] * l2_weight[i, j]
            # Clipped ReLU
            if sum_val < 0.0:
                l2_buf[i] = 0.0
            elif sum_val > 1.0:
                l2_buf[i] = 1.0
            else:
                l2_buf[i] = sum_val

        # L3: l2_buf @ l3_weight.T + l3_bias
        for i in range(l3_size):
            sum_val = l3_bias[i]
            for j in range(l2_size):
                sum_val = sum_val + l2_buf[j] * l3_weight[i, j]
            # Clipped ReLU
            if sum_val < 0.0:
                l3_buf[i] = 0.0
            elif sum_val > 1.0:
                l3_buf[i] = 1.0
            else:
                l3_buf[i] = sum_val

        # L4: l3_buf @ l4_weight.T + l4_bias (no activation)
        output = l4_bias[0]
        for j in range(l3_size):
            output = output + l3_buf[j] * l4_weight[0, j]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental(
    DTYPE_t[:] white_accumulator,
    DTYPE_t[:] black_accumulator,
    bint stm,  # True = white to move
    DTYPE_t[:, :] l1_weight,
    DTYPE_t[:] l1_bias,
    DTYPE_t[:, :] l2_weight,
    DTYPE_t[:] l2_bias,
    DTYPE_t[:, :] l3_weight,
    DTYPE_t[:] l3_bias,
    DTYPE_t[:] hidden_buf,
    DTYPE_t[:] l1_buf,
    DTYPE_t[:] l2_buf,
    DTYPE_t[:] white_clipped,
    DTYPE_t[:] black_clipped
) noexcept:
    """
    Fast NNUE incremental evaluation.

    Performs perspective-based concatenation and forward pass through layers.
    """
    cdef Py_ssize_t i, j
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef float sum_val, output

    with nogil:
        # Clipped ReLU on both accumulators
        for i in range(hidden_size):
            if white_accumulator[i] < 0.0:
                white_clipped[i] = 0.0
            elif white_accumulator[i] > 1.0:
                white_clipped[i] = 1.0
            else:
                white_clipped[i] = white_accumulator[i]

            if black_accumulator[i] < 0.0:
                black_clipped[i] = 0.0
            elif black_accumulator[i] > 1.0:
                black_clipped[i] = 1.0
            else:
                black_clipped[i] = black_accumulator[i]

        # Concatenate based on perspective
        if stm:  # White to move
            for i in range(hidden_size):
                hidden_buf[i] = white_clipped[i]
                hidden_buf[hidden_size + i] = black_clipped[i]
        else:  # Black to move
            for i in range(hidden_size):
                hidden_buf[i] = black_clipped[i]
                hidden_buf[hidden_size + i] = white_clipped[i]

        # L1: hidden_buf @ l1_weight.T + l1_bias
        for i in range(l1_size):
            sum_val = l1_bias[i]
            for j in range(concat_size):
                sum_val = sum_val + hidden_buf[j] * l1_weight[i, j]
            if sum_val < 0.0:
                l1_buf[i] = 0.0
            elif sum_val > 1.0:
                l1_buf[i] = 1.0
            else:
                l1_buf[i] = sum_val

        # L2: l1_buf @ l2_weight.T + l2_bias
        for i in range(l2_size):
            sum_val = l2_bias[i]
            for j in range(l1_size):
                sum_val = sum_val + l1_buf[j] * l2_weight[i, j]
            if sum_val < 0.0:
                l2_buf[i] = 0.0
            elif sum_val > 1.0:
                l2_buf[i] = 1.0
            else:
                l2_buf[i] = sum_val

        # L3: l2_buf @ l3_weight.T + l3_bias (no activation)
        output = l3_bias[0]
        for j in range(l2_size):
            output = output + l2_buf[j] * l3_weight[0, j]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental_int8(
    DTYPE_t[:] white_accumulator,
    DTYPE_t[:] black_accumulator,
    bint stm,  # True = white to move
    INT8_t[:, :] l1_weight_q,  # INT8 quantized weights
    DTYPE_t[:] l1_bias,
    float l1_combined_scale,  # Pre-computed scale: input_scale * weight_scale
    DTYPE_t[:, :] l2_weight,
    DTYPE_t[:] l2_bias,
    DTYPE_t[:, :] l3_weight,
    DTYPE_t[:] l3_bias,
    DTYPE_t[:] hidden_buf,    # FP32 buffer for clipped values
    INT8_t[:] hidden_buf_q,   # INT8 buffer for quantized input
    DTYPE_t[:] l1_buf,
    DTYPE_t[:] l2_buf,
    DTYPE_t[:] white_clipped,
    DTYPE_t[:] black_clipped
) noexcept:
    """
    NNUE incremental evaluation with INT8 quantized L1 layer.

    Quantization scheme:
    - Input (hidden_buf) is in [0, 1], quantized to [0, 127] as INT8
    - Weights are pre-quantized to [-127, 127] as INT8
    - Accumulation is done in INT32 to prevent overflow
    - Result is dequantized using pre-computed combined scale

    # TODO: add actual SIMD intrinsics via cython.parallel or direct C calls
    """
    # TODO: store accumulators in quantized form for additional speedup

    cdef Py_ssize_t i, j
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef INT32_t sum_q
    cdef float sum_val, output, val

    with nogil:
        # Clipped ReLU on both accumulators
        for i in range(hidden_size):
            if white_accumulator[i] < 0.0:
                white_clipped[i] = 0.0
            elif white_accumulator[i] > 1.0:
                white_clipped[i] = 1.0
            else:
                white_clipped[i] = white_accumulator[i]

            if black_accumulator[i] < 0.0:
                black_clipped[i] = 0.0
            elif black_accumulator[i] > 1.0:
                black_clipped[i] = 1.0
            else:
                black_clipped[i] = black_accumulator[i]

        # Concatenate based on perspective and quantize to INT8
        if stm:  # White to move
            for i in range(hidden_size):
                hidden_buf[i] = white_clipped[i]
                hidden_buf[hidden_size + i] = black_clipped[i]
        else:  # Black to move
            for i in range(hidden_size):
                hidden_buf[i] = black_clipped[i]
                hidden_buf[hidden_size + i] = white_clipped[i]

        # Quantize input: [0, 1] -> [0, 127]
        for i in range(concat_size):
            val = hidden_buf[i] * 127.0
            if val < 0.0:
                hidden_buf_q[i] = 0
            elif val > 127.0:
                hidden_buf_q[i] = 127
            else:
                hidden_buf_q[i] = <INT8_t>(val + 0.5)  # Round to nearest

        # L1: Quantized matmul with INT32 accumulation
        for i in range(l1_size):
            sum_q = 0
            for j in range(concat_size):
                sum_q = sum_q + <INT32_t>hidden_buf_q[j] * <INT32_t>l1_weight_q[i, j]

            # Dequantize and add bias
            sum_val = <float>sum_q * l1_combined_scale + l1_bias[i]

            # Clipped ReLU
            if sum_val < 0.0:
                l1_buf[i] = 0.0
            elif sum_val > 1.0:
                l1_buf[i] = 1.0
            else:
                l1_buf[i] = sum_val

        # L2: l1_buf @ l2_weight.T + l2_bias (FP32)
        for i in range(l2_size):
            sum_val = l2_bias[i]
            for j in range(l1_size):
                sum_val = sum_val + l1_buf[j] * l2_weight[i, j]
            if sum_val < 0.0:
                l2_buf[i] = 0.0
            elif sum_val > 1.0:
                l2_buf[i] = 1.0
            else:
                l2_buf[i] = sum_val

        # L3: l2_buf @ l3_weight.T + l3_bias (no activation)
        output = l3_bias[0]
        for j in range(l2_size):
            output = output + l2_buf[j] * l3_weight[0, j]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental_int16(
    DTYPE_t[:] white_accumulator,
    DTYPE_t[:] black_accumulator,
    bint stm,  # True = white to move
    INT16_t[:, :] l1_weight_q,  # INT16 quantized weights
    DTYPE_t[:] l1_bias,
    float l1_combined_scale,  # Pre-computed scale: input_scale * weight_scale
    DTYPE_t[:, :] l2_weight,
    DTYPE_t[:] l2_bias,
    DTYPE_t[:, :] l3_weight,
    DTYPE_t[:] l3_bias,
    DTYPE_t[:] hidden_buf,     # FP32 buffer for clipped values
    INT16_t[:] hidden_buf_q,   # INT16 buffer for quantized input
    DTYPE_t[:] l1_buf,
    DTYPE_t[:] l2_buf,
    DTYPE_t[:] white_clipped,
    DTYPE_t[:] black_clipped
) noexcept:
    """
    NNUE incremental evaluation with INT16 quantized L1 layer.

    Quantization scheme:
    - Input (hidden_buf) is in [0, 1], quantized to [0, 32767] as INT16
    - Weights are pre-quantized to [-32767, 32767] as INT16
    - Accumulation is done in INT64 to prevent overflow
      (INT32 would overflow: 512 * 32767 * 32767 > 2^31)
    - Result is dequantized using pre-computed combined scale

    # TODO: add actual SIMD intrinsics via cython.parallel or direct C calls
    """
    # TODO: store accumulators in quantized form for additional speedup

    cdef Py_ssize_t i, j
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    # FIX: Use INT64 for accumulator to prevent overflow with INT16 quantization
    # Worst case: 512 * 32767 * 32767 = ~549 billion, exceeds INT32_MAX (~2.1 billion)
    cdef INT64_t sum_q
    cdef float sum_val, output, val

    with nogil:
        # Clipped ReLU on both accumulators
        for i in range(hidden_size):
            if white_accumulator[i] < 0.0:
                white_clipped[i] = 0.0
            elif white_accumulator[i] > 1.0:
                white_clipped[i] = 1.0
            else:
                white_clipped[i] = white_accumulator[i]

            if black_accumulator[i] < 0.0:
                black_clipped[i] = 0.0
            elif black_accumulator[i] > 1.0:
                black_clipped[i] = 1.0
            else:
                black_clipped[i] = black_accumulator[i]

        # Concatenate based on perspective
        if stm:  # White to move
            for i in range(hidden_size):
                hidden_buf[i] = white_clipped[i]
                hidden_buf[hidden_size + i] = black_clipped[i]
        else:  # Black to move
            for i in range(hidden_size):
                hidden_buf[i] = black_clipped[i]
                hidden_buf[hidden_size + i] = white_clipped[i]

        # Quantize input: [0, 1] -> [0, 32767]
        for i in range(concat_size):
            val = hidden_buf[i] * 32767.0
            if val < 0.0:
                hidden_buf_q[i] = 0
            elif val > 32767.0:
                hidden_buf_q[i] = 32767
            else:
                hidden_buf_q[i] = <INT16_t>(val + 0.5)  # Round to nearest

        # L1: Quantized matmul with INT64 accumulation (prevents overflow)
        for i in range(l1_size):
            sum_q = 0
            for j in range(concat_size):
                # Cast to INT64 before multiplication to prevent overflow
                sum_q = sum_q + <INT64_t>hidden_buf_q[j] * <INT64_t>l1_weight_q[i, j]

            # Dequantize and add bias
            sum_val = <float>sum_q * l1_combined_scale + l1_bias[i]

            # Clipped ReLU
            if sum_val < 0.0:
                l1_buf[i] = 0.0
            elif sum_val > 1.0:
                l1_buf[i] = 1.0
            else:
                l1_buf[i] = sum_val

        # L2: l1_buf @ l2_weight.T + l2_bias (FP32)
        for i in range(l2_size):
            sum_val = l2_bias[i]
            for j in range(l1_size):
                sum_val = sum_val + l1_buf[j] * l2_weight[i, j]
            if sum_val < 0.0:
                l2_buf[i] = 0.0
            elif sum_val > 1.0:
                l2_buf[i] = 1.0
            else:
                l2_buf[i] = sum_val

        # L3: l2_buf @ l3_weight.T + l3_bias (no activation)
        output = l3_bias[0]
        for j in range(l2_size):
            output = output + l2_buf[j] * l3_weight[0, j]

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulator_add_features(
    DTYPE_t[:] accumulator,
    DTYPE_t[:, :] weights,
    INT64_t[:] features,
    Py_ssize_t max_feature
) noexcept:
    """Add weight columns for given features to accumulator."""
    cdef Py_ssize_t i, j, f
    cdef Py_ssize_t n_features = features.shape[0]
    cdef Py_ssize_t acc_size = accumulator.shape[0]

    with nogil:
        for i in range(n_features):
            f = features[i]
            if 0 <= f < max_feature:
                for j in range(acc_size):
                    accumulator[j] = accumulator[j] + weights[j, f]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulator_remove_features(
    DTYPE_t[:] accumulator,
    DTYPE_t[:, :] weights,
    INT64_t[:] features,
    Py_ssize_t max_feature
) noexcept:
    """Remove weight columns for given features from accumulator."""
    cdef Py_ssize_t i, j, f
    cdef Py_ssize_t n_features = features.shape[0]
    cdef Py_ssize_t acc_size = accumulator.shape[0]

    with nogil:
        for i in range(n_features):
            f = features[i]
            if 0 <= f < max_feature:
                for j in range(acc_size):
                    accumulator[j] = accumulator[j] - weights[j, f]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void dnn_update_accumulator(
    DTYPE_t[:] accumulator,
    DTYPE_t[:, :] weights,
    object added_features,  # Set[int]
    object removed_features,  # Set[int]
    Py_ssize_t max_feature
) noexcept:
    """
    Update DNN accumulator with added/removed features.
    Uses batched operations for efficiency.
    """
    cdef Py_ssize_t i, j, f
    cdef Py_ssize_t acc_size = accumulator.shape[0]
    cdef list added_list, removed_list

    # Convert sets to lists for iteration
    added_list = [f for f in added_features if 0 <= f < max_feature]
    removed_list = [f for f in removed_features if 0 <= f < max_feature]

    # Add features
    for f in added_list:
        for j in range(acc_size):
            accumulator[j] = accumulator[j] + weights[j, f]

    # Remove features
    for f in removed_list:
        for j in range(acc_size):
            accumulator[j] = accumulator[j] - weights[j, f]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void nnue_update_accumulator(
    DTYPE_t[:] white_accumulator,
    DTYPE_t[:] black_accumulator,
    DTYPE_t[:, :] weights,
    object added_white,  # Set[int]
    object removed_white,  # Set[int]
    object added_black,  # Set[int]
    object removed_black,  # Set[int]
    Py_ssize_t max_feature
) noexcept:
    """
    Update NNUE accumulators with added/removed features for both perspectives.
    """
    cdef Py_ssize_t j, f
    cdef Py_ssize_t acc_size = white_accumulator.shape[0]
    cdef list aw_list, rw_list, ab_list, rb_list

    # Convert sets to lists
    aw_list = [f for f in added_white if 0 <= f < max_feature]
    rw_list = [f for f in removed_white if 0 <= f < max_feature]
    ab_list = [f for f in added_black if 0 <= f < max_feature]
    rb_list = [f for f in removed_black if 0 <= f < max_feature]

    # White accumulator updates
    for f in aw_list:
        for j in range(acc_size):
            white_accumulator[j] = white_accumulator[j] + weights[j, f]
    for f in rw_list:
        for j in range(acc_size):
            white_accumulator[j] = white_accumulator[j] - weights[j, f]

    # Black accumulator updates
    for f in ab_list:
        for j in range(acc_size):
            black_accumulator[j] = black_accumulator[j] + weights[j, f]
    for f in rb_list:
        for j in range(acc_size):
            black_accumulator[j] = black_accumulator[j] - weights[j, f]


# =============================================================================
# Feature index computation (for NNUE)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int get_piece_index(int piece_type, bint is_friendly) noexcept nogil:
    """
    Convert piece type and color to index (0-9).
    piece_type: 1=PAWN, 2=KNIGHT, 3=BISHOP, 4=ROOK, 5=QUEEN, 6=KING
    """
    if piece_type == 6:  # KING
        return -1
    cdef int type_idx = piece_type - 1
    # Enemy pieces first
    cdef int color_idx = 1 if is_friendly else 0
    return type_idx + color_idx * 5  # 5 piece types


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int get_nnue_feature_index(
    int king_sq,
    int piece_sq,
    int piece_type,
    bint is_friendly
) noexcept nogil:
    """Calculate NNUE feature index."""
    cdef int piece_idx = get_piece_index(piece_type, is_friendly)
    if piece_idx == -1:
        return -1
    # king_sq * (64 * 5 * 2) + piece_sq * (5 * 2) + piece_idx
    return king_sq * 640 + piece_sq * 10 + piece_idx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int flip_square(int square) noexcept nogil:
    """Flip square vertically (A1 <-> A8)."""
    cdef int rank = square // 8
    cdef int file = square % 8
    return (7 - rank) * 8 + file


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int get_dnn_feature_index(
    int square,
    int piece_type,
    bint is_friendly,
    bint perspective
) noexcept nogil:
    """
    Calculate DNN feature index (768-dimensional encoding).

    Encoding: feature_idx = piece_idx * 64 + oriented_square

    Piece order (matches NNUE): P=0, N=1, B=2, R=3, Q=4, K=5
    piece_type from python-chess: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6

    Planes 0-5: Side-to-move pieces
    Planes 6-11: Opponent pieces
    """
    cdef int adj_square = square
    cdef int type_idx, piece_idx

    # Flip square for black's perspective
    if not perspective:
        adj_square = flip_square(square)

    # Map piece type: piece_type - 1 gives P=0, N=1, B=2, R=3, Q=4, K=5
    type_idx = piece_type - 1

    # piece_idx: 0-5 for friendly, 6-11 for opponent
    piece_idx = type_idx + (0 if is_friendly else 6)

    # New encoding: piece_idx * 64 + adj_square
    return piece_idx * 64 + adj_square


# =============================================================================
# Move integer encoding (for fast hashing)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int move_to_int_fast(int from_sq, int to_sq, int promo) noexcept nogil:
    """Convert move to integer key: from_sq | (to_sq << 6) | (promo << 12)"""
    return from_sq | (to_sq << 6) | (promo << 12)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple int_to_move_fast(int key) noexcept:
    """Convert integer key back to (from_sq, to_sq, promo) tuple."""
    cdef int from_sq = key & 0x3F
    cdef int to_sq = (key >> 6) & 0x3F
    cdef int promo = (key >> 12) & 0xF
    return (from_sq, to_sq, promo if promo else None)