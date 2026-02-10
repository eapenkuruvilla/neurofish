# cython: freethreading_compatible=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
"""
Optimized NN Inference Operations - Cython Implementation

Optimizations applied:
1. Contiguous memory declarations (::1) for better cache usage
2. Fused operations (bias + clipped_relu in one pass)
3. Local variable caching to avoid repeated indexing
4. Compiler hints for auto-vectorization
5. Minimized Python object interactions in hot paths
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy

# Type definitions
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.int8_t INT8_t
ctypedef np.int16_t INT16_t
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t


# =============================================================================
# Core element-wise operations
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void clipped_relu_inplace(DTYPE_t[::1] x) noexcept nogil:
    """In-place clipped ReLU: x = clip(x, 0, 1)"""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    cdef DTYPE_t val

    for i in range(n):
        val = x[i]
        if val < 0.0:
            x[i] = 0.0
        elif val > 1.0:
            x[i] = 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void clipped_relu_copy(DTYPE_t[::1] src, DTYPE_t[::1] dst) noexcept nogil:
    """Copy with clipped ReLU: dst = clip(src, 0, 1)"""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = src.shape[0]
    cdef DTYPE_t val

    for i in range(n):
        val = src[i]
        if val < 0.0:
            dst[i] = 0.0
        elif val > 1.0:
            dst[i] = 1.0
        else:
            dst[i] = val


# =============================================================================
# Optimized matrix-vector multiply with fused bias and activation
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _matmul_bias_crelu(
    const DTYPE_t[::1] input_vec,
    const DTYPE_t[:, ::1] weight,  # Row-major: [out_size, in_size]
    const DTYPE_t[::1] bias,
    DTYPE_t[::1] output,
    Py_ssize_t out_size,
    Py_ssize_t in_size
) noexcept nogil:
    """
    Fused: output = clipped_relu(input @ weight.T + bias)

    Processes 4 output neurons at a time for better instruction pipelining.
    """
    cdef Py_ssize_t i, j
    cdef DTYPE_t sum0, sum1, sum2, sum3
    cdef Py_ssize_t out_size_4 = (out_size // 4) * 4
    cdef DTYPE_t val

    # Process 4 outputs at a time (loop unrolling)
    for i in range(0, out_size_4, 4):
        sum0 = bias[i]
        sum1 = bias[i + 1]
        sum2 = bias[i + 2]
        sum3 = bias[i + 3]

        for j in range(in_size):
            val = input_vec[j]
            sum0 = sum0 + val * weight[i, j]
            sum1 = sum1 + val * weight[i + 1, j]
            sum2 = sum2 + val * weight[i + 2, j]
            sum3 = sum3 + val * weight[i + 3, j]

        # Fused clipped ReLU
        output[i] = 0.0 if sum0 < 0.0 else (1.0 if sum0 > 1.0 else sum0)
        output[i + 1] = 0.0 if sum1 < 0.0 else (1.0 if sum1 > 1.0 else sum1)
        output[i + 2] = 0.0 if sum2 < 0.0 else (1.0 if sum2 > 1.0 else sum2)
        output[i + 3] = 0.0 if sum3 < 0.0 else (1.0 if sum3 > 1.0 else sum3)

    # Handle remainder
    for i in range(out_size_4, out_size):
        sum0 = bias[i]
        for j in range(in_size):
            sum0 = sum0 + input_vec[j] * weight[i, j]
        output[i] = 0.0 if sum0 < 0.0 else (1.0 if sum0 > 1.0 else sum0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float _matmul_bias_scalar(
    const DTYPE_t[::1] input_vec,
    const DTYPE_t[:, ::1] weight,  # Shape [1, in_size]
    const DTYPE_t[::1] bias,
    Py_ssize_t in_size
) noexcept nogil:
    """Final layer: single output, no activation."""
    cdef Py_ssize_t j
    cdef float sum_val = bias[0]

    for j in range(in_size):
        sum_val = sum_val + input_vec[j] * weight[0, j]

    return sum_val


# =============================================================================
# DNN Evaluation
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float dnn_evaluate_incremental(
    DTYPE_t[::1] accumulator,
    DTYPE_t[:, ::1] l2_weight,
    DTYPE_t[::1] l2_bias,
    DTYPE_t[:, ::1] l3_weight,
    DTYPE_t[::1] l3_bias,
    DTYPE_t[:, ::1] l4_weight,
    DTYPE_t[::1] l4_bias,
    DTYPE_t[::1] l2_buf,
    DTYPE_t[::1] l3_buf,
    DTYPE_t[::1] acc_clipped
) noexcept:
    """Fast DNN incremental evaluation with fused operations."""
    cdef Py_ssize_t acc_size = accumulator.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef Py_ssize_t l3_size = l3_bias.shape[0]
    cdef float output

    with nogil:
        # Clipped ReLU on accumulator
        clipped_relu_copy(accumulator, acc_clipped)

        # L2: fused matmul + bias + clipped_relu
        _matmul_bias_crelu(acc_clipped, l2_weight, l2_bias, l2_buf, l2_size, acc_size)

        # L3: fused matmul + bias + clipped_relu
        _matmul_bias_crelu(l2_buf, l3_weight, l3_bias, l3_buf, l3_size, l2_size)

        # L4: final output (no activation)
        output = _matmul_bias_scalar(l3_buf, l4_weight, l4_bias, l3_size)

    return output


# =============================================================================
# NNUE Evaluation (FP32)
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental(
    DTYPE_t[::1] white_accumulator,
    DTYPE_t[::1] black_accumulator,
    bint stm,
    DTYPE_t[:, ::1] l1_weight,
    DTYPE_t[::1] l1_bias,
    DTYPE_t[:, ::1] l2_weight,
    DTYPE_t[::1] l2_bias,
    DTYPE_t[:, ::1] l3_weight,
    DTYPE_t[::1] l3_bias,
    DTYPE_t[::1] hidden_buf,
    DTYPE_t[::1] l1_buf,
    DTYPE_t[::1] l2_buf,
    DTYPE_t[::1] white_clipped,
    DTYPE_t[::1] black_clipped
) noexcept:
    """
    Fast NNUE incremental evaluation.

    Optimizations:
    - Fused clipped_relu + concatenation
    - Loop unrolling in matmul
    - All operations in nogil block
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef DTYPE_t w_val, b_val
    cdef float output

    with nogil:
        # Fused clipped_relu + concatenation
        if stm:  # White to move: [white, black]
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)
                hidden_buf[hidden_size + i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
        else:  # Black to move: [black, white]
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
                hidden_buf[hidden_size + i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)

        # L1: 512 -> 32 (fused matmul + bias + crelu)
        _matmul_bias_crelu(hidden_buf, l1_weight, l1_bias, l1_buf, l1_size, concat_size)

        # L2: 32 -> 32 (fused matmul + bias + crelu)
        _matmul_bias_crelu(l1_buf, l2_weight, l2_bias, l2_buf, l2_size, l1_size)

        # L3: 32 -> 1 (no activation)
        output = _matmul_bias_scalar(l2_buf, l3_weight, l3_bias, l2_size)

    return output


# =============================================================================
# NNUE Evaluation with INT8 Quantized L1
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _quantize_to_int8(
    const DTYPE_t[::1] input_buf,
    INT8_t[::1] output_buf,
    Py_ssize_t size
) noexcept nogil:
    """Quantize float32 [0,1] to int8 [0,127]."""
    cdef Py_ssize_t i
    cdef float val
    cdef int q_val

    for i in range(size):
        val = input_buf[i]
        q_val = <int>(val * 127.0 + 0.5)
        if q_val < 0:
            q_val = 0
        elif q_val > 127:
            q_val = 127
        output_buf[i] = <INT8_t>q_val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _quantize_to_int16(
    const DTYPE_t[::1] input_buf,
    INT16_t[::1] output_buf,
    Py_ssize_t size
) noexcept nogil:
    """Quantize float32 [0,1] to int16 [0,16383] (14-bit for headroom)."""
    cdef Py_ssize_t i
    cdef float val
    cdef int q_val

    for i in range(size):
        val = input_buf[i]
        q_val = <int>(val * 16383.0 + 0.5)
        if q_val < 0:
            q_val = 0
        elif q_val > 16383:
            q_val = 16383
        output_buf[i] = <INT16_t>q_val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _matmul_int8_dequant_crelu(
    const INT8_t[::1] input_q,
    const INT8_t[:, ::1] weight_q,
    const DTYPE_t[::1] bias,
    DTYPE_t[::1] output,
    float combined_scale,
    Py_ssize_t out_size,
    Py_ssize_t in_size
) noexcept nogil:
    """INT8 matmul with dequantization and fused clipped ReLU."""
    cdef Py_ssize_t i, j
    cdef INT32_t acc
    cdef float result

    for i in range(out_size):
        acc = 0
        for j in range(in_size):
            acc = acc + <INT32_t>input_q[j] * <INT32_t>weight_q[i, j]

        result = <float>acc * combined_scale + bias[i]

        if result < 0.0:
            output[i] = 0.0
        elif result > 1.0:
            output[i] = 1.0
        else:
            output[i] = result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _matmul_int16_dequant_crelu(
    const INT16_t[::1] input_q,
    const INT16_t[:, ::1] weight_q,
    const DTYPE_t[::1] bias,
    DTYPE_t[::1] output,
    float combined_scale,
    Py_ssize_t out_size,
    Py_ssize_t in_size
) noexcept nogil:
    """INT16 matmul with dequantization and fused clipped ReLU."""
    cdef Py_ssize_t i, j
    cdef INT64_t acc
    cdef float result

    for i in range(out_size):
        acc = 0
        for j in range(in_size):
            acc = acc + <INT64_t>input_q[j] * <INT64_t>weight_q[i, j]

        result = <float>acc * combined_scale + bias[i]

        if result < 0.0:
            output[i] = 0.0
        elif result > 1.0:
            output[i] = 1.0
        else:
            output[i] = result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental_int8(
    DTYPE_t[::1] white_accumulator,
    DTYPE_t[::1] black_accumulator,
    bint stm,
    INT8_t[:, ::1] l1_weight_q,
    DTYPE_t[::1] l1_bias,
    float l1_combined_scale,
    DTYPE_t[:, ::1] l2_weight,
    DTYPE_t[::1] l2_bias,
    DTYPE_t[:, ::1] l3_weight,
    DTYPE_t[::1] l3_bias,
    DTYPE_t[::1] hidden_buf,
    INT8_t[::1] hidden_buf_q,
    DTYPE_t[::1] l1_buf,
    DTYPE_t[::1] l2_buf,
    DTYPE_t[::1] white_clipped,
    DTYPE_t[::1] black_clipped
) noexcept:
    """NNUE evaluation with INT8 quantized L1 layer."""
    cdef Py_ssize_t i
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef DTYPE_t w_val, b_val
    cdef float output

    with nogil:
        # Fused clipped_relu + concatenation
        if stm:
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)
                hidden_buf[hidden_size + i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
        else:
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
                hidden_buf[hidden_size + i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)

        # Quantize input
        _quantize_to_int8(hidden_buf, hidden_buf_q, concat_size)

        # L1: INT8 matmul + dequant + crelu
        _matmul_int8_dequant_crelu(
            hidden_buf_q, l1_weight_q, l1_bias, l1_buf,
            l1_combined_scale, l1_size, concat_size
        )

        # L2: FP32 (fused)
        _matmul_bias_crelu(l1_buf, l2_weight, l2_bias, l2_buf, l2_size, l1_size)

        # L3: Final output
        output = _matmul_bias_scalar(l2_buf, l3_weight, l3_bias, l2_size)

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float nnue_evaluate_incremental_int16(
    DTYPE_t[::1] white_accumulator,
    DTYPE_t[::1] black_accumulator,
    bint stm,
    INT16_t[:, ::1] l1_weight_q,
    DTYPE_t[::1] l1_bias,
    float l1_combined_scale,
    DTYPE_t[:, ::1] l2_weight,
    DTYPE_t[::1] l2_bias,
    DTYPE_t[:, ::1] l3_weight,
    DTYPE_t[::1] l3_bias,
    DTYPE_t[::1] hidden_buf,
    INT16_t[::1] hidden_buf_q,
    DTYPE_t[::1] l1_buf,
    DTYPE_t[::1] l2_buf,
    DTYPE_t[::1] white_clipped,
    DTYPE_t[::1] black_clipped
) noexcept:
    """NNUE evaluation with INT16 quantized L1 layer."""
    cdef Py_ssize_t i
    cdef Py_ssize_t hidden_size = white_accumulator.shape[0]
    cdef Py_ssize_t concat_size = hidden_size * 2
    cdef Py_ssize_t l1_size = l1_bias.shape[0]
    cdef Py_ssize_t l2_size = l2_bias.shape[0]
    cdef DTYPE_t w_val, b_val
    cdef float output

    with nogil:
        # Fused clipped_relu + concatenation
        if stm:
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)
                hidden_buf[hidden_size + i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
        else:
            for i in range(hidden_size):
                w_val = white_accumulator[i]
                b_val = black_accumulator[i]
                hidden_buf[i] = 0.0 if b_val < 0.0 else (1.0 if b_val > 1.0 else b_val)
                hidden_buf[hidden_size + i] = 0.0 if w_val < 0.0 else (1.0 if w_val > 1.0 else w_val)

        # Quantize input
        _quantize_to_int16(hidden_buf, hidden_buf_q, concat_size)

        # L1: INT16 matmul + dequant + crelu
        _matmul_int16_dequant_crelu(
            hidden_buf_q, l1_weight_q, l1_bias, l1_buf,
            l1_combined_scale, l1_size, concat_size
        )

        # L2: FP32 (fused)
        _matmul_bias_crelu(l1_buf, l2_weight, l2_bias, l2_buf, l2_size, l1_size)

        # L3: Final output
        output = _matmul_bias_scalar(l2_buf, l3_weight, l3_bias, l2_size)

    return output


# =============================================================================
# Accumulator update functions
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void accumulator_add_features(
    DTYPE_t[::1] accumulator,
    DTYPE_t[:, ::1] weights,
    INT64_t[::1] features,
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
    DTYPE_t[::1] accumulator,
    DTYPE_t[:, ::1] weights,
    INT64_t[::1] features,
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
    DTYPE_t[::1] accumulator,
    DTYPE_t[:, ::1] weights,
    object added_features,
    object removed_features,
    Py_ssize_t max_feature
) noexcept:
    """Update DNN accumulator with added/removed features."""
    cdef Py_ssize_t j, f
    cdef Py_ssize_t acc_size = accumulator.shape[0]
    cdef list added_list, removed_list

    added_list = [f for f in added_features if 0 <= f < max_feature]
    removed_list = [f for f in removed_features if 0 <= f < max_feature]

    for f in added_list:
        for j in range(acc_size):
            accumulator[j] = accumulator[j] + weights[j, f]

    for f in removed_list:
        for j in range(acc_size):
            accumulator[j] = accumulator[j] - weights[j, f]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void nnue_update_accumulator(
    DTYPE_t[::1] white_accumulator,
    DTYPE_t[::1] black_accumulator,
    DTYPE_t[:, ::1] weights,
    object added_white,
    object removed_white,
    object added_black,
    object removed_black,
    Py_ssize_t max_feature
) noexcept:
    """Update NNUE accumulators with added/removed features."""
    cdef Py_ssize_t j, f
    cdef Py_ssize_t acc_size = white_accumulator.shape[0]
    cdef list aw_list, rw_list, ab_list, rb_list

    aw_list = [f for f in added_white if 0 <= f < max_feature]
    rw_list = [f for f in removed_white if 0 <= f < max_feature]
    ab_list = [f for f in added_black if 0 <= f < max_feature]
    rb_list = [f for f in removed_black if 0 <= f < max_feature]

    for f in aw_list:
        for j in range(acc_size):
            white_accumulator[j] = white_accumulator[j] + weights[j, f]
    for f in rw_list:
        for j in range(acc_size):
            white_accumulator[j] = white_accumulator[j] - weights[j, f]

    for f in ab_list:
        for j in range(acc_size):
            black_accumulator[j] = black_accumulator[j] + weights[j, f]
    for f in rb_list:
        for j in range(acc_size):
            black_accumulator[j] = black_accumulator[j] - weights[j, f]


# =============================================================================
# Feature index computation
# =============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int get_piece_index(int piece_type, bint is_friendly) noexcept nogil:
    """Convert piece type and color to index (0-9)."""
    if piece_type == 6:
        return -1
    cdef int type_idx = piece_type - 1
    cdef int color_idx = 1 if is_friendly else 0
    return type_idx + color_idx * 5


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
    """Calculate DNN feature index (768-dimensional encoding)."""
    cdef int adj_square = square
    cdef int type_idx, piece_idx

    if not perspective:
        adj_square = flip_square(square)

    type_idx = piece_type - 1
    piece_idx = type_idx + (0 if is_friendly else 6)

    return piece_idx * 64 + adj_square


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int move_to_int_fast(int from_sq, int to_sq, int promo) noexcept nogil:
    """Convert move to integer key."""
    return from_sq | (to_sq << 6) | (promo << 12)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple int_to_move_fast(int key):
    """Convert integer key back to (from_sq, to_sq, promo) tuple."""
    cdef int from_sq = key & 0x3F
    cdef int to_sq = (key >> 6) & 0x3F
    cdef int promo = (key >> 12) & 0xF
    return (from_sq, to_sq, promo if promo else None)