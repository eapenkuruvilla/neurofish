"""
Pure Python fallback for nn_ops_fast Cython module.

This module provides the same interface as the Cython version but uses
NumPy operations. It's used when the Cython extension isn't compiled.

For best performance, compile the Cython version with:
    python setup.py build_ext --inplace
"""

from typing import Set

import numpy as np


def clipped_relu_inplace(x: np.ndarray) -> None:
    """In-place clipped ReLU: x = clip(x, 0, 1)"""
    np.maximum(x, 0, out=x)
    np.minimum(x, 1, out=x)


def clipped_relu_copy(src: np.ndarray, dst: np.ndarray) -> None:
    """Copy with clipped ReLU: dst = clip(src, 0, 1)"""
    np.copyto(dst, src)
    np.maximum(dst, 0, out=dst)
    np.minimum(dst, 1, out=dst)


def dnn_evaluate_incremental(
        accumulator: np.ndarray,
        l2_weight: np.ndarray,
        l2_bias: np.ndarray,
        l3_weight: np.ndarray,
        l3_bias: np.ndarray,
        l4_weight: np.ndarray,
        l4_bias: np.ndarray,
        l2_buf: np.ndarray,
        l3_buf: np.ndarray,
        acc_clipped: np.ndarray
) -> float:
    """Fast DNN incremental evaluation."""
    # Clipped ReLU on accumulator
    np.copyto(acc_clipped, accumulator)
    np.maximum(acc_clipped, 0, out=acc_clipped)
    np.minimum(acc_clipped, 1, out=acc_clipped)

    # L2
    np.dot(acc_clipped, l2_weight.T, out=l2_buf)
    l2_buf += l2_bias
    np.maximum(l2_buf, 0, out=l2_buf)
    np.minimum(l2_buf, 1, out=l2_buf)

    # L3
    np.dot(l2_buf, l3_weight.T, out=l3_buf)
    l3_buf += l3_bias
    np.maximum(l3_buf, 0, out=l3_buf)
    np.minimum(l3_buf, 1, out=l3_buf)

    # L4 (no activation)
    return float(np.dot(l3_buf, l4_weight.T) + l4_bias)


def nnue_evaluate_incremental(
        white_accumulator: np.ndarray,
        black_accumulator: np.ndarray,
        stm: bool,
        l1_weight: np.ndarray,
        l1_bias: np.ndarray,
        l2_weight: np.ndarray,
        l2_bias: np.ndarray,
        l3_weight: np.ndarray,
        l3_bias: np.ndarray,
        hidden_buf: np.ndarray,
        l1_buf: np.ndarray,
        l2_buf: np.ndarray,
        white_clipped: np.ndarray,
        black_clipped: np.ndarray
) -> float:
    """Fast NNUE incremental evaluation."""
    hidden_size = white_accumulator.shape[0]

    # Clipped ReLU on both accumulators
    clipped_relu_copy(white_accumulator, white_clipped)
    clipped_relu_copy(black_accumulator, black_clipped)

    # Concatenate based on perspective
    if stm:
        hidden_buf[:hidden_size] = white_clipped
        hidden_buf[hidden_size:] = black_clipped
    else:
        hidden_buf[:hidden_size] = black_clipped
        hidden_buf[hidden_size:] = white_clipped

    # L1
    np.dot(hidden_buf, l1_weight.T, out=l1_buf)
    l1_buf += l1_bias
    np.maximum(l1_buf, 0, out=l1_buf)
    np.minimum(l1_buf, 1, out=l1_buf)

    # L2
    np.dot(l1_buf, l2_weight.T, out=l2_buf)
    l2_buf += l2_bias
    np.maximum(l2_buf, 0, out=l2_buf)
    np.minimum(l2_buf, 1, out=l2_buf)

    # L3 (no activation)
    return float(np.dot(l2_buf, l3_weight.T) + l3_bias)


def nnue_evaluate_incremental_int8(
        white_accumulator: np.ndarray,
        black_accumulator: np.ndarray,
        stm: bool,
        l1_weight_q: np.ndarray,  # INT8 quantized weights
        l1_bias: np.ndarray,  # FP32 bias
        l1_combined_scale: float,  # Pre-computed scale: input_scale * weight_scale
        l2_weight: np.ndarray,
        l2_bias: np.ndarray,
        l3_weight: np.ndarray,
        l3_bias: np.ndarray,
        hidden_buf: np.ndarray,  # FP32 buffer for clipped values
        hidden_buf_q: np.ndarray,  # INT8 buffer for quantized input
        l1_buf: np.ndarray,
        l2_buf: np.ndarray,
        white_clipped: np.ndarray,
        black_clipped: np.ndarray
) -> float:
    """
    NNUE incremental evaluation with INT8 quantized L1 layer.

    Quantization scheme:
    - Input (hidden_buf) is in [0, 1], quantized to [0, 127] as INT8
    - Weights are pre-quantized to [-127, 127] as INT8
    - Accumulation is done in INT32 to prevent overflow
    - Result is dequantized using pre-computed combined scale
    """
    hidden_size = white_accumulator.shape[0]

    # Clipped ReLU on both accumulators
    clipped_relu_copy(white_accumulator, white_clipped)
    clipped_relu_copy(black_accumulator, black_clipped)

    # Concatenate based on perspective
    if stm:
        hidden_buf[:hidden_size] = white_clipped
        hidden_buf[hidden_size:] = black_clipped
    else:
        hidden_buf[:hidden_size] = black_clipped
        hidden_buf[hidden_size:] = white_clipped

    # Quantize input: [0, 1] -> [0, 127]
    np.clip(np.round(hidden_buf * 127.0), 0, 127, out=hidden_buf)
    hidden_buf_q[:] = hidden_buf.astype(np.int8)

    # L1: Quantized matmul with INT32 accumulation
    # result = hidden_buf_q @ l1_weight_q.T (accumulated in INT32)
    result_q = np.dot(hidden_buf_q.astype(np.int32), l1_weight_q.T.astype(np.int32))

    # Dequantize and add bias
    l1_buf[:] = result_q.astype(np.float32) * l1_combined_scale + l1_bias

    # Clipped ReLU
    np.maximum(l1_buf, 0, out=l1_buf)
    np.minimum(l1_buf, 1, out=l1_buf)

    # L2 (FP32)
    np.dot(l1_buf, l2_weight.T, out=l2_buf)
    l2_buf += l2_bias
    np.maximum(l2_buf, 0, out=l2_buf)
    np.minimum(l2_buf, 1, out=l2_buf)

    # L3 (no activation)
    return float(np.dot(l2_buf, l3_weight.T) + l3_bias)


def nnue_evaluate_incremental_int16(
        white_accumulator: np.ndarray,
        black_accumulator: np.ndarray,
        stm: bool,
        l1_weight_q: np.ndarray,  # INT16 quantized weights
        l1_bias: np.ndarray,  # FP32 bias
        l1_combined_scale: float,  # Pre-computed scale: input_scale * weight_scale
        l2_weight: np.ndarray,
        l2_bias: np.ndarray,
        l3_weight: np.ndarray,
        l3_bias: np.ndarray,
        hidden_buf: np.ndarray,  # FP32 buffer for clipped values
        hidden_buf_q: np.ndarray,  # INT16 buffer for quantized input
        l1_buf: np.ndarray,
        l2_buf: np.ndarray,
        white_clipped: np.ndarray,
        black_clipped: np.ndarray
) -> float:
    """
    NNUE incremental evaluation with INT16 quantized L1 layer.

    Quantization scheme:
    - Input (hidden_buf) is in [0, 1], quantized to [0, 32767] as INT16
    - Weights are pre-quantized to [-32767, 32767] as INT16
    - Accumulation is done in INT64 to prevent overflow
      (INT32 would overflow: 512 * 32767 * 32767 > 2^31)
    - Result is dequantized using pre-computed combined scale
    """
    hidden_size = white_accumulator.shape[0]

    # Clipped ReLU on both accumulators
    clipped_relu_copy(white_accumulator, white_clipped)
    clipped_relu_copy(black_accumulator, black_clipped)

    # Concatenate based on perspective
    if stm:
        hidden_buf[:hidden_size] = white_clipped
        hidden_buf[hidden_size:] = black_clipped
    else:
        hidden_buf[:hidden_size] = black_clipped
        hidden_buf[hidden_size:] = white_clipped

    # Quantize input: [0, 1] -> [0, 32767]
    np.clip(np.round(hidden_buf * 32767.0), 0, 32767, out=hidden_buf)
    hidden_buf_q[:] = hidden_buf.astype(np.int16)

    # L1: Quantized matmul with INT64 accumulation (FIX: prevents overflow)
    # Worst case: 512 * 32767 * 32767 = ~549 billion, exceeds INT32_MAX (~2.1 billion)
    result_q = np.dot(hidden_buf_q.astype(np.int64), l1_weight_q.T.astype(np.int64))

    # Dequantize and add bias
    l1_buf[:] = result_q.astype(np.float32) * l1_combined_scale + l1_bias

    # Clipped ReLU
    np.maximum(l1_buf, 0, out=l1_buf)
    np.minimum(l1_buf, 1, out=l1_buf)

    # L2 (FP32)
    np.dot(l1_buf, l2_weight.T, out=l2_buf)
    l2_buf += l2_bias
    np.maximum(l2_buf, 0, out=l2_buf)
    np.minimum(l2_buf, 1, out=l2_buf)

    # L3 (no activation)
    return float(np.dot(l2_buf, l3_weight.T) + l3_bias)


def accumulator_add_features(
        accumulator: np.ndarray,
        weights: np.ndarray,
        features: np.ndarray,
        max_feature: int
) -> None:
    """Add weight columns for given features to accumulator."""
    valid = features[(features >= 0) & (features < max_feature)]
    if len(valid) > 0:
        accumulator += weights[:, valid].sum(axis=1)


def accumulator_remove_features(
        accumulator: np.ndarray,
        weights: np.ndarray,
        features: np.ndarray,
        max_feature: int
) -> None:
    """Remove weight columns for given features from accumulator."""
    valid = features[(features >= 0) & (features < max_feature)]
    if len(valid) > 0:
        accumulator -= weights[:, valid].sum(axis=1)


def dnn_update_accumulator(
        accumulator: np.ndarray,
        weights: np.ndarray,
        added_features: Set[int],
        removed_features: Set[int],
        max_feature: int
) -> None:
    """Update DNN accumulator with added/removed features."""
    valid_added = [f for f in added_features if 0 <= f < max_feature]
    valid_removed = [f for f in removed_features if 0 <= f < max_feature]

    if valid_added:
        accumulator += weights[:, valid_added].sum(axis=1)
    if valid_removed:
        accumulator -= weights[:, valid_removed].sum(axis=1)


def nnue_update_accumulator(
        white_accumulator: np.ndarray,
        black_accumulator: np.ndarray,
        weights: np.ndarray,
        added_white: Set[int],
        removed_white: Set[int],
        added_black: Set[int],
        removed_black: Set[int],
        max_feature: int
) -> None:
    """Update NNUE accumulators with added/removed features."""
    # White accumulator
    valid_aw = [f for f in added_white if 0 <= f < max_feature]
    valid_rw = [f for f in removed_white if 0 <= f < max_feature]
    if valid_aw:
        white_accumulator += weights[:, valid_aw].sum(axis=1)
    if valid_rw:
        white_accumulator -= weights[:, valid_rw].sum(axis=1)

    # Black accumulator
    valid_ab = [f for f in added_black if 0 <= f < max_feature]
    valid_rb = [f for f in removed_black if 0 <= f < max_feature]
    if valid_ab:
        black_accumulator += weights[:, valid_ab].sum(axis=1)
    if valid_rb:
        black_accumulator -= weights[:, valid_rb].sum(axis=1)


def get_piece_index(piece_type: int, is_friendly: bool) -> int:
    """Convert piece type and color to index (0-9)."""
    if piece_type == 6:  # KING
        return -1
    type_idx = piece_type - 1
    color_idx = 1 if is_friendly else 0
    return type_idx + color_idx * 5


def get_nnue_feature_index(king_sq: int, piece_sq: int, piece_type: int, is_friendly: bool) -> int:
    """Calculate NNUE feature index."""
    piece_idx = get_piece_index(piece_type, is_friendly)
    if piece_idx == -1:
        return -1
    return king_sq * 640 + piece_sq * 10 + piece_idx


def flip_square(square: int) -> int:
    """Flip square vertically (A1 <-> A8)."""
    rank = square // 8
    file = square % 8
    return (7 - rank) * 8 + file


def get_dnn_feature_index(square: int, piece_type: int, is_friendly: bool, perspective: bool) -> int:
    """Calculate DNN feature index (768-dimensional encoding)."""
    adj_square = square if perspective else flip_square(square)

    type_map = {6: 0, 5: 1, 4: 2, 3: 3, 2: 4, 1: 5}
    type_idx = type_map.get(piece_type, 5)

    piece_idx = type_idx + (0 if is_friendly else 6)
    return adj_square * 12 + piece_idx


def move_to_int_fast(from_sq: int, to_sq: int, promo: int) -> int:
    """Convert move to integer key."""
    return from_sq | (to_sq << 6) | (promo << 12)


def int_to_move_fast(key: int):
    """Convert integer key back to (from_sq, to_sq, promo) tuple."""
    from_sq = key & 0x3F
    to_sq = (key >> 6) & 0x3F
    promo = (key >> 12) & 0xF
    return (from_sq, to_sq, promo if promo else None)
