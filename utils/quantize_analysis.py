#!/usr/bin/env python3
"""
quantize_analysis.py - Analyze quantization potential for NNUE neural network.

STANDALONE VERSION - Works without PyTorch/zstandard (uses synthetic data)

This program evaluates:
1. Weight distribution analysis per layer  
2. Quantization error (INT8 vs INT16 vs FP32)
3. Output accuracy degradation on synthetic test data
4. Speed benchmarks for quantized inference

Focus: First hidden layer (512,32) x (32,32) matmul operations

Usage:
    python quantize_analysis.py --synthetic
    python quantize_analysis.py --weights nnue_weights.npz
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

# NNUE Architecture constants
NNUE_INPUT_SIZE = 40960
NNUE_HIDDEN_SIZE = 256

# Optional imports for full functionality
HAS_TORCH = False
HAS_SHARD_IO = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

try:
    from nn_train.shard_io import ShardReader, discover_shards

    HAS_SHARD_IO = True
except ImportError:
    pass


@dataclass
class QuantizationStats:
    dtype: str
    scale: float
    zero_point: int
    min_val: float
    max_val: float
    quant_error_mse: float
    quant_error_max: float
    bits: int


def analyze_weight_distribution(weights: np.ndarray, name: str) -> Dict[str, Any]:
    return {
        'name': name,
        'shape': weights.shape,
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'abs_max': float(np.max(np.abs(weights))),
        'sparsity': float(np.sum(np.abs(weights) < 1e-6) / weights.size),
        'percentile_1': float(np.percentile(weights, 1)),
        'percentile_99': float(np.percentile(weights, 99)),
    }


def quantize_symmetric_int8(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    abs_max = np.max(np.abs(weights))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    return quantized, scale


def quantize_symmetric_int16(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    abs_max = np.max(np.abs(weights))
    scale = abs_max / 32767.0 if abs_max > 0 else 1.0
    quantized = np.clip(np.round(weights / scale), -32767, 32767).astype(np.int16)
    return quantized, scale


def dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale


def compute_quantization_error(original: np.ndarray, quantized: np.ndarray,
                               scale: float) -> Tuple[float, float]:
    dequant = dequantize(quantized, scale)
    error = original - dequant
    return float(np.mean(error ** 2)), float(np.max(np.abs(error)))


def generate_synthetic_weights() -> Dict[str, np.ndarray]:
    """Generate realistic synthetic NNUE weights."""
    np.random.seed(42)
    weights = {}

    # Feature transformer: (256, 40960) - sparse, small weights
    weights['ft.weight'] = np.random.randn(NNUE_HIDDEN_SIZE, NNUE_INPUT_SIZE).astype(np.float32) * 0.02
    weights['ft.bias'] = np.random.randn(NNUE_HIDDEN_SIZE).astype(np.float32) * 0.01

    # L1: (32, 512) - Xavier init
    weights['l1.weight'] = np.random.randn(32, 512).astype(np.float32) * np.sqrt(2.0 / 512)
    weights['l1.bias'] = np.random.randn(32).astype(np.float32) * 0.01

    # L2: (32, 32)
    weights['l2.weight'] = np.random.randn(32, 32).astype(np.float32) * np.sqrt(2.0 / 32)
    weights['l2.bias'] = np.random.randn(32).astype(np.float32) * 0.01

    # L3: (1, 32)
    weights['l3.weight'] = np.random.randn(1, 32).astype(np.float32) * np.sqrt(2.0 / 32)
    weights['l3.bias'] = np.random.randn(1).astype(np.float32) * 0.01

    return weights


def generate_synthetic_positions(num_positions: int = 5000) -> List[Dict]:
    """Generate synthetic test positions."""
    np.random.seed(123)
    positions = []

    for _ in range(num_positions):
        num_pieces = np.random.randint(8, 31)
        positions.append({
            'white_features': np.random.choice(NNUE_INPUT_SIZE, size=num_pieces, replace=False).tolist(),
            'black_features': np.random.choice(NNUE_INPUT_SIZE, size=num_pieces, replace=False).tolist(),
            'stm': np.random.choice([0, 1]),
        })

    return positions


def load_positions_from_shards(data_dir: str, max_positions: int = 10000) -> List[Dict]:
    """
    Load real positions from shard files using shard_io module.

    Args:
        data_dir: Directory containing NNUE shard files
        max_positions: Maximum number of positions to load

    Returns:
        List of position dicts with white_features, black_features, stm, score_cp
    """
    if not HAS_SHARD_IO:
        raise ImportError("shard_io module not available. Install zstandard or use --synthetic")

    # Discover shard files
    shards = discover_shards(data_dir, 'NNUE')
    if not shards:
        raise FileNotFoundError(f"No NNUE shards found in {data_dir}")

    print(f"  Found {len(shards)} shard files")

    # Read positions from shards
    reader = ShardReader('NNUE')
    positions = []

    for shard_path in shards:
        try:
            shard_positions = reader.read_all_positions(shard_path, include_fen=False, skip_diagnostic=True)
            positions.extend(shard_positions)
            print(f"  Loaded {len(shard_positions)} from {os.path.basename(shard_path)}, total: {len(positions)}")

            if len(positions) >= max_positions:
                break
        except Exception as e:
            print(f"  Warning: Failed to read {shard_path}: {e}")
            continue

    # Ensure consistent format
    formatted_positions = []
    for pos in positions[:max_positions]:
        if 'white_features' in pos and 'black_features' in pos:
            formatted_positions.append({
                'white_features': pos['white_features'],
                'black_features': pos['black_features'],
                'stm': pos.get('stm', 1),
                'score_cp': pos.get('score_cp', 0),
            })

    return formatted_positions


class NNUEInferenceFP32:
    def __init__(self, weights: Dict[str, np.ndarray]):
        self.ft_weight = weights['ft.weight'].astype(np.float32)
        self.ft_bias = weights['ft.bias'].astype(np.float32)
        self.l1_weight = weights['l1.weight'].astype(np.float32)
        self.l1_bias = weights['l1.bias'].astype(np.float32)
        self.l2_weight = weights['l2.weight'].astype(np.float32)
        self.l2_bias = weights['l2.bias'].astype(np.float32)
        self.l3_weight = weights['l3.weight'].astype(np.float32)
        self.l3_bias = weights['l3.bias'].astype(np.float32)

    def evaluate_single(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        white_acc = self.ft_bias.copy()
        for f in white_features:
            if 0 <= f < self.ft_weight.shape[1]:
                white_acc += self.ft_weight[:, f]

        black_acc = self.ft_bias.copy()
        for f in black_features:
            if 0 <= f < self.ft_weight.shape[1]:
                black_acc += self.ft_weight[:, f]

        white_hidden = np.clip(white_acc, 0, 1)
        black_hidden = np.clip(black_acc, 0, 1)

        hidden = np.concatenate([white_hidden, black_hidden] if stm else [black_hidden, white_hidden])

        x = np.clip(np.dot(hidden, self.l1_weight.T) + self.l1_bias, 0, 1)
        x = np.clip(np.dot(x, self.l2_weight.T) + self.l2_bias, 0, 1)
        return float((np.dot(x, self.l3_weight.T) + self.l3_bias)[0])

    def evaluate_batch(self, white_features_batch, black_features_batch, stm_batch):
        return np.array([self.evaluate_single(wf, bf, stm)
                         for wf, bf, stm in zip(white_features_batch, black_features_batch, stm_batch)])


class NNUEInferenceQuantizedL1:
    def __init__(self, weights: Dict[str, np.ndarray], bits: int = 8):
        self.bits = bits
        self.ft_weight = weights['ft.weight'].astype(np.float32)
        self.ft_bias = weights['ft.bias'].astype(np.float32)

        l1_weight_fp32 = weights['l1.weight'].astype(np.float32)
        if bits == 8:
            self.l1_weight_q, self.l1_scale = quantize_symmetric_int8(l1_weight_fp32)
        else:
            self.l1_weight_q, self.l1_scale = quantize_symmetric_int16(l1_weight_fp32)

        self.l1_bias = weights['l1.bias'].astype(np.float32)
        self.l2_weight = weights['l2.weight'].astype(np.float32)
        self.l2_bias = weights['l2.bias'].astype(np.float32)
        self.l3_weight = weights['l3.weight'].astype(np.float32)
        self.l3_bias = weights['l3.bias'].astype(np.float32)

    def evaluate_single(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        white_acc = self.ft_bias.copy()
        for f in white_features:
            if 0 <= f < self.ft_weight.shape[1]:
                white_acc += self.ft_weight[:, f]

        black_acc = self.ft_bias.copy()
        for f in black_features:
            if 0 <= f < self.ft_weight.shape[1]:
                black_acc += self.ft_weight[:, f]

        white_hidden = np.clip(white_acc, 0, 1)
        black_hidden = np.clip(black_acc, 0, 1)
        hidden = np.concatenate([white_hidden, black_hidden] if stm else [black_hidden, white_hidden])

        if self.bits == 8:
            hidden_scale = 1.0 / 127.0
            hidden_q = np.clip(np.round(hidden / hidden_scale), 0, 127).astype(np.int8)
            result_q = np.dot(hidden_q.astype(np.int32), self.l1_weight_q.T.astype(np.int32))
        else:
            hidden_scale = 1.0 / 32767.0
            hidden_q = np.clip(np.round(hidden / hidden_scale), 0, 32767).astype(np.int16)
            result_q = np.dot(hidden_q.astype(np.int32), self.l1_weight_q.T.astype(np.int32))

        x = np.clip(result_q.astype(np.float32) * (hidden_scale * self.l1_scale) + self.l1_bias, 0, 1)
        x = np.clip(np.dot(x, self.l2_weight.T) + self.l2_bias, 0, 1)
        return float((np.dot(x, self.l3_weight.T) + self.l3_bias)[0])

    def evaluate_batch(self, white_features_batch, black_features_batch, stm_batch):
        return np.array([self.evaluate_single(wf, bf, stm)
                         for wf, bf, stm in zip(white_features_batch, black_features_batch, stm_batch)])


class NNUEInferenceFullQuantized:
    def __init__(self, weights: Dict[str, np.ndarray], bits: int = 8):
        self.bits = bits
        self.ft_weight = weights['ft.weight'].astype(np.float32)
        self.ft_bias = weights['ft.bias'].astype(np.float32)

        self.weights_q, self.scales, self.biases = {}, {}, {}
        for name in ['l1', 'l2', 'l3']:
            w = weights[f'{name}.weight'].astype(np.float32)
            if bits == 8:
                self.weights_q[name], self.scales[name] = quantize_symmetric_int8(w)
            else:
                self.weights_q[name], self.scales[name] = quantize_symmetric_int16(w)
            self.biases[name] = weights[f'{name}.bias'].astype(np.float32)

    def _quantized_matmul(self, x: np.ndarray, weight_q: np.ndarray, scale: float, bias: np.ndarray):
        max_int = 127 if self.bits == 8 else 32767
        x_scale = 1.0 / max_int
        dtype = np.int8 if self.bits == 8 else np.int16
        x_q = np.clip(np.round(x / x_scale), 0, max_int).astype(dtype)
        result_q = np.dot(x_q.astype(np.int32), weight_q.T.astype(np.int32))
        return result_q.astype(np.float32) * (x_scale * scale) + bias

    def evaluate_single(self, white_features: List[int], black_features: List[int], stm: bool) -> float:
        white_acc = self.ft_bias.copy()
        for f in white_features:
            if 0 <= f < self.ft_weight.shape[1]:
                white_acc += self.ft_weight[:, f]

        black_acc = self.ft_bias.copy()
        for f in black_features:
            if 0 <= f < self.ft_weight.shape[1]:
                black_acc += self.ft_weight[:, f]

        white_hidden = np.clip(white_acc, 0, 1)
        black_hidden = np.clip(black_acc, 0, 1)
        hidden = np.concatenate([white_hidden, black_hidden] if stm else [black_hidden, white_hidden])

        x = np.clip(self._quantized_matmul(hidden, self.weights_q['l1'], self.scales['l1'], self.biases['l1']), 0, 1)
        x = np.clip(self._quantized_matmul(x, self.weights_q['l2'], self.scales['l2'], self.biases['l2']), 0, 1)
        return float(self._quantized_matmul(x, self.weights_q['l3'], self.scales['l3'], self.biases['l3'])[0])

    def evaluate_batch(self, white_features_batch, black_features_batch, stm_batch):
        return np.array([self.evaluate_single(wf, bf, stm)
                         for wf, bf, stm in zip(white_features_batch, black_features_batch, stm_batch)])


def benchmark_matmul_fp32(hidden: np.ndarray, weight: np.ndarray, iterations: int = 10000) -> float:
    for _ in range(100): _ = np.dot(hidden, weight.T)
    start = time.perf_counter()
    for _ in range(iterations): _ = np.dot(hidden, weight.T)
    return iterations / (time.perf_counter() - start)


def benchmark_matmul_int8(hidden: np.ndarray, weight: np.ndarray, iterations: int = 10000) -> float:
    h_q = np.clip(np.round(hidden * 127), 0, 127).astype(np.int8)
    w_q, _ = quantize_symmetric_int8(weight)
    for _ in range(100): _ = np.dot(h_q.astype(np.int32), w_q.T.astype(np.int32))
    start = time.perf_counter()
    for _ in range(iterations): _ = np.dot(h_q.astype(np.int32), w_q.T.astype(np.int32))
    return iterations / (time.perf_counter() - start)


def benchmark_matmul_int16(hidden: np.ndarray, weight: np.ndarray, iterations: int = 10000) -> float:
    h_q = np.clip(np.round(hidden * 32767), 0, 32767).astype(np.int16)
    w_q, _ = quantize_symmetric_int16(weight)
    for _ in range(100): _ = np.dot(h_q.astype(np.int32), w_q.T.astype(np.int32))
    start = time.perf_counter()
    for _ in range(iterations): _ = np.dot(h_q.astype(np.int32), w_q.T.astype(np.int32))
    return iterations / (time.perf_counter() - start)


def compute_layer_quantization_stats(weight: np.ndarray, name: str) -> Dict[str, QuantizationStats]:
    stats = {}
    for dtype, bits, quant_fn in [('int8', 8, quantize_symmetric_int8), ('int16', 16, quantize_symmetric_int16)]:
        w_q, scale = quant_fn(weight)
        mse, max_err = compute_quantization_error(weight, w_q, scale)
        stats[dtype] = QuantizationStats(dtype=dtype, scale=scale, zero_point=0,
                                         min_val=float(np.min(weight)), max_val=float(np.max(weight)),
                                         quant_error_mse=mse, quant_error_max=max_err, bits=bits)
    return stats


def evaluate_accuracy(weights: Dict[str, np.ndarray], positions: List[Dict]) -> Dict[str, Dict]:
    white_features = [p['white_features'] for p in positions]
    black_features = [p['black_features'] for p in positions]
    stm_list = [p['stm'] == 1 for p in positions]

    print(f"Evaluating on {len(positions)} positions...")

    engines = {
        'fp32': NNUEInferenceFP32(weights),
        'int8_l1': NNUEInferenceQuantizedL1(weights, 8),
        'int16_l1': NNUEInferenceQuantizedL1(weights, 16),
        'int8_full': NNUEInferenceFullQuantized(weights, 8),
        'int16_full': NNUEInferenceFullQuantized(weights, 16),
    }

    print("  Running FP32 baseline...")
    preds_fp32 = engines['fp32'].evaluate_batch(white_features, black_features, stm_list)

    results = {'fp32_baseline': {'mean_output': float(np.mean(preds_fp32)), 'std_output': float(np.std(preds_fp32))}}

    for name, key in [('INT8 L1-only', 'int8_l1'), ('INT16 L1-only', 'int16_l1'),
                      ('INT8 full', 'int8_full'), ('INT16 full', 'int16_full')]:
        print(f"  Running {name}...")
        preds = engines[key].evaluate_batch(white_features, black_features, stm_list)
        diff = preds - preds_fp32
        results[key.replace('-', '_') + '_only' if 'l1' in key else key] = {
            'mse_vs_fp32': float(np.mean(diff ** 2)),
            'mae_vs_fp32': float(np.mean(np.abs(diff))),
            'max_err_vs_fp32': float(np.max(np.abs(diff))),
            'corr_vs_fp32': float(np.corrcoef(preds, preds_fp32)[0, 1]) if len(preds) > 1 else 1.0,
            'centipawn_mae_vs_fp32': float(np.mean(np.abs(diff))) * 400,
            'centipawn_max_err_vs_fp32': float(np.max(np.abs(diff))) * 400,
        }

    return results


def benchmark_inference_speed(weights: Dict[str, np.ndarray], iterations: int = 10000) -> Dict[str, float]:
    np.random.seed(42)
    hidden_512 = np.random.rand(512).astype(np.float32)
    hidden_32 = np.random.rand(32).astype(np.float32)
    l1_weight = weights['l1.weight']
    l2_weight = weights['l2.weight']

    results = {}
    print(f"\nBenchmarking L1 layer (512 x 32) x {iterations} iterations...")
    results['l1_fp32'] = benchmark_matmul_fp32(hidden_512, l1_weight, iterations)
    results['l1_int8'] = benchmark_matmul_int8(hidden_512, l1_weight, iterations)
    results['l1_int16'] = benchmark_matmul_int16(hidden_512, l1_weight, iterations)

    print(f"Benchmarking L2 layer (32 x 32) x {iterations} iterations...")
    results['l2_fp32'] = benchmark_matmul_fp32(hidden_32, l2_weight, iterations)
    results['l2_int8'] = benchmark_matmul_int8(hidden_32, l2_weight, iterations)
    results['l2_int16'] = benchmark_matmul_int16(hidden_32, l2_weight, iterations)

    return results


# ============== REPORTING ==============

def print_weight_analysis(weights: Dict[str, np.ndarray]):
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for name in ['ft.weight', 'ft.bias', 'l1.weight', 'l1.bias', 'l2.weight', 'l2.bias', 'l3.weight', 'l3.bias']:
        if name in weights:
            stats = analyze_weight_distribution(weights[name], name)
            print(f"\n{stats['name']}:")
            print(f"  Shape:        {stats['shape']}")
            print(f"  Range:        [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Abs max:      {stats['abs_max']:.6f}")
            print(f"  Mean +/- Std: {stats['mean']:.6f} +/- {stats['std']:.6f}")
            print(f"  1st/99th %:   [{stats['percentile_1']:.6f}, {stats['percentile_99']:.6f}]")
            print(f"  Sparsity:     {stats['sparsity'] * 100:.2f}% near-zero")


def print_quantization_error_analysis(weights: Dict[str, np.ndarray]):
    print("\n" + "=" * 80)
    print("QUANTIZATION ERROR ANALYSIS (Weight-level)")
    print("=" * 80)

    for name, label in [('l1.weight', 'l1.weight (512->32)'),
                        ('l2.weight', 'l2.weight (32->32)'),
                        ('l3.weight', 'l3.weight (32->1)')]:
        weight = weights[name]
        stats = compute_layer_quantization_stats(weight, name)

        print(f"\n{label}:")
        print(f"  Shape: {weight.shape}, Elements: {weight.size:,}")

        for dtype in ['int8', 'int16']:
            s = stats[dtype]
            rel_error = s.quant_error_max / s.max_val * 100 if s.max_val != 0 else 0
            print(f"  {dtype.upper()}:")
            print(f"    Scale:         {s.scale:.8f}")
            print(f"    MSE:           {s.quant_error_mse:.2e}")
            print(f"    Max error:     {s.quant_error_max:.6f}")
            print(f"    Relative err:  {rel_error:.2f}% of max weight")


def print_memory_analysis(weights: Dict[str, np.ndarray]):
    print("\n" + "=" * 80)
    print("MEMORY FOOTPRINT ANALYSIS")
    print("=" * 80)

    print("\n+----------------+--------------+--------------+--------------+")
    print("| Layer          |   FP32 (KB)  |   INT16 (KB) |   INT8 (KB)  |")
    print("+----------------+--------------+--------------+--------------+")

    total_fp32, total_int16, total_int8 = 0, 0, 0

    for name in ['ft.weight', 'l1.weight', 'l2.weight', 'l3.weight']:
        if name in weights:
            size = weights[name].size
            fp32_kb, int16_kb, int8_kb = size * 4 / 1024, size * 2 / 1024, size * 1 / 1024
            total_fp32 += fp32_kb
            total_int16 += int16_kb
            total_int8 += int8_kb
            print(f"| {name:<14} | {fp32_kb:>10.1f}   | {int16_kb:>10.1f}   | {int8_kb:>10.1f}   |")

    print("+----------------+--------------+--------------+--------------+")
    print(f"| {'TOTAL':<14} | {total_fp32:>10.1f}   | {total_int16:>10.1f}   | {total_int8:>10.1f}   |")
    print("+----------------+--------------+--------------+--------------+")

    print(f"\n   Memory reduction with INT16: {(1 - total_int16 / total_fp32) * 100:.1f}%")
    print(f"   Memory reduction with INT8:  {(1 - total_int8 / total_fp32) * 100:.1f}%")

    dense_fp32 = sum(weights[n].size * 4 / 1024 for n in ['l1.weight', 'l2.weight', 'l3.weight'])
    dense_int8 = sum(weights[n].size * 1 / 1024 for n in ['l1.weight', 'l2.weight', 'l3.weight'])
    print(f"\n   Dense layers only (L1, L2, L3): FP32={dense_fp32:.1f}KB, INT8={dense_int8:.1f}KB")


def print_accuracy_results(results: Dict):
    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON ON TEST DATA")
    print("=" * 80)

    print("\nFP32 Baseline:")
    print(f"  Mean output:  {results['fp32_baseline']['mean_output']:.6f}")
    print(f"  Std output:   {results['fp32_baseline']['std_output']:.6f}")

    print("\nQuantized vs FP32 Baseline:")
    print("-" * 70)
    print(f"{'Scheme':<16} {'MSE':<12} {'MAE (cp)':<12} {'Max Err (cp)':<14} {'Corr':>8}")
    print("-" * 70)

    for label, key in [('INT8 L1-only', 'int8_l1_only'), ('INT16 L1-only', 'int16_l1_only'),
                       ('INT8 Full', 'int8_full'), ('INT16 Full', 'int16_full')]:
        m = results[key]
        print(f"{label:<16} {m['mse_vs_fp32']:<12.2e} {m['centipawn_mae_vs_fp32']:<12.2f} "
              f"{m['centipawn_max_err_vs_fp32']:<14.2f} {m['corr_vs_fp32']:>8.6f}")
    print("-" * 70)


def print_speed_results(results: Dict):
    print("\n" + "=" * 80)
    print("SPEED BENCHMARK (NumPy, single-threaded)")
    print("=" * 80)

    print("\nNote: NumPy benchmarks simulate quantized ops. Real speedups depend on")
    print("hardware (AVX-512, VNNI) and optimized libraries (oneDNN, etc.).")

    for layer in ['l1', 'l2']:
        fp32, int8, int16 = results[f'{layer}_fp32'], results[f'{layer}_int8'], results[f'{layer}_int16']
        print(f"\n{layer.upper()} Layer:")
        print(f"  FP32:   {fp32:>12,.0f} ops/sec")
        print(f"  INT8:   {int8:>12,.0f} ops/sec ({int8 / fp32:.2f}x vs FP32)")
        print(f"  INT16:  {int16:>12,.0f} ops/sec ({int16 / fp32:.2f}x vs FP32)")


def print_recommendations(accuracy_results: Dict, speed_results: Dict):
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    int8_l1_err = accuracy_results['int8_l1_only']['centipawn_mae_vs_fp32']
    int16_l1_err = accuracy_results['int16_l1_only']['centipawn_mae_vs_fp32']
    int8_full_err = accuracy_results['int8_full']['centipawn_mae_vs_fp32']
    int16_full_err = accuracy_results['int16_full']['centipawn_mae_vs_fp32']

    print("\n1. ACCURACY ASSESSMENT:")
    print("-" * 40)

    for label, err in [("INT16 L1-only", int16_l1_err), ("INT8 L1-only", int8_l1_err)]:
        if err < 1.0:
            print(f"   [OK] {label}: EXCELLENT (<1 cp avg error)")
        elif err < 5.0:
            print(f"   [OK] {label}: GOOD ({err:.1f} cp avg error)")
        elif err < 15.0:
            print(f"   [..] {label}: ACCEPTABLE ({err:.1f} cp avg error)")
        else:
            print(f"   [XX] {label}: HIGH ERROR ({err:.1f} cp avg error)")

    print(f"\n   Full quantization: INT8={int8_full_err:.1f}cp, INT16={int16_full_err:.1f}cp")

    print("\n2. EXPECTED REAL-WORLD SPEEDUPS:")
    print("-" * 40)
    print("   +-------------------+--------------+--------------+")
    print("   | Hardware          | INT8 Speedup | INT16 Speedup|")
    print("   +-------------------+--------------+--------------+")
    print("   | AVX-512 + VNNI    |    3-4x      |    1.5-2x    |")
    print("   | AVX-512 (no VNNI) |    2-3x      |    1.3-1.8x  |")
    print("   | AVX2              |    1.5-2x    |    1.2-1.5x  |")
    print("   +-------------------+--------------+--------------+")

    print("\n3. IMPLEMENTATION RECOMMENDATIONS:")
    print("-" * 40)

    if int16_l1_err < 2.0:
        print("   PRIMARY: Use INT16 quantization for L1 layer")
        print("   - Minimal accuracy loss, ~1.5-2x speedup")
    elif int8_l1_err < 10.0:
        print("   PRIMARY: Use INT8 quantization for L1 layer")
        print("   - Good accuracy/speed tradeoff, ~2-4x speedup with VNNI")
    else:
        print("   PRIMARY: Consider per-channel quantization or mixed precision")

    print("\n4. IMPLEMENTATION CHECKLIST:")
    print("-" * 40)
    print("   [ ] Quantize L1 weights during model export")
    print("   [ ] Store scales as FP32 constants")
    print("   [ ] Use SIMD intrinsics (AVX2/AVX-512) for matmul")
    print("   [ ] Accumulate in INT32 to prevent overflow")
    print("   [ ] Consider Stockfish-style fixed-point (scale=64)")
    print("   [ ] Benchmark with representative positions")


def main():
    parser = argparse.ArgumentParser(description='Analyze NNUE quantization potential')
    parser.add_argument('--model', type=str, help='Path to PyTorch model file (.pt)')
    parser.add_argument('--weights', type=str, help='Path to NumPy weights file (.npz)')
    parser.add_argument('--data-dir', type=str, help='Directory containing NNUE shard files for testing')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic weights (if no --model/--weights)')
    parser.add_argument('--positions', type=int, default=10000, help='Test positions (default: 10000)')
    parser.add_argument('--bench-iterations', type=int, default=10000, help='Benchmark iterations')
    parser.add_argument('--skip-accuracy', action='store_true', help='Skip accuracy testing')
    parser.add_argument('--skip-benchmark', action='store_true', help='Skip speed benchmarks')
    parser.add_argument('--save-weights', type=str, help='Save weights to .npz file')

    args = parser.parse_args()

    # Load weights
    if args.model:
        if not HAS_TORCH:
            print("Error: PyTorch not available. Use --weights or --synthetic instead.")
            sys.exit(1)
        print(f"Loading model from {args.model}...")
        checkpoint = torch.load(args.model, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
        print("[OK] Model loaded")
    elif args.weights:
        print(f"Loading weights from {args.weights}...")
        data = np.load(args.weights)
        weights = {name: data[name] for name in data.files}
        print("[OK] Weights loaded")
    else:
        print("Generating synthetic weights (use --model or --weights for real analysis)...")
        weights = generate_synthetic_weights()
        print("[OK] Synthetic weights generated")

    if args.save_weights:
        np.savez(args.save_weights, **weights)
        print(f"[OK] Saved to {args.save_weights}")

    # Architecture
    print(f"\nNNUE Architecture:")
    print(f"  ft: {NNUE_INPUT_SIZE} -> {NNUE_HIDDEN_SIZE}")
    print(f"  l1: {NNUE_HIDDEN_SIZE * 2} (512) -> 32")
    print(f"  l2: 32 -> 32")
    print(f"  l3: 32 -> 1")
    print(f"  Total params: {sum(w.size for w in weights.values()):,}")

    # Analysis
    print_weight_analysis(weights)
    print_quantization_error_analysis(weights)
    print_memory_analysis(weights)

    # Load or generate test positions
    accuracy_results = None
    if not args.skip_accuracy:
        if args.data_dir:
            if not HAS_SHARD_IO:
                print("\nWarning: shard_io not available (missing zstandard?). Using synthetic positions.")
                print(f"Generating {args.positions} synthetic test positions...")
                positions = generate_synthetic_positions(args.positions)
            else:
                print(f"\nLoading test positions from {args.data_dir}...")
                try:
                    positions = load_positions_from_shards(args.data_dir, args.positions)
                    print(f"[OK] Loaded {len(positions)} real positions")
                except (FileNotFoundError, Exception) as e:
                    print(f"Warning: Could not load shards: {e}")
                    print(f"Falling back to {args.positions} synthetic positions...")
                    positions = generate_synthetic_positions(args.positions)
        else:
            #print(f"\nData dir not specified, generating {args.positions} synthetic test positions...")
            print(f"\033[91m\nData dir not specified, generating {args.positions} synthetic test positions...\033[00m")
            positions = generate_synthetic_positions(args.positions)

        accuracy_results = evaluate_accuracy(weights, positions)
        print_accuracy_results(accuracy_results)

    speed_results = None
    if not args.skip_benchmark:
        speed_results = benchmark_inference_speed(weights, args.bench_iterations)
        print_speed_results(speed_results)

    if accuracy_results and speed_results:
        print_recommendations(accuracy_results, speed_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Print usage hints based on what's available
    print("\nUsage examples:")
    if HAS_TORCH:
        print("  python quantize_analysis.py --model model/nnue.pt --data-dir data/nnue")
    else:
        print("  # Install torch for --model support: pip install torch")
    if HAS_SHARD_IO:
        print("  python quantize_analysis.py --weights nnue_weights.npz --data-dir data/nnue")
    else:
        print("  # Install zstandard for shard loading: pip install zstandard")


if __name__ == "__main__":
    main()