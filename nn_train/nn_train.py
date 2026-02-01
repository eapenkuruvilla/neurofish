#!/usr/bin/env python3
"""
nn_train.py - Train NNUE or DNN models from pre-processed binary shards.

MULTIPROCESSING VERSION WITH SPARSE COO TENSOR SUPPORT:
- Uses PyTorch DataLoader with multiprocessing for true parallel data loading
- pin_memory=True for faster GPU transfer
- persistent_workers=True to avoid worker restart overhead
- Sparse COO tensors for memory-efficient NNUE and DNN training with dense gradients
- C++ batch loaders for high-performance data loading (both NNUE and DNN)

This script reads binary shard files created by prepare_data.py and trains
neural network models for chess position evaluation.

Usage:
    python nn_train.py --nn-type NNUE --data-dir data/nnue
    python nn_train.py --nn-type DNN --data-dir data/dnn --resume model/dnn.pt

Features:
    - C++ batch loaders for both NNUE and DNN (highest performance)
    - Sparse COO tensor training (recommended fallback - memory efficient with dense gradients)
    - EmbeddingBag sparse training (alternative, may have convergence issues)
    - Dense one-hot training (slow, high memory, but reliable convergence)
    - Parallel data loading with multiprocessing (not threading)
    - Prefetch queue for efficient GPU utilization
    - Shuffling across multiple shards for better training
    - 90/10 train/validation split by shard files
    - Learning rate scheduling and early stopping
    - Handles diagnostic records (marker=0xFF) with embedded FEN strings

Binary shard format (created by prepare_data.py):
    DNN Normal:     [score:int16][num_features:uint8][features:uint16[]]
    DNN Diagnostic: [0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]]
                    [fen_length:uint8][fen_bytes]

    NNUE Normal:     [score:int16][stm:uint8][num_white:uint8][white:uint16[]]
                     [num_black:uint8][black:uint16[]]
    NNUE Diagnostic: [0xFF][score:int16][stm:uint8][num_white:uint8][white:uint16[]]
                     [num_black:uint8][black:uint16[]][fen_length:uint8][fen_bytes]

Training Modes:
    CPP_LOADER (best): Uses C++ multi-threaded batch loader
        - Highest performance (2-3x fewer CPU cores needed)
        - Available for both NNUE and DNN
        - Requires libbatch_loader.so (NNUE) or libdnn_batch_loader.so (DNN)

    SPARSE_COO (recommended fallback): Uses sparse COO tensors with nn.Linear
        - Memory efficient (~2MB per batch vs 1.3GB for dense)
        - Dense gradients (good convergence)
        - Standard Adam optimizer

    EMBEDDING_BAG: Uses nn.EmbeddingBag with sparse gradients
        - Memory efficient
        - Sparse gradients (may have convergence issues)
        - Requires SparseAdam optimizer

    DENSE: Uses explicit one-hot tensors
        - Very high memory usage
        - Dense gradients (good convergence)
        - Standard Adam optimizer
"""

import argparse
import os
import sys
import random
import time
import gc
import traceback
from typing import List, Tuple, Dict, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# Import network architectures and constants from nn_inference
from nn_inference import (
    NNUE_INPUT_SIZE, DNN_INPUT_SIZE, NNUE_HIDDEN_SIZE, DNN_HIDDEN_LAYERS
)
from config import TANH_SCALE

# Training configuration
VALIDATION_SPLIT_RATIO = 0.01
BATCH_SIZE = 16384  # SF 16384
LEARNING_RATE = 8.75e-4  # SF 8.75e-4

# Training mode selection (priority: CPP_LOADER > SPARSE_COO > EMBEDDING_BAG > DENSE)
CPP_LOADER = True  # Best: C++ multi-threaded loader (requires libbatch_loader.so)
SPARSE_COO = True  # Recommended fallback: sparse COO tensors with dense gradients
EMBEDDING_BAG = False  # Alternative: EmbeddingBag with sparse gradients
# If all are False, uses dense one-hot vectors (slow, high memory)

POSITIONS_PER_EPOCH = 100_000_000  # SF 100_000_000
VALIDATION_SIZE = 1_000_000  # SF 1_000_000
EPOCHS = 600  # SF 600
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 3

# DataLoader configuration
NUM_WORKERS = 4  # Number of worker processes (true parallelism)
PREFETCH_FACTOR = 4  # Batches to prefetch per worker (total buffer = workers × prefetch_factor)

# Checkpoint configuration
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs
VAL_INTERVAL = 1  # Validate every N epochs

# Garbage collection
GC_INTERVAL = 500  # Run GC every N batches

# Maximum features per position (for padding)
MAX_FEATURES_PER_POSITION = 32  # Chess has max 30 non-king pieces


# =============================================================================
# Sparse COO Tensor Utilities
# =============================================================================

def create_sparse_coo_features(
        batch_indices: List[List[int]],
        batch_size: int,
        num_features: int,
        device: torch.device = None
) -> torch.Tensor:
    """
    Create a sparse COO tensor from a list of feature indices.

    This follows the nnue-pytorch approach:
    1. Create 2D indices tensor: [[position_indices], [feature_indices]]
    2. Values are all 1.0 (binary features)
    3. Mark tensor as coalesced (indices are sorted)

    Args:
        batch_indices: List of lists, where batch_indices[i] contains the
                       active feature indices for position i
        batch_size: Number of positions in batch
        num_features: Total number of possible features (e.g., 40960)
        device: Target device (CPU or CUDA)

    Returns:
        Sparse COO tensor of shape [batch_size, num_features]
    """
    # Count total active features
    total_active = sum(len(indices) for indices in batch_indices)

    if total_active == 0:
        # Edge case: no active features
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
    else:
        # Pre-allocate arrays for indices
        position_indices = np.empty(total_active, dtype=np.int64)
        feature_indices = np.empty(total_active, dtype=np.int64)

        offset = 0
        for pos_idx, features in enumerate(batch_indices):
            n = len(features)
            if n > 0:
                position_indices[offset:offset + n] = pos_idx
                # Sort feature indices within each position (required for coalesced)
                sorted_features = sorted(features)
                feature_indices[offset:offset + n] = sorted_features
                offset += n

        # Create indices tensor: shape [2, total_active]
        indices = torch.tensor(
            np.stack([position_indices, feature_indices], axis=0),
            dtype=torch.long
        )

        # Values are all 1.0 (binary features)
        values = torch.ones(total_active, dtype=torch.float32)

    # Create sparse COO tensor
    # is_coalesced=True tells PyTorch our indices are already sorted and unique,
    # which skips the expensive coalescing step
    sparse_tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=(batch_size, num_features),
        is_coalesced=True
    )

    if device is not None:
        sparse_tensor = sparse_tensor.to(device)

    return sparse_tensor


# =============================================================================
# Sparse COO Network Implementation (Recommended for NNUE)
# =============================================================================

class NNUENetworkSparseCOO(nn.Module):
    """
    NNUE Network using sparse COO tensor inputs.

    This uses standard nn.Linear layers which support sparse input tensors.
    The key advantage over EmbeddingBag is that gradients are DENSE, which
    typically leads to better convergence.

    Architecture (matches standard NNUE):
    - Feature transformer: [input_size] -> [hidden_size] (shared for both perspectives)
    - Concatenate white + black accumulators based on side-to-move: [hidden_size * 2]
    - Hidden layers: [hidden_size * 2] -> 32 -> 32 -> 1
    """

    def __init__(self, input_size: int = NNUE_INPUT_SIZE, hidden_size: int = NNUE_HIDDEN_SIZE):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Feature transformer - standard Linear layer (supports sparse inputs!)
        self.ft = nn.Linear(input_size, hidden_size)

        # Output layers (dense)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_features: torch.Tensor, black_features: torch.Tensor,
                stm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse COO tensor inputs.

        Args:
            white_features: Sparse COO tensor [batch_size, input_size] - white's perspective
            black_features: Sparse COO tensor [batch_size, input_size] - black's perspective
            stm: Dense tensor [batch_size, 1] - side to move (1.0=white, 0.0=black)

        Returns:
            Dense tensor [batch_size, 1] - evaluation output
        """
        # Feature transformer (nn.Linear handles sparse inputs automatically)
        w = self.ft(white_features)  # [batch, hidden_size]
        b = self.ft(black_features)  # [batch, hidden_size]

        # Clipped ReLU activation [0, 1]
        w = torch.clamp(w, 0.0, 1.0)
        b = torch.clamp(b, 0.0, 1.0)

        # Concatenate based on side to move using interpolation trick
        # stm=1 (white to move): [white_accum, black_accum]
        # stm=0 (black to move): [black_accum, white_accum]
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # Output layers with clipped ReLU
        x = torch.clamp(self.l1(accumulator), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        x = self.l3(x)

        return x

    def to_inference_model(self) -> dict:
        """Convert to state dict compatible with nn_inference.NNUENetwork."""
        return {
            'ft.weight': self.ft.weight.data,
            'ft.bias': self.ft.bias.data,
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
        }

    def load_from_inference_model(self, state_dict: dict):
        """Load weights from standard NNUENetwork state dict."""
        self.ft.weight.data = state_dict['ft.weight']
        self.ft.bias.data = state_dict['ft.bias']
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']


# =============================================================================
# EmbeddingBag Network Implementation (Alternative sparse approach)
# =============================================================================

class NNUENetworkSparse(nn.Module):
    """
    NNUE Network optimized for sparse training using EmbeddingBag.

    Uses EmbeddingBag for efficient sparse feature accumulation in the first layer.
    This avoids creating huge dense tensors (batch_size × 40960).

    Note: This approach uses SPARSE gradients which may cause convergence issues.
    Consider using NNUENetworkSparseCOO instead for better convergence.

    The architecture matches NNUENetwork from nn_inference.py:
    - Input: 40960 sparse features -> 256 hidden (via EmbeddingBag)
    - Hidden: 512 (concatenated white+black) -> 32 -> 32 -> 1
    """

    def __init__(self, input_size=NNUE_INPUT_SIZE, hidden_size=NNUE_HIDDEN_SIZE):
        super(NNUENetworkSparse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer as EmbeddingBag (sparse-efficient)
        self.ft_weight = nn.EmbeddingBag(input_size, hidden_size, mode='sum', sparse=True)
        self.ft_bias = nn.Parameter(torch.zeros(hidden_size))

        # Remaining layers (dense, same as original)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_indices, white_offsets, black_indices, black_offsets, stm):
        """
        Forward pass with sparse indices.

        Args:
            white_indices: 1D tensor of all white feature indices (concatenated)
            white_offsets: 1D tensor of offsets into white_indices for each sample
            black_indices: 1D tensor of all black feature indices (concatenated)
            black_offsets: 1D tensor of offsets into black_indices for each sample
            stm: (batch_size, 1) tensor, 1.0 for white to move, 0.0 for black

        Returns:
            (batch_size, 1) output tensor
        """
        # Compute first layer activations using EmbeddingBag
        white_hidden = self.ft_weight(white_indices, white_offsets) + self.ft_bias
        black_hidden = self.ft_weight(black_indices, black_offsets) + self.ft_bias

        # Clipped ReLU [0, 1]
        white_hidden = torch.clamp(white_hidden, 0, 1)
        black_hidden = torch.clamp(black_hidden, 0, 1)

        # Concatenate based on side to move
        stm_expanded = stm.unsqueeze(-1) if stm.dim() == 1 else stm
        hidden = torch.where(
            stm_expanded.bool(),
            torch.cat([white_hidden, black_hidden], dim=-1),
            torch.cat([black_hidden, white_hidden], dim=-1)
        )

        # Dense layers with clipped ReLU
        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return x

    def to_inference_model(self):
        """Convert to state dict compatible with nn_inference.NNUENetwork."""
        state_dict = {
            'ft.weight': self.ft_weight.weight.data.t(),  # EmbeddingBag is (num_emb, emb_dim), Linear is (out, in)
            'ft.bias': self.ft_bias.data,
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard NNUENetwork state dict."""
        self.ft_weight.weight.data = state_dict['ft.weight'].t()
        self.ft_bias.data = state_dict['ft.bias']
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']


class DNNNetworkSparse(nn.Module):
    """
    DNN Network optimized for sparse training.

    Uses EmbeddingBag for efficient sparse feature accumulation in the first layer.

    The architecture matches DNNNetwork from nn_inference.py:
    - Input: 768 sparse features -> 1024 hidden (via EmbeddingBag)
    - Hidden: 1024 -> 256 -> 32 -> 1
    """

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetworkSparse, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # First layer as EmbeddingBag (sparse-efficient)
        self.l1_weight = nn.EmbeddingBag(input_size, hidden_layers[0], mode='sum', sparse=True)
        self.l1_bias = nn.Parameter(torch.zeros(hidden_layers[0]))

        # Remaining layers (dense)
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, indices, offsets):
        """Forward pass with sparse indices."""
        x = self.l1_weight(indices, offsets) + self.l1_bias
        x = torch.clamp(x, 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x

    def to_inference_model(self):
        """Convert to state dict compatible with nn_inference.DNNNetwork."""
        state_dict = {
            'l1.weight': self.l1_weight.weight.data.t(),
            'l1.bias': self.l1_bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
            'l4.weight': self.l4.weight.data,
            'l4.bias': self.l4.bias.data,
        }
        return state_dict

    def load_from_inference_model(self, state_dict):
        """Load weights from standard DNNNetwork state dict."""
        self.l1_weight.weight.data = state_dict['l1.weight'].t()
        self.l1_bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']
        self.l4.weight.data = state_dict['l4.weight']
        self.l4.bias.data = state_dict['l4.bias']


# =============================================================================
# Dense Network Implementations (using one-hot vectors instead of EmbeddingBag)
# =============================================================================

class NNUENetworkDense(nn.Module):
    """
    NNUE Network using dense one-hot vectors instead of EmbeddingBag.

    This creates explicit one-hot dense vectors and uses standard Linear layers
    for the first layer. More memory intensive but may train differently.
    """

    def __init__(self, input_size=NNUE_INPUT_SIZE, hidden_size=NNUE_HIDDEN_SIZE):
        super(NNUENetworkDense, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer as standard Linear (for dense one-hot input)
        self.ft = nn.Linear(input_size, hidden_size)

        # Remaining layers (dense, same as original)
        self.l1 = nn.Linear(hidden_size * 2, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_onehot, black_onehot, stm):
        """Forward pass with dense one-hot vectors."""
        white_hidden = self.ft(white_onehot)
        black_hidden = self.ft(black_onehot)

        white_hidden = torch.clamp(white_hidden, 0, 1)
        black_hidden = torch.clamp(black_hidden, 0, 1)

        stm_expanded = stm.unsqueeze(-1) if stm.dim() == 1 else stm
        hidden = torch.where(
            stm_expanded.bool(),
            torch.cat([white_hidden, black_hidden], dim=-1),
            torch.cat([black_hidden, white_hidden], dim=-1)
        )

        x = torch.clamp(self.l1(hidden), 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = self.l3(x)
        return x

    def to_inference_model(self):
        """Convert to inference format."""
        return {
            'ft.weight': self.ft.weight.data,
            'ft.bias': self.ft.bias.data,
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
        }

    def load_from_inference_model(self, state_dict):
        """Load weights from standard NNUENetwork state dict."""
        self.ft.weight.data = state_dict['ft.weight']
        self.ft.bias.data = state_dict['ft.bias']
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']


class DNNNetworkDense(nn.Module):
    """DNN Network using dense one-hot vectors instead of EmbeddingBag."""

    def __init__(self, input_size=DNN_INPUT_SIZE, hidden_layers=None):
        super(DNNNetworkDense, self).__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        self.l1 = nn.Linear(input_size, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, onehot):
        """Forward pass with dense one-hot vector."""
        x = self.l1(onehot)
        x = torch.clamp(x, 0, 1)
        x = torch.clamp(self.l2(x), 0, 1)
        x = torch.clamp(self.l3(x), 0, 1)
        x = self.l4(x)
        return x

    def to_inference_model(self):
        """Convert to inference format."""
        return {
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
            'l4.weight': self.l4.weight.data,
            'l4.bias': self.l4.bias.data,
        }

    def load_from_inference_model(self, state_dict):
        """Load weights from standard DNNNetwork state dict."""
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']
        self.l4.weight.data = state_dict['l4.weight']
        self.l4.bias.data = state_dict['l4.bias']


class DNNNetworkSparseCOO(nn.Module):
    """
    DNN Network using sparse COO tensor inputs.

    This uses standard nn.Linear layers which support sparse input tensors.
    The key advantage over EmbeddingBag is that gradients are DENSE, which
    typically leads to better convergence.

    Architecture (matches standard DNN):
    - Input: [input_size] sparse -> [hidden_layers[0]] (e.g., 768 -> 1024)
    - Hidden layers: 1024 -> 256 -> 32 -> 1
    """

    def __init__(self, input_size: int = DNN_INPUT_SIZE, hidden_layers: list = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = DNN_HIDDEN_LAYERS

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # First layer - standard Linear layer (supports sparse inputs!)
        self.l1 = nn.Linear(input_size, hidden_layers[0])

        # Remaining layers (dense)
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.l4 = nn.Linear(hidden_layers[2], 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse COO tensor input.

        Args:
            features: Sparse COO tensor [batch_size, input_size]

        Returns:
            Dense tensor [batch_size, 1] - evaluation output
        """
        # nn.Linear handles sparse inputs automatically
        x = self.l1(features)  # [batch, hidden_layers[0]]

        # Clipped ReLU activation [0, 1]
        x = torch.clamp(x, 0.0, 1.0)

        # Hidden layers with clipped ReLU
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        x = torch.clamp(self.l3(x), 0.0, 1.0)
        x = self.l4(x)

        return x

    def to_inference_model(self) -> dict:
        """Convert to state dict compatible with nn_inference.DNNNetwork."""
        return {
            'l1.weight': self.l1.weight.data,
            'l1.bias': self.l1.bias.data,
            'l2.weight': self.l2.weight.data,
            'l2.bias': self.l2.bias.data,
            'l3.weight': self.l3.weight.data,
            'l3.bias': self.l3.bias.data,
            'l4.weight': self.l4.weight.data,
            'l4.bias': self.l4.bias.data,
        }

    def load_from_inference_model(self, state_dict: dict):
        """Load weights from standard DNNNetwork state dict."""
        self.l1.weight.data = state_dict['l1.weight']
        self.l1.bias.data = state_dict['l1.bias']
        self.l2.weight.data = state_dict['l2.weight']
        self.l2.bias.data = state_dict['l2.bias']
        self.l3.weight.data = state_dict['l3.weight']
        self.l3.bias.data = state_dict['l3.bias']
        self.l4.weight.data = state_dict['l4.weight']
        self.l4.bias.data = state_dict['l4.bias']


# =============================================================================
# PyTorch Dataset Implementation (Multiprocessing-friendly)
# =============================================================================

class ShardIterableDataset(IterableDataset):
    """
    An IterableDataset that streams positions from compressed binary shards.

    This dataset is designed for use with PyTorch's DataLoader with num_workers > 0.
    Each worker process gets a subset of the shards to read, enabling true parallel
    data loading that bypasses the Python GIL.
    """

    def __init__(
            self,
            shard_files: List[str],
            nn_type: str,
            max_positions: Optional[int] = None,
            shuffle: bool = True,
            seed: Optional[int] = None
    ):
        super().__init__()
        self.shard_files = shard_files
        self.nn_type = nn_type.upper()
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.seed = seed if seed is not None else random.randint(0, 2 ** 31)

    def _get_worker_shards(self) -> List[str]:
        """Get the subset of shards for the current worker."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            shards = self.shard_files.copy()
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = self.shard_files[worker_id::num_workers]

        if self.shuffle:
            worker_id = worker_info.id if worker_info else 0
            rng = random.Random(self.seed + worker_id)
            shards = shards.copy()
            rng.shuffle(shards)

        return shards

    def _read_shard_positions(self, shard_path: str) -> Iterator[Dict]:
        """Read all positions from a shard file using shard_io."""
        from nn_train.shard_io import ShardReader
        reader = ShardReader(self.nn_type)
        positions = reader.read_all_positions(shard_path, include_fen=False, skip_diagnostic=True)

        if self.shuffle:
            random.shuffle(positions)

        return iter(positions)

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over positions from assigned shards."""
        shards = self._get_worker_shards()
        positions_yielded = 0

        worker_info = torch.utils.data.get_worker_info()
        if self.max_positions and worker_info:
            per_worker_max = self.max_positions // worker_info.num_workers
        else:
            per_worker_max = self.max_positions

        for shard_path in shards:
            try:
                for position in self._read_shard_positions(shard_path):
                    yield position
                    positions_yielded += 1

                    if per_worker_max and positions_yielded >= per_worker_max:
                        return
            except Exception as e:
                print(f"Warning: Error reading shard {shard_path}: {e}")
                continue


# =============================================================================
# Collate Functions for Different Training Modes
# =============================================================================

def collate_nnue_sparse_coo(batch: List[Dict]) -> Tuple:
    """
    Collate function for NNUE batches using sparse COO tensors.

    This creates sparse COO tensors with dense gradients - recommended approach.
    """
    batch_size = len(batch)

    # Extract feature lists and other data
    white_features_list = []
    black_features_list = []
    stm = np.empty(batch_size, dtype=np.float32)
    scores = np.empty(batch_size, dtype=np.float32)

    for i, pos in enumerate(batch):
        white_features_list.append(pos['white_features'])
        black_features_list.append(pos['black_features'])
        stm[i] = pos['stm']
        scores[i] = pos['score_cp']

    # Create sparse COO tensors
    white_sparse = create_sparse_coo_features(
        white_features_list, batch_size, NNUE_INPUT_SIZE
    )
    black_sparse = create_sparse_coo_features(
        black_features_list, batch_size, NNUE_INPUT_SIZE
    )

    # Create dense tensors for stm and targets
    targets = np.tanh(scores / TANH_SCALE).astype(np.float32)
    stm_t = torch.from_numpy(stm).unsqueeze(1)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return white_sparse, black_sparse, stm_t, targets_t


def collate_dnn_sparse(batch: List[Dict]) -> Tuple:
    """Collate function for DNN sparse (EmbeddingBag) batches."""
    batch_size = len(batch)
    total_features = sum(len(pos['features']) for pos in batch)

    all_features = np.empty(total_features, dtype=np.int64)
    offsets = np.empty(batch_size, dtype=np.int64)
    scores = np.empty(batch_size, dtype=np.float32)

    offset = 0
    for i, pos in enumerate(batch):
        offsets[i] = offset
        features = pos['features']
        scores[i] = pos['score_cp']

        for j, f in enumerate(features):
            all_features[offset + j] = f
        offset += len(features)

    targets = np.tanh(scores / TANH_SCALE)

    indices_t = torch.from_numpy(all_features)
    offsets_t = torch.from_numpy(offsets)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return indices_t, offsets_t, targets_t


def collate_dnn_dense(batch: List[Dict]) -> Tuple:
    """Collate function for DNN dense (one-hot) batches."""
    batch_size = len(batch)

    onehot = torch.zeros(batch_size, DNN_INPUT_SIZE)
    scores = np.empty(batch_size, dtype=np.float32)

    for i, pos in enumerate(batch):
        features = pos['features']
        scores[i] = pos['score_cp']
        for f in features:
            onehot[i, f] = 1.0

    targets = np.tanh(scores / TANH_SCALE)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return onehot, targets_t


def collate_dnn_sparse_coo(batch: List[Dict]) -> Tuple:
    """
    Collate function for DNN batches using sparse COO tensors.

    This creates sparse COO tensors with dense gradients - recommended approach.
    """
    batch_size = len(batch)

    # Extract feature lists and scores
    features_list = []
    scores = np.empty(batch_size, dtype=np.float32)

    for i, pos in enumerate(batch):
        features_list.append(pos['features'])
        scores[i] = pos['score_cp']

    # Count total active features
    total_active = sum(len(features) for features in features_list)

    if total_active == 0:
        # Edge case: no active features
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
    else:
        # Pre-allocate arrays for indices
        position_indices = np.empty(total_active, dtype=np.int64)
        feature_indices = np.empty(total_active, dtype=np.int64)

        offset = 0
        for pos_idx, features in enumerate(features_list):
            n = len(features)
            if n > 0:
                position_indices[offset:offset + n] = pos_idx
                # Sort feature indices within each position (required for coalesced)
                sorted_features = sorted(features)
                feature_indices[offset:offset + n] = sorted_features
                offset += n

        # Create indices tensor: shape [2, total_active]
        indices = torch.tensor(
            np.stack([position_indices, feature_indices], axis=0),
            dtype=torch.long
        )

        # Values are all 1.0 (binary features)
        values = torch.ones(total_active, dtype=torch.float32)

    # Create sparse COO tensor
    features_sparse = torch.sparse_coo_tensor(
        indices,
        values,
        size=(batch_size, DNN_INPUT_SIZE),
        is_coalesced=True
    )

    # Create dense target tensor
    targets = np.tanh(scores / TANH_SCALE).astype(np.float32)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return features_sparse, targets_t


def collate_nnue_sparse(batch: List[Dict]) -> Tuple:
    """Collate function for NNUE sparse (EmbeddingBag) batches."""
    batch_size = len(batch)

    total_white = sum(len(pos['white_features']) for pos in batch)
    total_black = sum(len(pos['black_features']) for pos in batch)

    all_white = np.empty(total_white, dtype=np.int64)
    white_offsets = np.empty(batch_size, dtype=np.int64)
    all_black = np.empty(total_black, dtype=np.int64)
    black_offsets = np.empty(batch_size, dtype=np.int64)
    stm = np.empty(batch_size, dtype=np.float32)
    scores = np.empty(batch_size, dtype=np.float32)

    w_offset = 0
    b_offset = 0
    for i, pos in enumerate(batch):
        white_offsets[i] = w_offset
        black_offsets[i] = b_offset
        stm[i] = pos['stm']
        scores[i] = pos['score_cp']

        white_feat = pos['white_features']
        for j, f in enumerate(white_feat):
            all_white[w_offset + j] = f
        w_offset += len(white_feat)

        black_feat = pos['black_features']
        for j, f in enumerate(black_feat):
            all_black[b_offset + j] = f
        b_offset += len(black_feat)

    targets = np.tanh(scores / TANH_SCALE)

    white_indices = torch.from_numpy(all_white)
    white_offsets_t = torch.from_numpy(white_offsets)
    black_indices = torch.from_numpy(all_black)
    black_offsets_t = torch.from_numpy(black_offsets)
    stm_t = torch.from_numpy(stm).unsqueeze(1)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return white_indices, white_offsets_t, black_indices, black_offsets_t, stm_t, targets_t


def collate_nnue_dense(batch: List[Dict]) -> Tuple:
    """Collate function for NNUE dense (one-hot) batches."""
    batch_size = len(batch)

    white_onehot = torch.zeros(batch_size, NNUE_INPUT_SIZE)
    black_onehot = torch.zeros(batch_size, NNUE_INPUT_SIZE)
    stm = np.empty(batch_size, dtype=np.float32)
    scores = np.empty(batch_size, dtype=np.float32)

    for i, pos in enumerate(batch):
        stm[i] = pos['stm']
        scores[i] = pos['score_cp']

        for f in pos['white_features']:
            white_onehot[i, f] = 1.0
        for f in pos['black_features']:
            black_onehot[i, f] = 1.0

    targets = np.tanh(scores / TANH_SCALE)
    stm_t = torch.from_numpy(stm).unsqueeze(1)
    targets_t = torch.from_numpy(targets).unsqueeze(1)

    return white_onehot, black_onehot, stm_t, targets_t


def get_collate_fn(nn_type: str, use_sparse_coo: bool, use_embedding_bag: bool):
    """Get the appropriate collate function for the network type and mode."""
    if nn_type.upper() == "DNN":
        if use_sparse_coo:
            return collate_dnn_sparse_coo
        elif use_embedding_bag:
            return collate_dnn_sparse
        else:
            return collate_dnn_dense
    else:  # NNUE
        if use_sparse_coo:
            return collate_nnue_sparse_coo
        elif use_embedding_bag:
            return collate_nnue_sparse
        else:
            return collate_nnue_dense


def create_data_loader(
        shard_files: List[str],
        nn_type: str,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        max_positions: Optional[int] = None,
        num_workers: int = NUM_WORKERS,
        use_sparse_coo: bool = SPARSE_COO,
        use_embedding_bag: bool = EMBEDDING_BAG,
        prefetch_factor: int = PREFETCH_FACTOR,
        seed: Optional[int] = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training or validation.

    Uses multiprocessing (not threading) for true parallel data loading.
    """
    dataset = ShardIterableDataset(
        shard_files=shard_files,
        nn_type=nn_type,
        max_positions=max_positions,
        shuffle=shuffle,
        seed=seed
    )

    collate_fn = get_collate_fn(nn_type, use_sparse_coo, use_embedding_bag)

    pin_memory = device.type == 'cuda' and num_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return loader


# =============================================================================
# C++ Batch Loader (High Performance)
# =============================================================================

_cpp_loader_available = None


def is_cpp_loader_available() -> bool:
    """Check if the C++ batch loader is available."""
    global _cpp_loader_available
    if _cpp_loader_available is not None:
        return _cpp_loader_available

    try:
        from nnue_batch_loader import CppBatchLoader
        _cpp_loader_available = True
    except (ImportError, RuntimeError) as e:
        _cpp_loader_available = False

    return _cpp_loader_available


def create_cpp_data_loader(
        shard_files: List[str],
        batch_size: int,
        device: torch.device,
        num_workers: int = NUM_WORKERS,
        shuffle: bool = True,
        max_positions: Optional[int] = None,
        seed: Optional[int] = None,
        num_features: int = NNUE_INPUT_SIZE
):
    """
    Create a C++ batch loader for high-performance NNUE training.

    This is significantly faster than the Python DataLoader, especially
    on systems with limited CPU cores.
    """
    from nnue_batch_loader import CppBatchLoader

    return CppBatchLoader(
        shard_paths=shard_files,
        batch_size=batch_size,
        num_workers=num_workers,
        num_features=num_features,
        shuffle=shuffle,
        seed=seed,
        device=device,
        max_positions=max_positions,
        tanh_scale=TANH_SCALE
    )


# =============================================================================
# DNN C++ Batch Loader (High Performance)
# =============================================================================

_dnn_cpp_loader_available = None


def is_dnn_cpp_loader_available() -> bool:
    """Check if the DNN C++ batch loader is available."""
    global _dnn_cpp_loader_available
    if _dnn_cpp_loader_available is not None:
        return _dnn_cpp_loader_available

    try:
        from dnn_batch_loader import DNNCppBatchLoader
        _dnn_cpp_loader_available = True
    except (ImportError, RuntimeError) as e:
        _dnn_cpp_loader_available = False

    return _dnn_cpp_loader_available


def create_dnn_cpp_data_loader(
        shard_files: List[str],
        batch_size: int,
        device: torch.device,
        num_workers: int = NUM_WORKERS,
        shuffle: bool = True,
        max_positions: Optional[int] = None,
        seed: Optional[int] = None,
        num_features: int = DNN_INPUT_SIZE
):
    """
    Create a C++ batch loader for high-performance DNN training.

    This is significantly faster than the Python DataLoader, especially
    on systems with limited CPU cores.
    """
    from dnn_batch_loader import DNNCppBatchLoader

    return DNNCppBatchLoader(
        shard_paths=shard_files,
        batch_size=batch_size,
        num_workers=num_workers,
        num_features=num_features,
        shuffle=shuffle,
        seed=seed,
        device=device,
        max_positions=max_positions,
        tanh_scale=TANH_SCALE
    )


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Handles the training loop with validation and checkpointing."""

    def __init__(
            self,
            model: nn.Module,
            nn_type: str,
            data_dir: str,
            device: torch.device,
            batch_size: int = BATCH_SIZE,
            lr: float = LEARNING_RATE,
            num_workers: int = NUM_WORKERS,
            prefetch_factor: int = PREFETCH_FACTOR,
            use_cpp_loader: bool = CPP_LOADER,
            use_sparse_coo: bool = SPARSE_COO,
            use_embedding_bag: bool = EMBEDDING_BAG,
            val_ratio: float = VALIDATION_SPLIT_RATIO
    ):
        self.model = model.to(device)
        self.nn_type = nn_type.upper()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # Determine if C++ loader is available for the given network type
        if nn_type.upper() == "NNUE":
            self.use_cpp_loader = use_cpp_loader and is_cpp_loader_available()
        else:  # DNN
            self.use_cpp_loader = use_cpp_loader and is_dnn_cpp_loader_available()

        self.use_sparse_coo = use_sparse_coo
        self.use_embedding_bag = use_embedding_bag

        # Initial shard discovery
        self._refresh_shards()

        self.criterion = nn.MSELoss()

        # Optimizer setup based on training mode
        # C++ loader and sparse COO both use dense gradients
        if self.use_cpp_loader or use_sparse_coo:
            # Dense gradients - standard Adam works!
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.optimizer_dense = None
        elif use_embedding_bag:
            # EmbeddingBag with sparse=True needs SparseAdam for sparse params
            sparse_params = []
            dense_params = []
            for name, param in model.named_parameters():
                if 'weight' in name and hasattr(model, name.split('.')[0]):
                    module = getattr(model, name.split('.')[0])
                    if isinstance(module, nn.EmbeddingBag) and module.sparse:
                        sparse_params.append(param)
                    else:
                        dense_params.append(param)
                else:
                    dense_params.append(param)

            if sparse_params:
                self.optimizer = optim.SparseAdam(sparse_params, lr=lr)
                self.optimizer_dense = optim.Adam(dense_params, lr=lr)
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=lr)
                self.optimizer_dense = None
        else:
            # Dense mode - standard Adam
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.optimizer_dense = None

        # Scheduler on main optimizer
        main_optimizer = self.optimizer_dense if self.optimizer_dense else self.optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            main_optimizer, mode='min', factor=0.5, patience=LR_PATIENCE
        )

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

    def _refresh_shards(self):
        """Re-discover shards from data directory and split into train/val."""
        all_shards = discover_shards(self.data_dir, self.nn_type)
        self.train_shards, self.val_shards = split_shards(all_shards, val_ratio=self.val_ratio)
        return len(all_shards)

    def _move_batch_to_device(self, batch):
        """Move batch tensors to the target device."""
        if self.nn_type == "DNN":
            if self.use_sparse_coo:
                # Sparse COO batch: (features_sparse, targets)
                features_sparse, targets = batch
                return (
                    features_sparse.to(self.device),
                    targets.to(self.device, non_blocking=True)
                )
            elif self.use_embedding_bag:
                indices, offsets, targets = batch
                return (
                    indices.to(self.device, non_blocking=True),
                    offsets.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True)
                )
            else:
                onehot, targets = batch
                return (
                    onehot.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True)
                )
        else:  # NNUE
            if self.use_sparse_coo:
                # Sparse COO batch: (white_sparse, black_sparse, stm, targets)
                white_sparse, black_sparse, stm, targets = batch
                return (
                    white_sparse.to(self.device),
                    black_sparse.to(self.device),
                    stm.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True)
                )
            elif self.use_embedding_bag:
                white_indices, white_offsets, black_indices, black_offsets, stm, targets = batch
                return (
                    white_indices.to(self.device, non_blocking=True),
                    white_offsets.to(self.device, non_blocking=True),
                    black_indices.to(self.device, non_blocking=True),
                    black_offsets.to(self.device, non_blocking=True),
                    stm.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True)
                )
            else:
                white_onehot, black_onehot, stm, targets = batch
                return (
                    white_onehot.to(self.device, non_blocking=True),
                    black_onehot.to(self.device, non_blocking=True),
                    stm.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True)
                )

    def _forward_dnn(self, batch):
        """Forward pass for DNN."""
        if self.use_sparse_coo:
            # Sparse COO batch: (features_sparse, targets)
            features_sparse, targets = batch
            outputs = self.model(features_sparse)
        elif self.use_embedding_bag:
            indices, offsets, targets = batch
            outputs = self.model(indices, offsets)
        else:
            onehot, targets = batch
            outputs = self.model(onehot)
        return outputs, targets

    def _forward_nnue(self, batch):
        """Forward pass for NNUE."""
        if self.use_sparse_coo:
            white_sparse, black_sparse, stm, targets = batch
            outputs = self.model(white_sparse, black_sparse, stm)
        elif self.use_embedding_bag:
            white_indices, white_offsets, black_indices, black_offsets, stm, targets = batch
            outputs = self.model(white_indices, white_offsets, black_indices, black_offsets, stm)
        else:
            white_onehot, black_onehot, stm, targets = batch
            outputs = self.model(white_onehot, black_onehot, stm)
        return outputs, targets

    def train_epoch(self, positions_per_epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        positions_seen = 0

        expected_batches = positions_per_epoch // self.batch_size
        report_interval = max(1, expected_batches // 100)

        # Create appropriate loader
        if self.use_cpp_loader:
            if self.nn_type == "NNUE":
                loader = create_cpp_data_loader(
                    self.train_shards,
                    self.batch_size,
                    self.device,
                    num_workers=self.num_workers,
                    shuffle=True,
                    max_positions=positions_per_epoch,
                    seed=self.current_epoch,
                    num_features=NNUE_INPUT_SIZE
                )
            else:  # DNN
                loader = create_dnn_cpp_data_loader(
                    self.train_shards,
                    self.batch_size,
                    self.device,
                    num_workers=self.num_workers,
                    shuffle=True,
                    max_positions=positions_per_epoch,
                    seed=self.current_epoch,
                    num_features=DNN_INPUT_SIZE
                )
        else:
            loader = create_data_loader(
                self.train_shards,
                self.nn_type,
                self.batch_size,
                self.device,
                shuffle=True,
                max_positions=positions_per_epoch,
                num_workers=self.num_workers,
                use_sparse_coo=self.use_sparse_coo,
                use_embedding_bag=self.use_embedding_bag,
                prefetch_factor=self.prefetch_factor,
                seed=self.current_epoch
            )

        start_time = time.time()

        for batch in loader:
            # C++ loader returns tensors already on device
            if not self.use_cpp_loader:
                batch = self._move_batch_to_device(batch)

            positions_seen += self.batch_size
            if positions_seen > positions_per_epoch:
                break

            # Handle different batch formats
            if self.use_cpp_loader:
                if self.nn_type == "NNUE":
                    # NNUE C++ loader returns: (white_sparse, black_sparse, stm, targets)
                    white_sparse, black_sparse, stm, targets = batch
                    outputs = self.model(white_sparse, black_sparse, stm)
                else:  # DNN
                    # DNN C++ loader returns: (features_sparse, targets)
                    features_sparse, targets = batch
                    outputs = self.model(features_sparse)
            elif self.nn_type == "DNN":
                outputs, targets = self._forward_dnn(batch)
            else:
                outputs, targets = self._forward_nnue(batch)

            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            if self.optimizer_dense:
                self.optimizer_dense.zero_grad()

            loss.backward()

            # Gradient clipping (only for dense parameters)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.grad is not None and not p.grad.is_sparse],
                1.0
            )

            self.optimizer.step()
            if self.optimizer_dense:
                self.optimizer_dense.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % report_interval == 0:
                pct = 100 * positions_seen / positions_per_epoch
                elapsed = time.time() - start_time
                pos_per_sec = positions_seen / elapsed if elapsed > 0 else 0
                print(f"\r  [{pct:5.1f}%] Loss: {loss.item():.6f} | {pos_per_sec / 1e6:.2f}M pos/s", end='', flush=True)

            if num_batches % GC_INTERVAL == 0:
                gc.collect()

        print()

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def validate(self, max_positions: int = VALIDATION_SIZE) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        positions_seen = 0

        # Create appropriate loader
        if self.use_cpp_loader:
            if self.nn_type == "NNUE":
                loader = create_cpp_data_loader(
                    self.val_shards,
                    self.batch_size,
                    self.device,
                    num_workers=self.num_workers,
                    shuffle=False,
                    max_positions=max_positions,
                    num_features=NNUE_INPUT_SIZE
                )
            else:  # DNN
                loader = create_dnn_cpp_data_loader(
                    self.val_shards,
                    self.batch_size,
                    self.device,
                    num_workers=self.num_workers,
                    shuffle=False,
                    max_positions=max_positions,
                    num_features=DNN_INPUT_SIZE
                )
        else:
            loader = create_data_loader(
                self.val_shards,
                self.nn_type,
                self.batch_size,
                self.device,
                shuffle=False,
                max_positions=max_positions,
                num_workers=self.num_workers,
                use_sparse_coo=self.use_sparse_coo,
                use_embedding_bag=self.use_embedding_bag,
                prefetch_factor=self.prefetch_factor
            )

        with torch.no_grad():
            for batch in loader:
                # C++ loader returns tensors already on device
                if not self.use_cpp_loader:
                    batch = self._move_batch_to_device(batch)

                positions_seen += self.batch_size
                if positions_seen > max_positions:
                    break

                # Handle different batch formats
                if self.use_cpp_loader:
                    if self.nn_type == "NNUE":
                        # NNUE C++ loader returns: (white_sparse, black_sparse, stm, targets)
                        white_sparse, black_sparse, stm, targets = batch
                        outputs = self.model(white_sparse, black_sparse, stm)
                    else:  # DNN
                        # DNN C++ loader returns: (features_sparse, targets)
                        features_sparse, targets = batch
                        outputs = self.model(features_sparse)
                elif self.nn_type == "DNN":
                    outputs, targets = self._forward_dnn(batch)
                else:
                    outputs, targets = self._forward_nnue(batch)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint in inference-compatible format."""
        inference_state_dict = self.model.to_inference_model()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': inference_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_dense_state_dict': self.optimizer_dense.state_dict() if self.optimizer_dense else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_val_loss,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else float('inf'),
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else float('inf'),
            'nn_type': self.nn_type,
            'history': self.history
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if checkpoint.get('nn_type', self.nn_type) != self.nn_type:
            raise ValueError(f"Checkpoint nn_type ({checkpoint.get('nn_type')}) "
                             f"does not match current ({self.nn_type})")

        self.model.load_from_inference_model(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Could not load optimizer state, starting fresh")

        if self.optimizer_dense and checkpoint.get('optimizer_dense_state_dict'):
            try:
                self.optimizer_dense.load_state_dict(checkpoint['optimizer_dense_state_dict'])
            except:
                print("Warning: Could not load dense optimizer state, starting fresh")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Could not load scheduler state, starting fresh")

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'lr': []})

        print(f"Loaded checkpoint from epoch {self.current_epoch}, best loss: {self.best_val_loss:.6f}")

    def train(self, epochs: int, positions_per_epoch: int, checkpoint_path: str,
              early_stopping_patience: int = EARLY_STOPPING_PATIENCE) -> Dict:
        """Main training loop."""
        print(f"\nStarting training from epoch {self.current_epoch + 1}")
        print(f"  Positions per epoch: {positions_per_epoch:,}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Workers: {self.num_workers} (multiprocessing)")
        print(f"  Device: {self.device}")
        print(f"  Pin memory: {self.device.type == 'cuda'}")
        print()

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            num_shards = self._refresh_shards()
            print(f"Epoch {self.current_epoch}/{epochs} ({num_shards} shards)")

            train_loss = self.train_epoch(positions_per_epoch)
            self.history['train_loss'].append(train_loss)

            if self.current_epoch % VAL_INTERVAL == 0:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)

                self.scheduler.step(val_loss)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                main_optimizer = self.optimizer_dense if self.optimizer_dense else self.optimizer
                current_lr = main_optimizer.param_groups[0]['lr']
                self.history['lr'].append(current_lr)

                epoch_time = time.time() - epoch_start
                print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
                      f"{' *BEST*' if is_best else ''}")

                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping after {early_stopping_patience} epochs without improvement")
                    break

            if self.current_epoch % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint(checkpoint_path, is_best if 'is_best' in dir() else False)
                print(f"  Checkpoint saved to {checkpoint_path}")

            print()

            gc.collect()

        self.save_checkpoint(checkpoint_path, False)

        return self.history


# Import discover_shards from shared module
from nn_train.shard_io import discover_shards


def split_shards(shards: List[str], val_ratio: float = VALIDATION_SPLIT_RATIO) -> Tuple[List[str], List[str]]:
    """Split shards into training and validation sets."""
    num_val = max(1, int(len(shards) * val_ratio))
    num_train = len(shards) - num_val

    train_shards = shards[:num_train]
    val_shards = shards[num_train:]

    return train_shards, val_shards


def main():
    parser = argparse.ArgumentParser(
        description='Train NNUE or DNN model from binary shards (Multiprocessing Version).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train NNUE model (uses sparse COO by default - recommended)
    python nn_train.py --nn-type NNUE --data-dir data/nnue

    # Train DNN model with custom parameters
    python nn_train.py --nn-type DNN --data-dir data/dnn --batch-size 8192 --lr 0.0005

    # Resume training from checkpoint
    python nn_train.py --nn-type NNUE --data-dir data/nnue --resume model/nnue.pt

    # High-end GPU optimization (A100, H100, RTX 4090)
    python nn_train.py --nn-type NNUE --data-dir data/nnue --num-workers 8 --prefetch-factor 8

Training Modes:
    SPARSE_COO (default for NNUE): Memory efficient with dense gradients
    EMBEDDING_BAG: Memory efficient but sparse gradients (may have convergence issues)
    DENSE: High memory usage but reliable convergence

Performance Notes:
    - Uses PyTorch DataLoader with multiprocessing for true parallel data loading
    - pin_memory=True enables faster CPU->GPU transfer
    - persistent_workers=True avoids worker restart overhead

    Tuning for high-end GPUs:
    - Total batch buffer = num_workers × prefetch_factor
    - Default: 4 workers × 4 prefetch = 16 batches buffered
    - High-end: 8 workers × 8 prefetch = 64 batches buffered
    - More buffer = less chance of GPU stalling, but more memory
"""
    )

    parser.add_argument(
        '--nn-type',
        type=str,
        required=True,
        choices=['NNUE', 'DNN'],
        help='Neural network type'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing shard files (or parent directory with nnue/dnn subdirs)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to save checkpoints (default: model/{nn_type}.pt)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of epochs (default: {EPOCHS})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=LEARNING_RATE,
        help=f'Learning rate (default: {LEARNING_RATE})'
    )

    parser.add_argument(
        '--positions-per-epoch',
        type=int,
        default=POSITIONS_PER_EPOCH,
        help=f'Positions per epoch (default: {POSITIONS_PER_EPOCH:,})'
    )

    parser.add_argument(
        '--val-size',
        type=int,
        default=VALIDATION_SIZE,
        help=f'Validation set size (default: {VALIDATION_SIZE:,})'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=NUM_WORKERS,
        help=f'Number of DataLoader worker processes (default: {NUM_WORKERS})'
    )

    parser.add_argument(
        '--prefetch-factor',
        type=int,
        default=PREFETCH_FACTOR,
        help=f'Batches to prefetch per worker (default: {PREFETCH_FACTOR}). '
             f'Total buffer = num_workers × prefetch_factor. Increase for high-end GPUs.'
    )

    args = parser.parse_args()

    # Determine checkpoint path
    checkpoint_path = args.checkpoint or f"model/{args.nn_type.lower()}.pt"

    # Discover shards
    print(f"Discovering shards in {args.data_dir}...")
    shards = discover_shards(args.data_dir, args.nn_type)

    if not shards:
        print(f"Error: No shard files found in {args.data_dir}")
        print("Expected files matching: *.bin.zst")
        sys.exit(1)

    print(f"Found {len(shards)} shard files")

    # Preview train/val split
    train_shards, val_shards = split_shards(shards, val_ratio=VALIDATION_SPLIT_RATIO)
    print(f"Train shards: {len(train_shards)}")
    print(f"Validation shards: {len(val_shards)}")

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Pin memory: {device.type == 'cuda'}")

    # Determine training mode (priority: CPP_LOADER > SPARSE_COO > EMBEDDING_BAG > DENSE)
    if args.nn_type == "NNUE":
        use_cpp_loader = CPP_LOADER and is_cpp_loader_available()
    else:  # DNN
        use_cpp_loader = CPP_LOADER and is_dnn_cpp_loader_available()

    use_sparse_coo = SPARSE_COO and not use_cpp_loader
    use_embedding_bag = EMBEDDING_BAG and not use_sparse_coo and not use_cpp_loader

    # Print training mode
    if use_cpp_loader:
        print(f"Training mode: C++ batch loader (high performance)")
    elif use_sparse_coo:
        print(f"Training mode: Sparse COO tensors (dense gradients)")
    elif use_embedding_bag:
        print(f"Training mode: EmbeddingBag (sparse gradients - may have convergence issues)")
    else:
        print(f"Training mode: Dense one-hot vectors (high memory usage)")

    # Show fallback info if C++ loader requested but not available
    if CPP_LOADER:
        if args.nn_type == "NNUE" and not is_cpp_loader_available():
            print(f"Note: C++ NNUE loader not available, falling back to Python loader")
            print(f"      Build with: cd cpp_batch_loader && ./build.sh")
        elif args.nn_type == "DNN" and not is_dnn_cpp_loader_available():
            print(f"Note: C++ DNN loader not available, falling back to Python loader")
            print(f"      Build with: cd cpp_batch_loader && ./build.sh")

    # Create training model based on mode
    # C++ loader and sparse COO both use sparse COO format
    if args.nn_type == "NNUE":
        if use_cpp_loader or use_sparse_coo:
            model = NNUENetworkSparseCOO(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
        elif use_embedding_bag:
            model = NNUENetworkSparse(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
        else:
            model = NNUENetworkDense(NNUE_INPUT_SIZE, NNUE_HIDDEN_SIZE)
    else:  # DNN
        if use_cpp_loader or use_sparse_coo:
            model = DNNNetworkSparseCOO(DNN_INPUT_SIZE)
        elif use_embedding_bag:
            model = DNNNetworkSparse(DNN_INPUT_SIZE)
        else:
            model = DNNNetworkDense(DNN_INPUT_SIZE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        nn_type=args.nn_type,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_cpp_loader=use_cpp_loader,
        use_sparse_coo=use_sparse_coo,
        use_embedding_bag=use_embedding_bag,
        val_ratio=VALIDATION_SPLIT_RATIO
    )

    # Load checkpoint if resuming
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"Error: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.load_checkpoint(args.resume)

    # Train
    history = trainer.train(
        epochs=args.epochs,
        positions_per_epoch=args.positions_per_epoch,
        checkpoint_path=checkpoint_path,
        early_stopping_patience=args.early_stopping
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"\nNote: Checkpoint is saved in inference-compatible format.")
    print(f"      Use with nn_inference.py NNUENetwork/DNNNetwork directly.")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms (Windows, macOS with spawn)
    import multiprocessing

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, Exiting...\n")
        exit()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        exit()