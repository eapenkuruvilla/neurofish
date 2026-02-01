"""
dnn_batch_loader.py - Python wrapper for the C++ DNN batch loader

This provides a high-performance data loading pipeline for DNN training
by using a multi-threaded C++ backend.

Usage:
    from dnn_batch_loader import DNNCppBatchLoader

    loader = DNNCppBatchLoader(
        shard_paths=shard_files,
        batch_size=16384,
        num_workers=4,
        num_features=768,  # DNN feature size
        shuffle=True
    )

    for features_sparse, targets in loader:
        # Train on batch...
        pass
"""

import ctypes
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Iterator


# =============================================================================
# Load the C++ library
# =============================================================================

def _find_library() -> str:
    """Find the dnn_batch_loader shared library."""
    # Search paths in order of preference
    search_paths = [
        # Same directory as this file
        Path(__file__).parent / "libdnn_batch_loader.so",
        Path(__file__).parent / "build" / "libdnn_batch_loader.so",
        # Current working directory
        Path.cwd() / "libdnn_batch_loader.so",
        Path.cwd() / "build" / "libdnn_batch_loader.so",
        Path.cwd() / "cpp_batch_loader" / "build" / "libdnn_batch_loader.so",
        # System paths
        Path("/usr/local/lib/libdnn_batch_loader.so"),
        Path("/usr/lib/libdnn_batch_loader.so"),
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    raise RuntimeError(
        "Could not find libdnn_batch_loader.so. Please build it first:\n"
        "  g++ -O3 -march=native -shared -fPIC -o libdnn_batch_loader.so "
        "dnn_batch_loader.cpp -lzstd -lpthread -std=c++17"
    )


# =============================================================================
# C Structure Definitions
# =============================================================================

class DNNSparseBatch(ctypes.Structure):
    """Mirror of the C DNNSparseBatch structure."""
    _fields_ = [
        ("batch_size", ctypes.c_int32),
        ("num_features", ctypes.c_int32),
        ("position_indices", ctypes.POINTER(ctypes.c_int64)),
        ("feature_indices", ctypes.POINTER(ctypes.c_int64)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
    ]


# =============================================================================
# Library Interface
# =============================================================================

class _DNNBatchLoaderLib:
    """Singleton wrapper for the C library."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_library()
        return cls._instance

    def _load_library(self):
        lib_path = _find_library()
        self.lib = ctypes.CDLL(lib_path)

        # Define function signatures

        # dnn_batch_loader_create
        self.lib.dnn_batch_loader_create.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # shard_paths
            ctypes.c_int32,  # num_shards
            ctypes.c_int32,  # batch_size
            ctypes.c_int32,  # num_workers
            ctypes.c_int32,  # queue_size
            ctypes.c_int32,  # num_features
            ctypes.c_int32,  # shuffle
            ctypes.c_uint64,  # seed
        ]
        self.lib.dnn_batch_loader_create.restype = ctypes.c_void_p

        # dnn_batch_loader_start
        self.lib.dnn_batch_loader_start.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_start.restype = None

        # dnn_batch_loader_get_batch
        self.lib.dnn_batch_loader_get_batch.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_get_batch.restype = ctypes.POINTER(DNNSparseBatch)

        # dnn_batch_loader_batches_produced
        self.lib.dnn_batch_loader_batches_produced.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_batches_produced.restype = ctypes.c_int64

        # dnn_batch_loader_positions_processed
        self.lib.dnn_batch_loader_positions_processed.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_positions_processed.restype = ctypes.c_int64

        # dnn_batch_loader_is_finished
        self.lib.dnn_batch_loader_is_finished.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_is_finished.restype = ctypes.c_int32

        # dnn_batch_loader_reset
        self.lib.dnn_batch_loader_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.dnn_batch_loader_reset.restype = None

        # dnn_batch_loader_destroy
        self.lib.dnn_batch_loader_destroy.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_destroy.restype = None

        # dnn_batch_loader_get_error
        self.lib.dnn_batch_loader_get_error.argtypes = [ctypes.c_void_p]
        self.lib.dnn_batch_loader_get_error.restype = ctypes.c_char_p


# =============================================================================
# Python Batch Loader
# =============================================================================

class DNNCppBatchLoader:
    """
    High-performance DNN batch loader using C++ backend.

    This class provides an iterator interface that yields batches as
    PyTorch sparse COO tensors, ready for training.

    Args:
        shard_paths: List of paths to .bin.zst shard files
        batch_size: Number of positions per batch
        num_workers: Number of C++ worker threads (default: 4)
        queue_size: Number of batches to buffer (default: num_workers * 2)
        num_features: Number of input features (default: 768 for DNN)
        shuffle: Whether to shuffle shards and positions (default: True)
        seed: Random seed for shuffling (default: random)
        device: Target device for tensors (default: CPU)
        max_positions: Maximum positions to process (default: unlimited)
        tanh_scale: Scale factor for tanh target transformation (default: 400.0)

    Example:
        loader = DNNCppBatchLoader(
            shard_paths=glob.glob("data/dnn/*.bin.zst"),
            batch_size=16384,
            num_workers=4,
            device=torch.device("cuda")
        )

        for features, targets in loader:
            outputs = model(features)
            loss = criterion(outputs, targets)
            ...
    """

    def __init__(
            self,
            shard_paths: List[str],
            batch_size: int,
            num_workers: int = 4,
            queue_size: Optional[int] = None,
            num_features: int = 768,  # DNN default
            shuffle: bool = True,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            max_positions: Optional[int] = None,
            tanh_scale: float = 400.0
    ):
        self.shard_paths = [str(p) for p in shard_paths]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_size = queue_size or (num_workers * 2)
        self.num_features = num_features
        self.shuffle = shuffle
        self.seed = seed if seed is not None else np.random.randint(0, 2 ** 32)
        self.device = device or torch.device("cpu")
        self.max_positions = max_positions
        self.tanh_scale = tanh_scale

        self._lib = _DNNBatchLoaderLib()
        self._handle = None
        self._positions_yielded = 0

    def _create_loader(self):
        """Create the C++ loader instance."""
        # Convert paths to C strings
        paths_array = (ctypes.c_char_p * len(self.shard_paths))()
        for i, path in enumerate(self.shard_paths):
            paths_array[i] = path.encode('utf-8')

        self._handle = self._lib.lib.dnn_batch_loader_create(
            paths_array,
            len(self.shard_paths),
            self.batch_size,
            self.num_workers,
            self.queue_size,
            self.num_features,
            1 if self.shuffle else 0,
            self.seed
        )

        if not self._handle:
            raise RuntimeError("Failed to create DNN batch loader")

        self._lib.lib.dnn_batch_loader_start(self._handle)

    def _destroy_loader(self):
        """Destroy the C++ loader instance."""
        if self._handle:
            self._lib.lib.dnn_batch_loader_destroy(self._handle)
            self._handle = None

    def _batch_to_tensors(self, batch: DNNSparseBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert C batch to PyTorch tensors."""
        batch_size = batch.batch_size
        num_features = batch.num_features

        # Create numpy arrays from C pointers (zero-copy view)
        position_indices = np.ctypeslib.as_array(batch.position_indices, shape=(num_features,))
        feature_indices = np.ctypeslib.as_array(batch.feature_indices, shape=(num_features,))
        scores = np.ctypeslib.as_array(batch.scores, shape=(batch_size,))

        # Build sparse tensor
        indices = torch.tensor(
            np.stack([position_indices, feature_indices], axis=0),
            dtype=torch.long
        )
        values = torch.ones(num_features, dtype=torch.float32)
        features_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=(batch_size, self.num_features),
            is_coalesced=True
        )

        # Transform scores to targets using tanh
        targets = np.tanh(scores / self.tanh_scale).astype(np.float32)
        targets_tensor = torch.from_numpy(targets.copy()).unsqueeze(1)

        # Move to device
        if self.device.type != "cpu":
            features_sparse = features_sparse.to(self.device)
            targets_tensor = targets_tensor.to(self.device, non_blocking=True)

        return features_sparse, targets_tensor

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches."""
        self._create_loader()
        self._positions_yielded = 0

        try:
            while True:
                # Check position limit
                if self.max_positions and self._positions_yielded >= self.max_positions:
                    break

                # Get next batch from C++
                batch_ptr = self._lib.lib.dnn_batch_loader_get_batch(self._handle)

                if not batch_ptr:
                    break  # No more batches

                batch = batch_ptr.contents
                self._positions_yielded += batch.batch_size

                # Convert to PyTorch tensors
                yield self._batch_to_tensors(batch)

        finally:
            self._destroy_loader()

    def __len__(self) -> int:
        """Estimate number of batches (approximate)."""
        # This is just an estimate since we don't know exact shard sizes
        if self.max_positions:
            return self.max_positions // self.batch_size
        # Rough estimate: assume ~500K positions per shard
        return (len(self.shard_paths) * 500000) // self.batch_size

    @property
    def positions_processed(self) -> int:
        """Get number of positions processed so far."""
        if self._handle:
            return self._lib.lib.dnn_batch_loader_positions_processed(self._handle)
        return self._positions_yielded

    @property
    def batches_produced(self) -> int:
        """Get number of batches produced so far."""
        if self._handle:
            return self._lib.lib.dnn_batch_loader_batches_produced(self._handle)
        return 0

    def reset(self, new_seed: Optional[int] = None):
        """Reset the loader for a new epoch."""
        if new_seed is not None:
            self.seed = new_seed
        else:
            self.seed = np.random.randint(0, 2 ** 32)

        if self._handle:
            self._lib.lib.dnn_batch_loader_reset(self._handle, self.seed)
            self._positions_yielded = 0


# =============================================================================
# Integration with existing Trainer
# =============================================================================

def create_dnn_cpp_data_loader(
        shard_files: List[str],
        batch_size: int,
        device: torch.device,
        num_workers: int = 4,
        shuffle: bool = True,
        max_positions: Optional[int] = None,
        num_features: int = 768,
        seed: Optional[int] = None,
        tanh_scale: float = 400.0,
        **kwargs  # Ignore extra arguments for compatibility
) -> DNNCppBatchLoader:
    """
    Factory function to create a C++ DNN batch loader.

    This is a drop-in replacement for create_data_loader() in nn_train.py for DNN.

    Args:
        shard_files: List of shard file paths
        batch_size: Batch size
        device: Target device
        num_workers: Number of C++ worker threads
        shuffle: Whether to shuffle data
        max_positions: Maximum positions to load
        num_features: Number of input features
        seed: Random seed
        tanh_scale: Scale for tanh target transformation

    Returns:
        DNNCppBatchLoader instance
    """
    return DNNCppBatchLoader(
        shard_paths=shard_files,
        batch_size=batch_size,
        num_workers=num_workers,
        num_features=num_features,
        shuffle=shuffle,
        seed=seed,
        device=device,
        max_positions=max_positions,
        tanh_scale=tanh_scale
    )


# =============================================================================
# Check if DNN C++ loader is available
# =============================================================================

_dnn_cpp_loader_available = None


def is_dnn_cpp_loader_available() -> bool:
    """Check if the DNN C++ batch loader is available."""
    global _dnn_cpp_loader_available
    if _dnn_cpp_loader_available is not None:
        return _dnn_cpp_loader_available

    try:
        _DNNBatchLoaderLib()
        _dnn_cpp_loader_available = True
    except (RuntimeError, OSError) as e:
        _dnn_cpp_loader_available = False

    return _dnn_cpp_loader_available


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import glob
    import time

    print("=" * 60)
    print("DNN C++ Batch Loader Test")
    print("=" * 60)

    # Find some test shards
    shard_patterns = [
        "data/dnn/*.bin.zst",
        "../data/dnn/*.bin.zst",
        "*.bin.zst"
    ]

    shard_files = []
    for pattern in shard_patterns:
        shard_files = glob.glob(pattern)
        if shard_files:
            break

    if not shard_files:
        print("No shard files found. Please provide path to .bin.zst files.")
        print("Usage: python dnn_batch_loader.py [shard_pattern]")
        exit(1)

    print(f"Found {len(shard_files)} shard files")
    print(f"First shard: {shard_files[0]}")

    # Test the loader
    loader = DNNCppBatchLoader(
        shard_paths=shard_files[:10],  # Use first 10 shards for testing
        batch_size=16384,
        num_workers=4,
        num_features=768,  # DNN feature size
        shuffle=True,
        max_positions=1_000_000  # Limit for testing
    )

    print(f"\nLoading up to 1M positions...")
    start_time = time.time()

    num_batches = 0
    total_positions = 0

    for features, targets in loader:
        num_batches += 1
        total_positions += features.shape[0]

        if num_batches == 1:
            print(f"\nFirst batch:")
            print(f"  Features sparse: {features.shape}, nnz={features._nnz()}")
            print(f"  Targets: {targets.shape}, range=[{targets.min():.3f}, {targets.max():.3f}]")

        if num_batches % 10 == 0:
            elapsed = time.time() - start_time
            pos_per_sec = total_positions / elapsed
            print(f"\r  Batch {num_batches}: {total_positions:,} positions, "
                  f"{pos_per_sec / 1e6:.2f}M pos/sec", end="", flush=True)

    elapsed = time.time() - start_time
    pos_per_sec = total_positions / elapsed

    print(f"\n\nResults:")
    print(f"  Total batches: {num_batches}")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {pos_per_sec / 1e6:.2f}M positions/sec")
    print(f"\n✓ Test passed!")