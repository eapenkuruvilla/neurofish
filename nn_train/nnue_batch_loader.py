"""
nnue_batch_loader.py - Python wrapper for the C++ batch loader

This provides a high-performance data loading pipeline for NNUE training
by using a multi-threaded C++ backend.

Usage:
    from batch_loader import CppBatchLoader
    
    loader = CppBatchLoader(
        shard_paths=shard_files,
        batch_size=16384,
        num_workers=4,
        num_features=40960,
        shuffle=True
    )
    
    for white_sparse, black_sparse, stm, scores in loader:
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
    """Find the batch_loader shared library."""
    # Search paths in order of preference
    search_paths = [
        # Same directory as this file
        Path(__file__).parent / "libnnue_batch_loader.so",
        Path(__file__).parent / "build" / "libnnue_batch_loader.so",
        # Current working directory
        Path.cwd() / "libnnue_batch_loader.so",
        Path.cwd() / "build" / "libnnue_batch_loader.so",
        Path.cwd() / "cpp_batch_loader" / "build" / "libnnue_batch_loader.so",
        # System paths
        Path("/usr/local/lib/libnnue_batch_loader.so"),
        Path("/usr/lib/libnnue_batch_loader.so"),
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
    
    raise RuntimeError(
        "Could not find libnnue_batch_loader.so. Please build it first:\n"
        "  cd cpp_batch_loader && mkdir build && cd build\n"
        "  cmake -DCMAKE_BUILD_TYPE=Release ..\n"
        "  make -j$(nproc)"
    )

# =============================================================================
# C Structure Definitions
# =============================================================================

class SparseBatch(ctypes.Structure):
    """Mirror of the C SparseBatch structure."""
    _fields_ = [
        ("batch_size", ctypes.c_int32),
        ("num_white_features", ctypes.c_int32),
        ("white_position_indices", ctypes.POINTER(ctypes.c_int64)),
        ("white_feature_indices", ctypes.POINTER(ctypes.c_int64)),
        ("num_black_features", ctypes.c_int32),
        ("black_position_indices", ctypes.POINTER(ctypes.c_int64)),
        ("black_feature_indices", ctypes.POINTER(ctypes.c_int64)),
        ("stm", ctypes.POINTER(ctypes.c_float)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
    ]

# =============================================================================
# Library Interface
# =============================================================================

class _BatchLoaderLib:
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
        
        # batch_loader_create
        self.lib.batch_loader_create.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),  # shard_paths
            ctypes.c_int32,                    # num_shards
            ctypes.c_int32,                    # batch_size
            ctypes.c_int32,                    # num_workers
            ctypes.c_int32,                    # queue_size
            ctypes.c_int32,                    # num_features
            ctypes.c_int32,                    # shuffle
            ctypes.c_uint64,                   # seed
        ]
        self.lib.batch_loader_create.restype = ctypes.c_void_p
        
        # batch_loader_start
        self.lib.batch_loader_start.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_start.restype = None
        
        # batch_loader_get_batch
        self.lib.batch_loader_get_batch.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_get_batch.restype = ctypes.POINTER(SparseBatch)
        
        # batch_loader_batches_produced
        self.lib.batch_loader_batches_produced.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_batches_produced.restype = ctypes.c_int64
        
        # batch_loader_positions_processed
        self.lib.batch_loader_positions_processed.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_positions_processed.restype = ctypes.c_int64
        
        # batch_loader_is_finished
        self.lib.batch_loader_is_finished.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_is_finished.restype = ctypes.c_int32
        
        # batch_loader_reset
        self.lib.batch_loader_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.batch_loader_reset.restype = None
        
        # batch_loader_destroy
        self.lib.batch_loader_destroy.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_destroy.restype = None
        
        # batch_loader_get_error
        self.lib.batch_loader_get_error.argtypes = [ctypes.c_void_p]
        self.lib.batch_loader_get_error.restype = ctypes.c_char_p

# =============================================================================
# Python Batch Loader
# =============================================================================

class CppBatchLoader:
    """
    High-performance batch loader using C++ backend.
    
    This class provides an iterator interface that yields batches as
    PyTorch sparse COO tensors, ready for training.
    
    Args:
        shard_paths: List of paths to .bin.zst shard files
        batch_size: Number of positions per batch
        num_workers: Number of C++ worker threads (default: 4)
        queue_size: Number of batches to buffer (default: num_workers * 2)
        num_features: Number of input features (default: 40960 for HalfKP)
        shuffle: Whether to shuffle shards and positions (default: True)
        seed: Random seed for shuffling (default: random)
        device: Target device for tensors (default: CPU)
        max_positions: Maximum positions to process (default: unlimited)
        tanh_scale: Scale factor for tanh target transformation (default: 400.0)
    
    Example:
        loader = CppBatchLoader(
            shard_paths=glob.glob("data/nnue/*.bin.zst"),
            batch_size=16384,
            num_workers=4,
            device=torch.device("cuda")
        )
        
        for white, black, stm, targets in loader:
            outputs = model(white, black, stm)
            loss = criterion(outputs, targets)
            ...
    """
    
    def __init__(
        self,
        shard_paths: List[str],
        batch_size: int,
        num_workers: int = 4,
        queue_size: Optional[int] = None,
        num_features: int = 40960,
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
        self.seed = seed if seed is not None else np.random.randint(0, 2**32)
        self.device = device or torch.device("cpu")
        self.max_positions = max_positions
        self.tanh_scale = tanh_scale
        
        self._lib = _BatchLoaderLib()
        self._handle = None
        self._positions_yielded = 0
        
    def _create_loader(self):
        """Create the C++ loader instance."""
        # Convert paths to C strings
        paths_array = (ctypes.c_char_p * len(self.shard_paths))()
        for i, path in enumerate(self.shard_paths):
            paths_array[i] = path.encode('utf-8')
        
        self._handle = self._lib.lib.batch_loader_create(
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
            raise RuntimeError("Failed to create batch loader")
        
        self._lib.lib.batch_loader_start(self._handle)
    
    def _destroy_loader(self):
        """Destroy the C++ loader instance."""
        if self._handle:
            self._lib.lib.batch_loader_destroy(self._handle)
            self._handle = None
    
    def _batch_to_tensors(self, batch: SparseBatch) -> Tuple[torch.Tensor, ...]:
        """Convert C batch to PyTorch tensors."""
        batch_size = batch.batch_size
        num_white = batch.num_white_features
        num_black = batch.num_black_features
        
        # Create numpy arrays from C pointers (zero-copy view)
        white_pos = np.ctypeslib.as_array(batch.white_position_indices, shape=(num_white,))
        white_feat = np.ctypeslib.as_array(batch.white_feature_indices, shape=(num_white,))
        black_pos = np.ctypeslib.as_array(batch.black_position_indices, shape=(num_black,))
        black_feat = np.ctypeslib.as_array(batch.black_feature_indices, shape=(num_black,))
        stm = np.ctypeslib.as_array(batch.stm, shape=(batch_size,))
        scores = np.ctypeslib.as_array(batch.scores, shape=(batch_size,))
        
        # Build white sparse tensor
        white_indices = torch.tensor(
            np.stack([white_pos, white_feat], axis=0),
            dtype=torch.long
        )
        white_values = torch.ones(num_white, dtype=torch.float32)
        white_sparse = torch.sparse_coo_tensor(
            white_indices,
            white_values,
            size=(batch_size, self.num_features),
            is_coalesced=True
        )
        
        # Build black sparse tensor
        black_indices = torch.tensor(
            np.stack([black_pos, black_feat], axis=0),
            dtype=torch.long
        )
        black_values = torch.ones(num_black, dtype=torch.float32)
        black_sparse = torch.sparse_coo_tensor(
            black_indices,
            black_values,
            size=(batch_size, self.num_features),
            is_coalesced=True
        )
        
        # Dense tensors
        stm_tensor = torch.from_numpy(stm.copy()).unsqueeze(1)
        
        # Transform scores to targets using tanh
        targets = np.tanh(scores / self.tanh_scale).astype(np.float32)
        targets_tensor = torch.from_numpy(targets).unsqueeze(1)
        
        # Move to device
        if self.device.type != "cpu":
            white_sparse = white_sparse.to(self.device)
            black_sparse = black_sparse.to(self.device)
            stm_tensor = stm_tensor.to(self.device, non_blocking=True)
            targets_tensor = targets_tensor.to(self.device, non_blocking=True)
        
        return white_sparse, black_sparse, stm_tensor, targets_tensor
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Iterate over batches."""
        self._create_loader()
        self._positions_yielded = 0
        
        try:
            while True:
                # Check position limit
                if self.max_positions and self._positions_yielded >= self.max_positions:
                    break
                
                # Get next batch from C++
                batch_ptr = self._lib.lib.batch_loader_get_batch(self._handle)
                
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
            return self._lib.lib.batch_loader_positions_processed(self._handle)
        return self._positions_yielded
    
    @property
    def batches_produced(self) -> int:
        """Get number of batches produced so far."""
        if self._handle:
            return self._lib.lib.batch_loader_batches_produced(self._handle)
        return 0
    
    def reset(self, new_seed: Optional[int] = None):
        """Reset the loader for a new epoch."""
        if new_seed is not None:
            self.seed = new_seed
        else:
            self.seed = np.random.randint(0, 2**32)
        
        if self._handle:
            self._lib.lib.batch_loader_reset(self._handle, self.seed)
            self._positions_yielded = 0


# =============================================================================
# Integration with existing Trainer
# =============================================================================

def create_cpp_data_loader(
    shard_files: List[str],
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    shuffle: bool = True,
    max_positions: Optional[int] = None,
    num_features: int = 40960,
    seed: Optional[int] = None,
    tanh_scale: float = 400.0,
    **kwargs  # Ignore extra arguments for compatibility
) -> CppBatchLoader:
    """
    Factory function to create a C++ batch loader.
    
    This is a drop-in replacement for create_data_loader() in nn_train.py.
    
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
        CppBatchLoader instance
    """
    return CppBatchLoader(
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
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import glob
    import time
    
    print("=" * 60)
    print("C++ Batch Loader Test")
    print("=" * 60)
    
    # Find some test shards
    shard_patterns = [
        "data/nnue/*.bin.zst",
        "../data/nnue/*.bin.zst",
        "*.bin.zst"
    ]
    
    shard_files = []
    for pattern in shard_patterns:
        shard_files = glob.glob(pattern)
        if shard_files:
            break
    
    if not shard_files:
        print("No shard files found. Please provide path to .bin.zst files.")
        print("Usage: python nnue_batch_loader.py [shard_pattern]")
        exit(1)
    
    print(f"Found {len(shard_files)} shard files")
    print(f"First shard: {shard_files[0]}")
    
    # Test the loader
    loader = CppBatchLoader(
        shard_paths=shard_files[:10],  # Use first 10 shards for testing
        batch_size=16384,
        num_workers=4,
        num_features=40960,
        shuffle=True,
        max_positions=1_000_000  # Limit for testing
    )
    
    print(f"\nLoading up to 1M positions...")
    start_time = time.time()
    
    num_batches = 0
    total_positions = 0
    
    for white, black, stm, targets in loader:
        num_batches += 1
        total_positions += white.shape[0]
        
        if num_batches == 1:
            print(f"\nFirst batch:")
            print(f"  White sparse: {white.shape}, nnz={white._nnz()}")
            print(f"  Black sparse: {black.shape}, nnz={black._nnz()}")
            print(f"  STM: {stm.shape}")
            print(f"  Targets: {targets.shape}, range=[{targets.min():.3f}, {targets.max():.3f}]")
        
        if num_batches % 10 == 0:
            elapsed = time.time() - start_time
            pos_per_sec = total_positions / elapsed
            print(f"\r  Batch {num_batches}: {total_positions:,} positions, "
                  f"{pos_per_sec/1e6:.2f}M pos/sec", end="", flush=True)
    
    elapsed = time.time() - start_time
    pos_per_sec = total_positions / elapsed
    
    print(f"\n\nResults:")
    print(f"  Total batches: {num_batches}")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {pos_per_sec/1e6:.2f}M positions/sec")
    print(f"\n✓ Test passed!")
