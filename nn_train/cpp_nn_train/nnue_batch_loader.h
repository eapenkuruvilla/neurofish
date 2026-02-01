/**
 * batch_loader.h - High-performance C++ batch loader for NNUE training
 * 
 * This provides a multi-threaded data loading pipeline that can saturate
 * high-end GPUs even with limited CPU cores.
 * 
 * Architecture:
 * - Multiple worker threads read and decompress shards
 * - Batches are prepared in C++ with pre-allocated buffers
 * - Thread-safe queue holds ready batches for Python consumption
 * - Zero-copy transfer to PyTorch via numpy arrays
 */

#ifndef BATCH_LOADER_H
#define BATCH_LOADER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sparse batch structure for NNUE training.
 * 
 * This is designed to be directly convertible to PyTorch sparse COO tensors.
 * The indices are pre-sorted for coalesced tensor creation.
 */
typedef struct {
    int32_t batch_size;
    
    // White features sparse representation
    int32_t num_white_features;      // Total non-zero features across batch
    int64_t* white_position_indices; // Which sample each feature belongs to
    int64_t* white_feature_indices;  // The feature index (0 to NUM_FEATURES-1)
    
    // Black features sparse representation  
    int32_t num_black_features;
    int64_t* black_position_indices;
    int64_t* black_feature_indices;
    
    // Dense arrays (one per sample)
    float* stm;                      // Side to move: 1.0 = white, 0.0 = black
    float* scores;                   // Raw scores in centipawns
    
} SparseBatch;

/**
 * Opaque handle to the batch loader instance.
 */
typedef void* BatchLoaderHandle;

/**
 * Create a new batch loader.
 * 
 * @param shard_paths Array of null-terminated shard file paths
 * @param num_shards Number of shard paths
 * @param batch_size Number of positions per batch
 * @param num_workers Number of worker threads
 * @param queue_size Maximum batches to buffer (recommended: 2-4x num_workers)
 * @param num_features Number of input features (e.g., 40960 for HalfKP)
 * @param shuffle Whether to shuffle shards and positions
 * @param seed Random seed for shuffling
 * @return Handle to the batch loader, or NULL on failure
 */
BatchLoaderHandle batch_loader_create(
    const char** shard_paths,
    int32_t num_shards,
    int32_t batch_size,
    int32_t num_workers,
    int32_t queue_size,
    int32_t num_features,
    int32_t shuffle,
    uint64_t seed
);

/**
 * Start the worker threads.
 * Call this after create() and before get_batch().
 */
void batch_loader_start(BatchLoaderHandle handle);

/**
 * Get the next batch.
 * 
 * This blocks until a batch is available or all data is exhausted.
 * 
 * @param handle Batch loader handle
 * @return Pointer to batch, or NULL if no more data
 * 
 * IMPORTANT: The returned batch is owned by the loader. Do not free it.
 * The batch remains valid until the next call to get_batch() or destroy().
 */
SparseBatch* batch_loader_get_batch(BatchLoaderHandle handle);

/**
 * Get the number of batches produced so far.
 */
int64_t batch_loader_batches_produced(BatchLoaderHandle handle);

/**
 * Get the number of positions processed so far.
 */
int64_t batch_loader_positions_processed(BatchLoaderHandle handle);

/**
 * Check if all data has been consumed.
 */
int32_t batch_loader_is_finished(BatchLoaderHandle handle);

/**
 * Stop all workers and reset for a new epoch.
 * 
 * @param new_seed New random seed for shuffling (ignored if shuffle=false)
 */
void batch_loader_reset(BatchLoaderHandle handle, uint64_t new_seed);

/**
 * Destroy the batch loader and free all resources.
 */
void batch_loader_destroy(BatchLoaderHandle handle);

/**
 * Get the last error message (if any).
 * Returns empty string if no error.
 */
const char* batch_loader_get_error(BatchLoaderHandle handle);

#ifdef __cplusplus
}
#endif

#endif // BATCH_LOADER_H
