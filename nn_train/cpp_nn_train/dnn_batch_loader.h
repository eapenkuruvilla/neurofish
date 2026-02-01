/**
 * dnn_batch_loader.h - High-performance C++ batch loader for DNN training
 * 
 * This provides a multi-threaded data loading pipeline for DNN models.
 * Unlike NNUE which has separate white/black perspectives, DNN uses a
 * single feature set per position.
 * 
 * Architecture:
 * - Multiple worker threads read and decompress shards
 * - Batches are prepared in C++ with pre-allocated buffers
 * - Thread-safe queue holds ready batches for Python consumption
 * - Zero-copy transfer to PyTorch via numpy arrays
 * 
 * Binary shard format (DNN):
 *   Normal:     [score:int16][num_features:uint8][features:uint16[]]
 *   Diagnostic: [0xFF][score:int16][stm:uint8][num_features:uint8][features:uint16[]]
 *               [fen_length:uint8][fen_bytes]
 */

#ifndef DNN_BATCH_LOADER_H
#define DNN_BATCH_LOADER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sparse batch structure for DNN training.
 * 
 * This is designed to be directly convertible to PyTorch sparse COO tensors.
 * The indices are pre-sorted for coalesced tensor creation.
 */
typedef struct {
    int32_t batch_size;
    
    // Features sparse representation
    int32_t num_features;            // Total non-zero features across batch
    int64_t* position_indices;       // Which sample each feature belongs to
    int64_t* feature_indices;        // The feature index (0 to NUM_FEATURES-1)
    
    // Dense arrays (one per sample)
    float* scores;                   // Raw scores in centipawns
    
} DNNSparseBatch;

/**
 * Opaque handle to the batch loader instance.
 */
typedef void* DNNBatchLoaderHandle;

/**
 * Create a new DNN batch loader.
 * 
 * @param shard_paths Array of null-terminated shard file paths
 * @param num_shards Number of shard paths
 * @param batch_size Number of positions per batch
 * @param num_workers Number of worker threads
 * @param queue_size Maximum batches to buffer (recommended: 2-4x num_workers)
 * @param num_features Number of input features (e.g., 768 for DNN)
 * @param shuffle Whether to shuffle shards and positions
 * @param seed Random seed for shuffling
 * @return Handle to the batch loader, or NULL on failure
 */
DNNBatchLoaderHandle dnn_batch_loader_create(
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
void dnn_batch_loader_start(DNNBatchLoaderHandle handle);

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
DNNSparseBatch* dnn_batch_loader_get_batch(DNNBatchLoaderHandle handle);

/**
 * Get the number of batches produced so far.
 */
int64_t dnn_batch_loader_batches_produced(DNNBatchLoaderHandle handle);

/**
 * Get the number of positions processed so far.
 */
int64_t dnn_batch_loader_positions_processed(DNNBatchLoaderHandle handle);

/**
 * Check if all data has been consumed.
 */
int32_t dnn_batch_loader_is_finished(DNNBatchLoaderHandle handle);

/**
 * Stop all workers and reset for a new epoch.
 * 
 * @param new_seed New random seed for shuffling (ignored if shuffle=false)
 */
void dnn_batch_loader_reset(DNNBatchLoaderHandle handle, uint64_t new_seed);

/**
 * Destroy the batch loader and free all resources.
 */
void dnn_batch_loader_destroy(DNNBatchLoaderHandle handle);

/**
 * Get the last error message (if any).
 * Returns empty string if no error.
 */
const char* dnn_batch_loader_get_error(DNNBatchLoaderHandle handle);

#ifdef __cplusplus
}
#endif

#endif // DNN_BATCH_LOADER_H
