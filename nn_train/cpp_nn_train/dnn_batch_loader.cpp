/**
 * dnn_batch_loader.cpp - High-performance C++ batch loader for DNN training
 * 
 * Compile with:
 *   g++ -O3 -march=native -shared -fPIC -o libdnn_batch_loader.so dnn_batch_loader.cpp -lzstd -lpthread -std=c++17
 * 
 * Or use CMake (recommended).
 */

#include "dnn_batch_loader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <zstd.h>

// =============================================================================
// Configuration
// =============================================================================

// Maximum features per position (for buffer sizing)
// DNN: 12 piece types × 64 squares = 768 max, but typically ~16-30 active
constexpr int MAX_FEATURES_PER_POSITION = 32;

// Read buffer size for shard files
constexpr size_t READ_BUFFER_SIZE = 4 * 1024 * 1024;  // 4 MB

// Decompression buffer size
constexpr size_t DECOMPRESS_BUFFER_SIZE = 16 * 1024 * 1024;  // 16 MB

// =============================================================================
// Position structure (internal)
// =============================================================================

struct DNNPosition {
    int16_t score_cp;
    std::vector<uint16_t> features;
};

// =============================================================================
// Thread-safe queue
// =============================================================================

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size), finished_(false) {}
    
    // Push item, blocks if queue is full
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || finished_; });
        
        if (finished_) return false;
        
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }
    
    // Pop item, blocks if queue is empty
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || finished_; });
        
        if (queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    // Try to pop without blocking
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    void set_finished() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
        finished_ = false;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool is_finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_ && queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool finished_;
};

// =============================================================================
// Batch with owned memory
// =============================================================================

struct DNNOwnedBatch {
    DNNSparseBatch batch;
    
    // Owned storage
    std::vector<int64_t> position_indices;
    std::vector<int64_t> feature_indices;
    std::vector<float> scores;
    
    DNNOwnedBatch() {
        memset(&batch, 0, sizeof(batch));
    }
    
    void resize(int32_t batch_size, int32_t max_features) {
        position_indices.resize(max_features);
        feature_indices.resize(max_features);
        scores.resize(batch_size);
        
        // Update pointers
        batch.position_indices = position_indices.data();
        batch.feature_indices = feature_indices.data();
        batch.scores = scores.data();
    }
};

// =============================================================================
// Shard reader for DNN format
// =============================================================================

class DNNShardReader {
public:
    DNNShardReader(int32_t num_features) : dctx_(ZSTD_createDCtx()), num_features_(num_features) {}
    
    ~DNNShardReader() {
        if (dctx_) ZSTD_freeDCtx(dctx_);
    }
    
    // Read all positions from a shard file
    bool read_shard(const std::string& path, std::vector<DNNPosition>& positions) {
        positions.clear();
        
        // Read compressed file
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            error_ = "Failed to open file: " + path;
            return false;
        }
        
        size_t compressed_size = file.tellg();
        file.seekg(0);
        
        compressed_buffer_.resize(compressed_size);
        file.read(reinterpret_cast<char*>(compressed_buffer_.data()), compressed_size);
        
        if (!file) {
            error_ = "Failed to read file: " + path;
            return false;
        }
        
        // Get decompressed size
        unsigned long long decompressed_size = ZSTD_getFrameContentSize(
            compressed_buffer_.data(), compressed_size);
        
        if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
            error_ = "Invalid zstd file: " + path;
            return false;
        }
        
        if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            // Use streaming decompression for unknown size
            decompressed_buffer_.resize(DECOMPRESS_BUFFER_SIZE);
        } else {
            decompressed_buffer_.resize(decompressed_size);
        }
        
        // Decompress
        size_t result = ZSTD_decompressDCtx(
            dctx_,
            decompressed_buffer_.data(), decompressed_buffer_.size(),
            compressed_buffer_.data(), compressed_size
        );
        
        if (ZSTD_isError(result)) {
            error_ = "Decompression failed: " + std::string(ZSTD_getErrorName(result));
            return false;
        }
        
        // Parse positions
        return parse_dnn_positions(decompressed_buffer_.data(), result, positions);
    }
    
    const std::string& get_error() const { return error_; }

private:
    bool parse_dnn_positions(const uint8_t* data, size_t size, std::vector<DNNPosition>& positions) {
        size_t offset = 0;
        
        while (offset < size) {
            // Read first byte to check for diagnostic marker
            uint8_t first_byte = data[offset];

            if (first_byte == 0xFF) {
                // Might be diagnostic record - try to parse and skip it
                size_t saved_offset = offset;
                offset++;  // skip marker

                if (offset + 2 > size) break;

                // Read score
                int16_t score = static_cast<int16_t>(data[offset] | (data[offset + 1] << 8));
                offset += 2;

                if (offset >= size) break;
                uint8_t stm = data[offset++];

                // Validate STM (must be 0 or 1)
                if (stm > 1) {
                    // Not a valid diagnostic record - revert and read as normal
                    offset = saved_offset;
                    // Fall through to normal record parsing below
                } else {
                    // Continue trying to parse as diagnostic
                    if (offset >= size) break;
                    uint8_t num_features = data[offset++];

                    // Validate: max 32 pieces on board
                    if (num_features > 32) {
                        offset = saved_offset;
                    } else {
                        // Skip features
                        bool valid = true;
                        for (int i = 0; i < num_features && valid; i++) {
                            if (offset + 2 > size) { valid = false; break; }
                            uint16_t feat = data[offset] | (data[offset + 1] << 8);
                            if (feat >= static_cast<uint16_t>(num_features_)) valid = false;
                            offset += 2;
                        }

                        if (!valid) {
                            offset = saved_offset;
                        } else {
                            // Skip FEN
                            if (offset >= size) break;
                            uint8_t fen_len = data[offset++];
                            if (fen_len < 15 || fen_len > 100) {
                                offset = saved_offset;
                            } else {
                                offset += fen_len;
                                // Successfully skipped diagnostic record
                                continue;
                            }
                        }
                    }
                }

                // If we get here, it wasn't a valid diagnostic record
                // Reset and parse as normal record (0xFF is part of score)
                offset = saved_offset;
            }

            // Normal DNN record: [score:int16][num_features:uint8][features:uint16[]]
            if (offset + 3 > size) break;

            DNNPosition pos;

            // Score (int16, little-endian)
            pos.score_cp = static_cast<int16_t>(data[offset] | (data[offset + 1] << 8));
            offset += 2;

            // Number of features (uint8)
            uint8_t num_features = data[offset++];
            if (num_features > 32) {
                // Invalid, skip this record
                error_ = "Invalid num_features: " + std::to_string(num_features);
                continue;
            }
            pos.features.reserve(num_features);

            // Read features
            for (int i = 0; i < num_features; i++) {
                if (offset + 2 > size) {
                    error_ = "Unexpected end of data reading features";
                    return true;  // Return what we have so far
                }
                uint16_t feature = data[offset] | (data[offset + 1] << 8);
                if (feature >= static_cast<uint16_t>(num_features_)) {
                    error_ = "Invalid feature index: " + std::to_string(feature) + 
                             " (max: " + std::to_string(num_features_ - 1) + ")";
                    return true;  // Return what we have so far
                }
                pos.features.push_back(feature);
                offset += 2;
            }

            positions.push_back(std::move(pos));
        }

        return true;
    }

    ZSTD_DCtx* dctx_;
    int32_t num_features_;
    std::vector<uint8_t> compressed_buffer_;
    std::vector<uint8_t> decompressed_buffer_;
    std::string error_;
};

// =============================================================================
// Batch Loader Implementation
// =============================================================================

class DNNBatchLoader {
public:
    DNNBatchLoader(
        const std::vector<std::string>& shard_paths,
        int32_t batch_size,
        int32_t num_workers,
        int32_t queue_size,
        int32_t num_features,
        bool shuffle,
        uint64_t seed
    ) : shard_paths_(shard_paths),
        batch_size_(batch_size),
        num_workers_(num_workers),
        num_features_(num_features),
        shuffle_(shuffle),
        seed_(seed),
        batch_queue_(queue_size),
        positions_processed_(0),
        batches_produced_(0),
        running_(false)
    {
        // Pre-allocate current batch
        int max_features = batch_size * MAX_FEATURES_PER_POSITION;
        current_batch_.resize(batch_size, max_features);
    }

    ~DNNBatchLoader() {
        stop();
    }

    void start() {
        if (running_) return;

        running_ = true;
        positions_processed_ = 0;
        batches_produced_ = 0;

        // Prepare shard order
        shard_order_.clear();
        for (size_t i = 0; i < shard_paths_.size(); i++) {
            shard_order_.push_back(i);
        }

        if (shuffle_) {
            std::mt19937_64 rng(seed_);
            std::shuffle(shard_order_.begin(), shard_order_.end(), rng);
        }

        // Distribute shards among workers
        next_shard_idx_ = 0;

        // Initialize active workers counter
        active_workers_ = num_workers_;

        // Start workers
        workers_.clear();
        for (int i = 0; i < num_workers_; i++) {
            workers_.emplace_back(&DNNBatchLoader::worker_thread, this, i);
        }
    }

    void stop() {
        if (!running_) return;

        running_ = false;
        batch_queue_.set_finished();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    DNNSparseBatch* get_batch() {
        std::unique_ptr<DNNOwnedBatch> batch;
        if (!batch_queue_.pop(batch)) {
            return nullptr;
        }

        // Transfer to current_batch_ for stable pointer
        current_batch_ = std::move(*batch);
        return &current_batch_.batch;
    }

    void reset(uint64_t new_seed) {
        stop();
        batch_queue_.reset();
        seed_ = new_seed;
        start();
    }

    int64_t positions_processed() const { return positions_processed_.load(); }
    int64_t batches_produced() const { return batches_produced_.load(); }
    bool is_finished() const { return !running_ && batch_queue_.is_finished(); }
    const std::string& get_error() const { return error_; }

private:
    void worker_thread(int worker_id) {
        DNNShardReader reader(num_features_);
        std::vector<DNNPosition> positions;
        std::vector<DNNPosition> batch_positions;
        batch_positions.reserve(batch_size_);

        std::mt19937_64 rng(seed_ + worker_id);

        while (running_) {
            // Get next shard
            size_t shard_idx;
            {
                std::lock_guard<std::mutex> lock(shard_mutex_);
                if (next_shard_idx_ >= shard_order_.size()) {
                    break;  // No more shards
                }
                shard_idx = shard_order_[next_shard_idx_++];
            }

            // Read shard
            if (!reader.read_shard(shard_paths_[shard_idx], positions)) {
                std::cerr << "Warning: " << reader.get_error() << std::endl;
                continue;
            }

            // Shuffle positions within shard
            if (shuffle_) {
                std::shuffle(positions.begin(), positions.end(), rng);
            }

            // Build batches
            for (auto& pos : positions) {
                batch_positions.push_back(std::move(pos));

                if (batch_positions.size() >= static_cast<size_t>(batch_size_)) {
                    auto batch = build_batch(batch_positions);
                    if (!batch_queue_.push(std::move(batch))) {
                        return;  // Queue closed
                    }
                    batches_produced_++;
                    positions_processed_ += batch_size_;
                    batch_positions.clear();
                }
            }
        }

        // Handle remaining positions (partial batch)
        if (!batch_positions.empty() && running_) {
            auto batch = build_batch(batch_positions);
            batch_queue_.push(std::move(batch));
            batches_produced_++;
            positions_processed_ += batch_positions.size();
        }

        // Check if this is the last worker to finish
        int remaining = --active_workers_;
        if (remaining == 0) {
            batch_queue_.set_finished();
        }
    }

    std::unique_ptr<DNNOwnedBatch> build_batch(const std::vector<DNNPosition>& positions) {
        auto batch = std::make_unique<DNNOwnedBatch>();

        int32_t batch_size = positions.size();

        // Count total features
        int32_t total_features = 0;
        for (const auto& pos : positions) {
            total_features += pos.features.size();
        }

        // Resize buffers
        batch->resize(batch_size, total_features);

        // Fill in data
        int32_t feature_offset = 0;

        for (int32_t i = 0; i < batch_size; i++) {
            const auto& pos = positions[i];

            batch->scores[i] = static_cast<float>(pos.score_cp);

            // Features (sorted for coalesced tensor)
            std::vector<uint16_t> sorted_features = pos.features;
            std::sort(sorted_features.begin(), sorted_features.end());

            for (uint16_t feat : sorted_features) {
                batch->position_indices[feature_offset] = i;
                batch->feature_indices[feature_offset] = feat;
                feature_offset++;
            }
        }

        // Set batch metadata
        batch->batch.batch_size = batch_size;
        batch->batch.num_features = total_features;

        return batch;
    }

    // Configuration
    std::vector<std::string> shard_paths_;
    int32_t batch_size_;
    int32_t num_workers_;
    int32_t num_features_;
    bool shuffle_;
    uint64_t seed_;

    // Shard distribution
    std::vector<size_t> shard_order_;
    size_t next_shard_idx_ = 0;
    std::mutex shard_mutex_;

    // Workers
    std::vector<std::thread> workers_;
    std::atomic<int> active_workers_{0};
    std::atomic<bool> running_{false};

    // Batch queue
    ThreadSafeQueue<std::unique_ptr<DNNOwnedBatch>> batch_queue_;

    // Current batch (for stable pointer return)
    DNNOwnedBatch current_batch_;

    // Statistics
    std::atomic<int64_t> positions_processed_;
    std::atomic<int64_t> batches_produced_;

    // Error handling
    std::string error_;
};

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

DNNBatchLoaderHandle dnn_batch_loader_create(
    const char** shard_paths,
    int32_t num_shards,
    int32_t batch_size,
    int32_t num_workers,
    int32_t queue_size,
    int32_t num_features,
    int32_t shuffle,
    uint64_t seed
) {
    try {
        std::vector<std::string> paths;
        paths.reserve(num_shards);
        for (int i = 0; i < num_shards; i++) {
            paths.push_back(shard_paths[i]);
        }

        auto loader = new DNNBatchLoader(
            paths, batch_size, num_workers, queue_size,
            num_features, shuffle != 0, seed
        );

        return static_cast<DNNBatchLoaderHandle>(loader);
    } catch (const std::exception& e) {
        std::cerr << "dnn_batch_loader_create failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void dnn_batch_loader_start(DNNBatchLoaderHandle handle) {
    if (!handle) return;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    loader->start();
}

DNNSparseBatch* dnn_batch_loader_get_batch(DNNBatchLoaderHandle handle) {
    if (!handle) return nullptr;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    return loader->get_batch();
}

int64_t dnn_batch_loader_batches_produced(DNNBatchLoaderHandle handle) {
    if (!handle) return 0;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    return loader->batches_produced();
}

int64_t dnn_batch_loader_positions_processed(DNNBatchLoaderHandle handle) {
    if (!handle) return 0;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    return loader->positions_processed();
}

int32_t dnn_batch_loader_is_finished(DNNBatchLoaderHandle handle) {
    if (!handle) return 1;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    return loader->is_finished() ? 1 : 0;
}

void dnn_batch_loader_reset(DNNBatchLoaderHandle handle, uint64_t new_seed) {
    if (!handle) return;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    loader->reset(new_seed);
}

void dnn_batch_loader_destroy(DNNBatchLoaderHandle handle) {
    if (!handle) return;
    auto loader = static_cast<DNNBatchLoader*>(handle);
    delete loader;
}

const char* dnn_batch_loader_get_error(DNNBatchLoaderHandle handle) {
    if (!handle) return "";
    auto loader = static_cast<DNNBatchLoader*>(handle);
    return loader->get_error().c_str();
}

}  // extern "C"
