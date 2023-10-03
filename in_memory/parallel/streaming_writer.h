// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_STREAMING_WRITER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_STREAMING_WRITER_H_

#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "in_memory/parallel/per_worker.h"

namespace graph_mining::in_memory {

// Writes a stream of elements to a collection of vectors in a (mostly)
// lock-free way. This class should only be used with Parlay scheduler (i.e.,
// parlay::parallel_for)
template <typename T>
class StreamingWriter {
 public:
  // Constructs a new writer, specifying a maximum block size (see Build() for
  // details). 0 means no limit.
  explicit StreamingWriter(int64_t block_size) {
    block_size_ =
        block_size == 0 ? std::numeric_limits<int64_t>::max() : block_size;
    ABSL_CHECK_GT(block_size_, 0);
    for (auto& buffer : unfinished_buffer_.GetAll()) {
      // Note that we reserve `block_size` instead of `block_size_`. This is
      // because if the caller explicitly specifies `block_size`, then it is
      // reasonable to believe that the caller has knowledge on how large the
      // per-thread buffer needs to be and that it should fit in memory.
      buffer.reserve(block_size);
    }
  }

  // Add a single element.
  void Add(T new_element);

  // Add a collection of elements. All these elements are guaranteed to be in a
  // single output vector. If buffer size is greater than block_size, buffer
  // becomes one of the output vectors. It is recommended to std::move the
  // argument.
  void AddBuffer(std::vector<T> buffer);

  // Returns all elements added so far. This should only be called once, after
  // all calls to Add() have completed.
  // The vectors in the result have length in range (block_size/2, block_size],
  // with the following exceptions:
  //  * there may be a vector of size > block_size, equal to an argument of some
  //    AddBuffer() call.
  //  * there may be at most one vector of size less than block_size/2 per
  //    worker.
  // Note that choosing a small/moderate block size allows the produced blocks
  // to be later processed sequentially.
  std::vector<std::vector<T>> Build();

 private:
  int64_t block_size_;

  // This is guarded by mutex_ within Add/AddBuffer, but unguarded within Build
  // (so we can't use ABSL_GUARDED_BY here).
  std::vector<std::vector<T>> finished_buffers_;

  PerWorker<std::vector<T>> unfinished_buffer_;

  absl::Mutex mutex_;
};

template <typename T>
void StreamingWriter<T>::Add(T new_element) {
  std::vector<T>& unfinished_buffer = unfinished_buffer_.Get();

  if (unfinished_buffer.size() == block_size_) {
    mutex_.Lock();
    finished_buffers_.push_back(std::move(unfinished_buffer));
    mutex_.Unlock();
    unfinished_buffer.clear();
  }
  unfinished_buffer.push_back(std::move(new_element));
}

template <typename T>
void StreamingWriter<T>::AddBuffer(std::vector<T> buffer) {
  std::vector<T>& unfinished_buffer = unfinished_buffer_.Get();
  if (unfinished_buffer.size() + buffer.size() <= block_size_) {
    unfinished_buffer.insert(unfinished_buffer.end(), buffer.begin(),
                             buffer.end());
  } else if (2 * buffer.size() > block_size_) {
    mutex_.Lock();
    finished_buffers_.push_back(std::move(buffer));
    mutex_.Unlock();
  } else {
    // We have that:
    //  unfinished_buffer.size() + buffer.size() > block_size
    //  buffer.size() <= block_size / 2
    // This gives
    //  unfinished_buffer.size() + block_size / 2 > block_size
    //  unfinished_buffer.size() > block_size / 2,
    // so unfinished_buffer can be added to finished_buffers_ by itself.
    mutex_.Lock();
    finished_buffers_.push_back(std::move(unfinished_buffer));
    mutex_.Unlock();

    unfinished_buffer = std::move(buffer);
  }
}

template <typename T>
std::vector<std::vector<T>> StreamingWriter<T>::Build() {
  for (auto& unfinished_buffer : unfinished_buffer_.GetAll()) {
    if (!unfinished_buffer.empty())
      finished_buffers_.push_back(std::move(unfinished_buffer));
  }
  return std::move(finished_buffers_);
}

// Concatenates a collection of vectors into one large sequence (preserving the
// order of elements).
template <typename T>
parlay::sequence<T> Flatten(std::vector<std::vector<T>> buffers) {
  parlay::sequence<size_t> buffer_sizes_prefix_sum(buffers.size());

  parlay::parallel_for(0, buffers.size(), [&](size_t i) {
    buffer_sizes_prefix_sum[i] = buffers[i].size();
  });

  size_t total_size = parlay::scan_inplace(buffer_sizes_prefix_sum);

  if (total_size == 0) return {};

  parlay::sequence<T> result(total_size);

  parlay::parallel_for(
      0, buffers.size(),
      [&](size_t i) {
        size_t dst = buffer_sizes_prefix_sum[i];
        parlay::parallel_for(0, buffers[i].size(), [&](size_t j) {
          result[dst + j] = buffers[i][j];
        });
      },
      /*granularity=*/1);
  return result;
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_STREAMING_WRITER_H_
