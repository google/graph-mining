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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PER_WORKER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PER_WORKER_H_

#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "parlay/scheduler.h"

namespace graph_mining::in_memory {

// Stores one copy of a variable of type T per each Parlay scheduler worker.
// When using Parlay scheduler, i.e., parlay::parallel_for, this can effectively
// provide "per-thread" variables, which can be modified without locking, as
// each worker has its own variable.
// Initially, each variable is default-allocated.
// The items are stored cacheline aligned to prevent threads from interfering
// with each other. Consequently the vector of items uses at least 64 bytes per
// worker on x86. This is negligible for most use cases since one usually only
// allocates a constant number of PerWorker objects.
//
// Only one of the methods of this class, namely Get() with no arguments, is
// intended to be called inside parlay parallel fors. The other functions are
// intended to be called from serial code only.
template <class T>
class PerWorker {
 public:
  PerWorker() {
    elements_ =
        std::vector<CachelineAlignedT>(parlay::num_workers());
  }

  // Returns a reference to the respective per-worker variable. When not called
  // within a parallel context, returns the variable for worker 0.
  // This function is thread-safe.
  T& Get() { return Get(parlay::worker_id()); }

  // Returns a reference to the respective per-worker variable.
  // WARNING: This function is not thread safe. (The current implementation is
  // actually thread-safe but the intended use cases don't call this
  // multi-threaded so we don't promise to keep it thread safe in the future.)
  // Also any use of the returned reference is unsafe unless the client adds
  // their own synchronization.
  const T& Get(int worker_id) const {
    ABSL_DCHECK_GE(worker_id, 0);
    ABSL_DCHECK_LT(worker_id, elements_.size());
    return elements_[worker_id].wrapped;
  }

  // Same as above but returns a non-const reference.
  T& Get(int worker_id) {
    ABSL_DCHECK_GE(worker_id, 0);
    ABSL_DCHECK_LT(worker_id, elements_.size());
    return elements_[worker_id].wrapped;
  }

  // Returns the number of workers.
  int NumWorkers() const { return elements_.size(); }

  // Returns a copy of the vector of all per-worker variables. The length of the
  // returned vector is parlay::GetScheduler().num_workers() and the elements
  // are indexed with parlay::GetScheduler().worker_id(). WARNING: This function
  // is not thread safe.
  std::vector<T> GetCopyAll() {
    std::vector<T> result;
    result.reserve(elements_.size());
    for (const auto& element : elements_) {
      result.push_back(element.wrapped);
    }
    return result;
  }

  // Returns a copy of the vector of all per-worker variables and resets the
  // values maintained by this per-worker instance. This invalides the
  // references returned by Get(). The length of the returned vector is
  // parlay::GetScheduler().num_workers() and the elements are indexed with
  // parlay::GetScheduler().worker_id().
  // WARNING: not thread safe.
  std::vector<T> ReleaseAll() {
    std::vector<T> result;
    result.reserve(elements_.size());
    for (auto& element : elements_) {
      result.push_back(std::move(element.wrapped));
    }
    elements_ =
        std::vector<CachelineAlignedT>(parlay::num_workers());
    return result;
  }

 private:
  struct CachelineAlignedT {
    ABSL_CACHELINE_ALIGNED T wrapped;
  };

  std::vector<CachelineAlignedT> elements_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PER_WORKER_H_
