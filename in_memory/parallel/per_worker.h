// Copyright 2010-2023 Google LLC
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

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PER_WORKER_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PER_WORKER_H_

#include <vector>

#include "absl/log/absl_check.h"
#include "parlay/parallel.h"
#include "parlay/scheduler.h"

namespace graph_mining::in_memory {

// Stores one copy of a variable of type T per each Parlay scheduler worker.
// When using Parlay scheduler, i.e., parlay::parallel_for, this can effectively
// provide "per-thread" variables, which can be modified without locking, as
// each worker has its own variable.
// Initially, each variable is default-allocated.
// TODO: As pointed out by laxmand, the performance may not be great if
// sizeof(T) is less than the cache line size. In particular, each modification
// may invalidate the cache of other CPUs therefore greatly impacting
// performance.
// Note that T cannot be bool, as std::vector<bool> does not provide references
// to individual elements.
template <class T>
class PerWorker {
 public:
  PerWorker() {
    elements_ =
        std::vector<T>(parlay::num_workers());
  }

  // Return the reference to the respective per-worker variable. When not called
  // within a parallel context, returns the variable for worker 0.
  T& Get() { return elements_[parlay::worker_id()]; }

  // Returns a const reference to the vector of all per-worker variables. The
  // length of this vector is parlay::GetScheduler().num_workers() and the
  // elements are indexed with parlay::GetScheduler().worker_id(). WARNING: The
  // caller needs to take care of synchronization if the per-worker elements are
  // being modified in parallel with accessing them.
  const std::vector<T>& GetAll() const { return elements_; }

  // Same as above but returns a non-const reference.
  std::vector<T>& GetAll() { return elements_; }

  // Returns a copy of the vector of all per-worker variables and resets the
  // values maintained by this per-worker instance. The length of the returned
  // vector is parlay::GetScheduler().num_workers() and the elements are indexed
  // with parlay::GetScheduler().worker_id().
  //
  // WARNING: The caller needs to take care of synchronization if the per-worker
  // elements are being modified in parallel with accessing them.
  std::vector<T> ReleaseAll() {
    std::vector<T> result(parlay::num_workers());
    result.swap(elements_);
    return result;
  }

 private:
  std::vector<T> elements_;
};

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PER_WORKER_H_
