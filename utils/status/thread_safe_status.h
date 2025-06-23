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

#ifndef THIRD_PARTY_GRAPH_MINING_UTILS_STATUS_THREAD_SAFE_STATUS_H_
#define THIRD_PARTY_GRAPH_MINING_UTILS_STATUS_THREAD_SAFE_STATUS_H_

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

namespace graph_mining {

// A thread-safe status class.
class ThreadSafeStatus {
 public:
  absl::Status status() const {
    absl::MutexLock l(&mutex_);
    return status_;
  }

  // If the internal status is OK, updates it to `status`. Otherwise, this is a
  // no-op.
  //
  // This method has the same semantics as `absl::Status::Update()`.
  void Update(const absl::Status& status) {
    // Avoid acquiring the mutex if `status` is OK, since the update would be a
    // no-op.
    if (status.ok()) {
      return;
    }
    absl::MutexLock l(&mutex_);
    status_.Update(status);
  }

 private:
  mutable absl::Mutex mutex_;
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_UTILS_STATUS_THREAD_SAFE_STATUS_H_
