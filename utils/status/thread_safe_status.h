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

  void Update(const absl::Status& status) {
    absl::MutexLock l(&mutex_);
    status_.Update(status);
  }

  // Updates the internal status if the input is not OK. Returns true if the
  // internal status is updated. Returns false otherwise.
  bool MaybeUpdateStatus(const absl::Status& status) {
    if (status.ok()) {
      return false;
    } else {
      Update(status);
      return true;
    }
  }

 private:
  mutable absl::Mutex mutex_;
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_UTILS_STATUS_THREAD_SAFE_STATUS_H_
