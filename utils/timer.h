/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_GRAPH_MINING_UTILS_TIMER_H_
#define THIRD_PARTY_GRAPH_MINING_UTILS_TIMER_H_

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace graph_mining {

// An interface to the system clock.
// This class is thread-compatible.
//
class WallTimer {
 public:
  // Creates a new wall timer and starts the timer.
  inline WallTimer() : start_time_(absl::Now()) {}

  inline void Restart() {
    start_time_ = absl::Now();
  }

  inline double GetSeconds() const {
    return absl::ToDoubleSeconds(absl::Now() - start_time_);
  }

 private:
  absl::Time start_time_;
};

}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_UTILS_TIMER_H_
