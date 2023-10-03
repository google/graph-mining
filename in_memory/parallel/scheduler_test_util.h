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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_SCHEDULER_TEST_UTIL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_SCHEDULER_TEST_UTIL_H_

#include "in_memory/parallel/scheduler.h"
#include "gtest/gtest.h"

namespace graph_mining::in_memory {

class ParallelSchedulerReferenceTest : public testing::Test {
 protected:
  static constexpr int kNumWorkers = 6;

 private:
  graph_mining::in_memory::ParallelSchedulerReference scheduler_{kNumWorkers};
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_SCHEDULER_TEST_UTIL_H_
