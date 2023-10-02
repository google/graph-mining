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

#include "in_memory/pagerank/parallel_pagerank.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "benchmarks/PageRank/PageRank.h"
#include "in_memory/parallel/scheduler.h"

namespace third_party::graph_mining {

  absl::StatusOr<parlay::sequence<double>> ParallelPageRank::Run() const {
    if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "graph_ must be initialized before triangle counting.");
  }

  
  return ::gbbs::PageRank(*(graph_.Graph()), config_.approx_precision(),
                          config_.num_iterations());
  }

}  // namespace third_party::graph_mining
