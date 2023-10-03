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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TRIANGLE_COUNTING_PARALLEL_TRIANGLE_COUNTING_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TRIANGLE_COUNTING_PARALLEL_TRIANGLE_COUNTING_H_

#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace third_party::graph_mining {

// Parallel triangle counting.
class ParallelTriangleCounting {
 public:
  ::graph_mining::in_memory::InMemoryClusterer::Graph* MutableGraph() {
    return &graph_;
  }

  // Returns the number of triangles in `graph_`.
  absl::StatusOr<uint64_t> Count() const;

 private:
  // Must use a graph representation with sorted neighbors due to assumptions in
  // Gbbs library's internal implementation.
  ::graph_mining::in_memory::UnweightedSortedNeighborGbbsGraph graph_;
};

}  // namespace third_party::graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TRIANGLE_COUNTING_PARALLEL_TRIANGLE_COUNTING_H_
