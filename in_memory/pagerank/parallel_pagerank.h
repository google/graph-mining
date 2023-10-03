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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_

#include "absl/status/statusor.h"
#include "parlay/sequence.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/pagerank/pagerank.pb.h"

namespace third_party::graph_mining {

class ParallelPageRank {
 public:
  explicit ParallelPageRank(
      const ::graph_mining::in_memory::PageRankConfig& config)
      : config_(config) {}

  ::graph_mining::in_memory::UnweightedGbbsGraph* MutableGraph() {
    return &graph_;
  }

  // Returns a sequence where the i-th element contains the pagerank value for
  // the i-th node.
  absl::StatusOr<::parlay::sequence<double>> Run() const;

 private:
  ::graph_mining::in_memory::UnweightedGbbsGraph graph_;
  ::graph_mining::in_memory::PageRankConfig config_;
};

}  // namespace third_party::graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_
