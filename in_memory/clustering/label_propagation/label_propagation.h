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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_

#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

// This file provides a parallel implementation of weighted-majority label
// propagation. The algorithm starts by setting the label of each node to its
// own node_id (by default; optionally the user can supply an initial set of
// labels TODO).
//
// In each round of the algorithm, the algorithm wakes up all active vertices
// and computes their new label by scanning their neighborhood and picking the
// most frequent label, taking into account the edge weights. If there are ties,
// the algorithm chooses the lexicographically largest label of equal weight.
//
// The algorithm runs asynchronously, i.e., the set of labels read-from and
// written-to in the same round are the same. This makes the behavior of the
// algorithm non-deterministic by default.
//
// If determinism is desired, one can set the use_graph_coloring flag to true.
// Then the algorithm will color the input graph using a greedy graph coloring
// algorithm that tries to minimize the number of colors. In each round, the
// algorithm runs a number of sub-rounds equal to the number of colors, and
// processes all active vertices of the same color together. Using this
// approach, only a single label-set is required, but the behavior becomes
// deterministic due to using a coloring.
namespace graph_mining::in_memory {

class ParallelLabelPropagationClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override;

 protected:
  graph_mining::in_memory::GbbsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_
