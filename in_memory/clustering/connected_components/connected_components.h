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

#include <cstddef>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/connected_components/connected_components_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

// Clusterer that produces one cluster for each connected component in the
// graph, ignoring edge directions.
class ParallelConnectedComponentsClusterer : public InMemoryClusterer {
 public:
  absl::Nonnull<Graph*> MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override {
    
    ASSIGN_OR_RETURN(absl::Span<const NodeId> parents, graph_.ParentArray());
    return graph_mining::in_memory::OutputIndicesById<NodeId, NodeId>(parents);
  }

  absl::StatusOr<std::vector<NodeId>> ClusterAndReturnClusterIds(
      const graph_mining::in_memory::ClustererConfig& config) const override {
    
    ASSIGN_OR_RETURN(absl::Span<const NodeId> parents, graph_.ParentArray());
    std::vector<NodeId> cluster_ids(parents.size());
    parlay::parallel_for(0, cluster_ids.size(),
                         [&](std::size_t i) { cluster_ids[i] = parents[i]; });
    return cluster_ids;
  }

 private:
  ConnectedComponentsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_
