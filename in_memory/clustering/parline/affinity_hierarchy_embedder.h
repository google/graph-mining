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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_AFFINITY_HIERARCHY_EMBEDDER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_AFFINITY_HIERARCHY_EMBEDDER_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/parline/linear_embedder.h"

namespace graph_mining::in_memory {

// Embeds the nodes of an input graph into a line using a clustering hierarchy
// which is built by running affinity clustering separately at each level.
class AffinityHierarchyEmbedder : public LinearEmbedder {
 public:
  // Constructs an embedder using the given config. Note that several fields of
  // the config are filled in with defaults if not provided.
  explicit AffinityHierarchyEmbedder(
      const AffinityClustererConfig& config);
  ~AffinityHierarchyEmbedder() override {}

  // TODO: Replace std::vector with a parallel version.
  absl::StatusOr<std::vector<gbbs::uintE>> EmbedGraph(
      const GbbsGraph& graph) override;

  absl::StatusOr<std::vector<std::pair<gbbs::uintE, double>>>
  EmbedGraphWeighted(const GbbsGraph& graph) override;

  // Computes affinity hierarchy paths for all the nodes in the graph. Entry i
  // contains the vector of cluster ids {C_0, C_1, ..., C_k} along the hierarchy
  // path that node with id i belongs to (where C_0 = i is the leaf node id and
  // C_k is the top level cluster id which is usually a connected component).
  absl::StatusOr<std::vector<std::vector<gbbs::uintE>>>
  ComputeAffinityHierarchyPaths(const GbbsGraph& graph);

 private:
  AffinityClustererConfig affinity_config_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_AFFINITY_HIERARCHY_EMBEDDER_H_
