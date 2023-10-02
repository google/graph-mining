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

#include "in_memory/clustering/correlation/parallel_modularity.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

namespace {

using research_graph::in_memory::ClustererConfig;

struct NodeWeightsTotalWeight {
  std::vector<double> node_weights;
  double total_node_weight;
};

NodeWeightsTotalWeight GetNodeWeightsAndTotalWeight(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) {
  double total_node_weight = 0.0;
  std::vector<double> node_weights(graph->n, 0);
  for (std::size_t i = 0; i < graph->n; i++) {
    auto map_weight = [&](gbbs::uintE vertex, gbbs::uintE neighbor,
                          double weight) { node_weights[i] += weight; };
    graph->get_vertex(i).out_neighbors().map(map_weight, false);

    total_node_weight += node_weights[i];
  }
  return NodeWeightsTotalWeight{std::move(node_weights), total_node_weight};
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelModularityClusterer::Cluster(const ClustererConfig& config) const {
  
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  parlay::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<int32_t>(i)};
  });

  RETURN_IF_ERROR(
      ParallelModularityClusterer::RefineClusters(config, &clustering));

  return clustering;
}

absl::Status ParallelModularityClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    Clustering* initial_clustering) const {
  if (clusterer_config.has_correlation_clusterer_config()) {
    return ParallelCorrelationClusterer::RefineClusters(clusterer_config,
                                                        initial_clustering);
  }

  // Set modularity clustering config
  ClustererConfig modularity_config;
  (*modularity_config.mutable_correlation_clusterer_config()) =
      clusterer_config.modularity_clusterer_config().correlation_config();
  auto node_weights_total_weight = GetNodeWeightsAndTotalWeight(graph_.Graph());
  modularity_config.mutable_correlation_clusterer_config()->set_resolution(
      clusterer_config.modularity_clusterer_config().resolution() /
      node_weights_total_weight.total_node_weight);
  modularity_config.mutable_correlation_clusterer_config()
      ->set_edge_weight_offset(0.0);

  ClusteringHelper helper{static_cast<NodeId>(graph_.Graph()->n),
                          modularity_config,
                          std::move(node_weights_total_weight.node_weights),
                          *initial_clustering, graph_.GetNodeParts()};
  return ParallelCorrelationClusterer::RefineClusters(
      modularity_config, initial_clustering, &helper);
}

}  // namespace graph_mining::in_memory
