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

#include "in_memory/clustering/correlation/correlation_util.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ClusterId = int32_t;
using std::vector;

double CorrelationClusteringObjective(
    const SimpleUndirectedGraph& graph,
    const graph_mining::in_memory::CorrelationClustererConfig& config,
    const InMemoryClusterer::Clustering& clustering) {
  double total_agreements = 0;
  vector<ClusterId> cluster_of_node(graph.NumNodes());

  for (ClusterId cluster_id = 0; cluster_id < clustering.size(); ++cluster_id) {
    for (NodeId node_id : clustering[cluster_id]) {
      cluster_of_node[node_id] = cluster_id;
    }
  }

  for (ClusterId cluster_id = 0; cluster_id < clustering.size(); ++cluster_id) {
    double sum_node_weights = 0;
    double agreements_in_cluster = 0;
    for (NodeId node_id : clustering[cluster_id]) {
      double node_weight = graph.NodeWeight(node_id);
      agreements_in_cluster -=
          node_weight * sum_node_weights * config.resolution();
      sum_node_weights += graph.NodeWeight(node_id);

      for (const auto& edge : graph.Neighbors(node_id)) {
        const auto neighbor = edge.first;
        const auto weight = edge.second - config.edge_weight_offset();

        if (neighbor != node_id && cluster_of_node[neighbor] == cluster_id) {
          // It is divided by two to account for double counting edges.
          agreements_in_cluster += weight / 2;
        }
      }
    }

    total_agreements += agreements_in_cluster;
  }

  return total_agreements;
}

}  // namespace graph_mining::in_memory
