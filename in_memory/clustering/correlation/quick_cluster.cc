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

#include "in_memory/clustering/correlation/quick_cluster.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
// #include "absl/random/random.h"
// #include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {
using ::research_graph::in_memory::CorrelationClustererConfig;
using ::std::vector;

vector<NodeId> RandomVisitOrder(int num_nodes, absl::BitGenRef rand) {
  vector<NodeId> visit_order;
  visit_order.reserve(num_nodes);
  for (NodeId i = 0; i < num_nodes; ++i) {
    visit_order.push_back(i);
  }
  std::shuffle(visit_order.begin(), visit_order.end(), rand);
  return visit_order;
}

vector<vector<NodeId>> QuickCluster(
    const SimpleUndirectedGraph& graph,
    const CorrelationClustererConfig& config,
    const vector<NodeId>& visit_order) {
  // It is assumed that resolution as well as node weights are non-negative.
  // Having this, the rescaled weight of a node to another node cannot be
  // positive if there is no edge between them. Therefore, to detect positive
  // rescaled edge weights of a node, we only need to loop through its edges.
  ABSL_CHECK_GE(config.resolution(), 0);
  auto cluster_together = [&graph, &config](NodeId center,
                                            NodeId other) -> bool {
    ABSL_CHECK_GE(graph.NodeWeight(center), 0);
    auto edge_weight = graph.EdgeWeight(center, other);
    // This CHECK is safe because cluster_together is only called on pairs of
    // vertices connected by an edge.
    ABSL_CHECK(edge_weight.has_value());
    const double offset = config.edge_weight_offset();
    const auto weight = *edge_weight - offset -
                        config.resolution() * graph.NodeWeight(center) *
                            graph.NodeWeight(other);
    return weight > 0;
  };
  return GeneralizedQuickCluster(graph, cluster_together, visit_order);
}

vector<vector<NodeId>> GeneralizedQuickCluster(
    const SimpleUndirectedGraph& graph,
    absl::FunctionRef<bool(NodeId center, NodeId other)> cluster_together,
    const vector<NodeId>& visit_order) {
  vector<vector<NodeId>> result;
  vector<bool> used(graph.NumNodes());
  for (const NodeId center : visit_order) {
    ABSL_CHECK_GE(center, 0);
    ABSL_CHECK_LT(center, graph.NumNodes());
    if (!used[center]) {
      used[center] = true;
      vector<NodeId> cluster = {center};
      for (const auto& edge : graph.Neighbors(center)) {
        const auto neighbor = edge.first;
        if (!used[neighbor] && cluster_together(center, neighbor)) {
          used[neighbor] = true;
          cluster.push_back(neighbor);
        }
      }
      result.push_back(std::move(cluster));
    }
  }
  return result;
}
}  // namespace graph_mining::in_memory
