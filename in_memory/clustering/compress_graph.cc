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

#include "in_memory/clustering/compress_graph.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "in_memory/clustering/graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining {
namespace in_memory {

absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> CompressGraph(
    const SimpleUndirectedGraph& graph,
    const std::vector<SimpleUndirectedGraph::NodeId>& cluster_ids,
    const std::function<double(double, double)>& edge_aggregation_function,
    bool ignore_self_loops) {
  auto result = std::make_unique<SimpleUndirectedGraph>();
  using NodeId = SimpleUndirectedGraph::NodeId;
  result->SetNumNodes(graph.NumNodes());
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    result->SetNodeWeight(i, 0);
  }

  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    if (cluster_ids[i] == -1) continue;

    double compressed_node_weight = result->NodeWeight(cluster_ids[i]);
    result->SetNodeWeight(cluster_ids[i],
                          compressed_node_weight + graph.NodeWeight(i));

    for (const auto& [neighbor_id, edge_weight] : graph.Neighbors(i)) {
      // Process each undirected edge once
      if (neighbor_id > i || cluster_ids[neighbor_id] == -1) continue;

      if (ignore_self_loops && cluster_ids[i] == cluster_ids[neighbor_id]) {
        continue;
      }

      double compressed_weight =
          result->EdgeWeight(cluster_ids[i], cluster_ids[neighbor_id])
              .value_or(0);

      double new_weight =
          edge_aggregation_function(compressed_weight, edge_weight);

      RETURN_IF_ERROR(result->SetEdgeWeight(
          cluster_ids[i], cluster_ids[neighbor_id], new_weight));
    }
  }
  return result;
}

}  // namespace in_memory
}  // namespace graph_mining
