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

#include "in_memory/clustering/gbbs_graph.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/bridge.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "utils/status/thread_safe_status.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

absl::Status GbbsGraph::ReweightGraph(
    const std::function<absl::StatusOr<float>(
        gbbs::uintE node_id, gbbs::uintE neighbor_id, std::size_t node_degree,
        std::size_t neighbor_degree, float current_edge_weight)>&
        edge_reweighter) {
  ThreadSafeStatus status;
  parlay::parallel_for(0, nodes_.size(), [&](std::size_t i) {
    for (std::size_t j = 0; j < nodes_[i].out_degree(); ++j) {
      auto& [neighbor_id, weight] = edges_[i][j];
      auto first_node_id = std::min(i, static_cast<std::size_t>(neighbor_id));
      auto second_node_id = std::max(i, static_cast<std::size_t>(neighbor_id));
      auto new_weight = edge_reweighter(
          first_node_id, second_node_id, nodes_[first_node_id].out_degree(),
          nodes_[second_node_id].out_degree(), weight);
      if (!new_weight.ok()) {
        status.Update(new_weight.status());
      } else {
        weight = *new_weight;
      }
    }
  });
  return status.status();
}

absl::Status UnweightedSortedNeighborGbbsGraph::Import(
    AdjacencyList adjacency_list) {
  std::sort(adjacency_list.outgoing_edges.begin(),
            adjacency_list.outgoing_edges.end());
  return GbbsGraphBase<gbbs::symmetric_ptr_graph<
      gbbs::symmetric_vertex, gbbs::empty>>::Import(std::move(adjacency_list));
}

}  // namespace graph_mining::in_memory
