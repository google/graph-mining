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

#include "in_memory/clustering/graph_utils.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;

double WeightedDegree(InMemoryClusterer::NodeId node,
                      const SimpleUndirectedGraph& graph) {
  double degree = 0.0;
  for (const auto& entry : graph.Neighbors(node)) {
    degree += entry.second;
  }
  return degree;
}

std::vector<double> WeightedDegrees(const SimpleUndirectedGraph& graph) {
  std::vector<double> degrees(graph.NumNodes());
  for (InMemoryClusterer::NodeId node = 0; node < graph.NumNodes(); node++) {
    degrees[node] = WeightedDegree(node, graph);
  }
  return degrees;
}

void IntersectEdgeSets(
    const absl::flat_hash_map<InMemoryClusterer::NodeId, double>& neighbors_i,
    const absl::flat_hash_map<InMemoryClusterer::NodeId, double>& neighbors_j,
    const std::function<void(SimpleUndirectedGraph::NodeId, double, double)>&
        f) {
  const auto [smaller, larger] = neighbors_i.size() <= neighbors_j.size()
                                     ? std::tie(neighbors_i, neighbors_j)
                                     : std::tie(neighbors_j, neighbors_i);

  for (const auto& [neighbor, weight] : smaller) {
    if (auto in_neighbor = larger.find(neighbor); in_neighbor != larger.end())
      f(neighbor, weight, in_neighbor->second);
  }
}

std::unique_ptr<SimpleDirectedGraph> DirectGraphByDegree(
    const SimpleUndirectedGraph& input) {
  auto directed_input = std::make_unique<SimpleDirectedGraph>();
  NodeId num_nodes = input.NumNodes();
  directed_input->SetNumNodes(num_nodes);

  std::vector<std::tuple<NodeId /*rank*/, NodeId /*vertex*/>> rank_to_vertex;

  rank_to_vertex.reserve(num_nodes);
  for (NodeId i = 0; i < num_nodes; i++) {
    rank_to_vertex.emplace_back(input.Neighbors(i).size(), i);
  }
  std::sort(rank_to_vertex.begin(), rank_to_vertex.end());

  // map ranks back to the degree
  std::vector<NodeId> ranks(num_nodes);
  for (NodeId i = 0; i < num_nodes; i++) {
    ranks[std::get<1>(rank_to_vertex[i])] = i;
  }

  // filter based on ranks, keeping edges from low->high rank
  for (NodeId i = 0; i < num_nodes; i++) {
    NodeId our_rank = ranks[i];
    for (const auto& [neighbor_id, weight] : input.Neighbors(i)) {
      NodeId neighbor_rank = ranks[neighbor_id];
      if (our_rank < neighbor_rank) {
        ABSL_CHECK_OK(directed_input->AddEdge(i, neighbor_id, weight));
      }
    }
  }
  return directed_input;
}

}  // namespace graph_mining::in_memory
