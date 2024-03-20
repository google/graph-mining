// Copyright 2024 Google LLC
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

#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "in_memory/clustering/graph.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "parlay/parallel.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using graph_mining::in_memory::NodeId;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;

absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
ConvertToAdjList(const SimpleUndirectedGraph& graph) {
  auto num_nodes = graph.NumNodes();
  std::vector<graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>
      adj_lists(num_nodes);
  auto status = absl::OkStatus();
  parlay::parallel_for(0, num_nodes, [&](size_t i) {
    auto& neighbors = graph.Neighbors(i);
    adj_lists[i].weight = graph.NodeWeight(i);
    adj_lists[i].id = i;
    for (const auto kv : neighbors) {
      if (kv.second <= 0) {
        status = absl::InvalidArgumentError(absl::StrCat(
            "Non-positive edge weights, ", i, ", ", kv.first, ", ", kv.second));
      }
      if (kv.first == i) continue;  // skip self-loop
      adj_lists[i].outgoing_edges.push_back(kv);
    }
  });
  if (!status.ok()) return status;
  return adj_lists;
}

absl::StatusOr<std::vector<AdjacencyList>> ConvertToAdjList(
    absl::Span<const std::pair<NodeId, NodeId>> edges) {
  SimpleUndirectedGraph graph;
  for (const auto& [u, v] : edges) {
    RETURN_IF_ERROR(graph.AddEdge(u, v, 1));
  }
  return ConvertToAdjList(graph);
}

absl::StatusOr<std::vector<AdjacencyList>> ConvertToAdjList(
    absl::Span<const std::tuple<NodeId, NodeId, double>> edges,
    const absl::flat_hash_map<NodeId, NodeId>& node_weights) {
  SimpleUndirectedGraph graph;
  for (const auto& [u, v, w] : edges) {
    RETURN_IF_ERROR(graph.AddEdge(u, v, w));
  }
  for (int i = 0; i < graph.NumNodes(); ++i) {
    const auto it = node_weights.find(i);
    if (it == node_weights.end()) {
      graph.SetNodeWeight(i, 1);
    } else {
      graph.SetNodeWeight(i, it->second);
    }
  }
  return ConvertToAdjList(graph);
}

}  // namespace graph_mining::in_memory
