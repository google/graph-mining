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

#include "in_memory/clustering/clique_aggregator/degeneracy_orientation.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "in_memory/clustering/clique_aggregator/graphs.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining {
namespace in_memory {

namespace {

// Helper function to validate the ordering and create a position map.
// Returns the position map if the ordering is valid, `std::nullopt` otherwise.
std::optional<std::vector<NodeId>> ValidateOrderingAndCreatePositionMap(
    NodeId num_nodes, absl::Span<const NodeId> ordering) {
  if (ordering.size() != num_nodes) return std::nullopt;
  std::vector<NodeId> position(num_nodes, -1);
  for (NodeId i = 0; i < ordering.size(); ++i) {
    if (ordering[i] >= num_nodes || ordering[i] < 0 ||
        position[ordering[i]] != -1) {
      return std::nullopt;
    }
    position[ordering[i]] = i;
  }
  return position;
}

}  // namespace

std::pair<absl_nullable std::unique_ptr<BitSetGraph>,
          absl_nullable std::unique_ptr<BitSetGraph>>
DirectGraph(const BitSetGraph& graph, absl::Span<const NodeId> ordering) {
  auto position =
      ValidateOrderingAndCreatePositionMap(graph.NumNodes(), ordering);
  if (!position.has_value()) {
    return {nullptr, nullptr};
  }
  auto directed_graph = std::make_unique<BitSetGraph>(graph.NumNodes());
  auto transposed_graph = std::make_unique<BitSetGraph>(graph.NumNodes());

  for (NodeId node_id = 0; node_id < graph.NumNodes(); ++node_id) {
    graph.MapNeighbors(node_id, [&](NodeId neighbor_id) {
      if ((*position)[node_id] < (*position)[neighbor_id]) {
        directed_graph->AddEdge(node_id, neighbor_id);
        transposed_graph->AddEdge(neighbor_id, node_id);
      }
    });
  }

  return {std::move(directed_graph), std::move(transposed_graph)};
}

std::pair<absl_nullable std::unique_ptr<
              GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>,
          absl_nullable std::unique_ptr<
              GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>>
DirectGraph(const GbbsGraphWrapper<UnweightedGbbsGraph>& graph,
            absl::Span<const NodeId> ordering) {
  auto position =
      ValidateOrderingAndCreatePositionMap(graph.NumNodes(), ordering);
  if (!position.has_value()) {
    return {nullptr, nullptr};
  }
  auto directed_graph = std::make_unique<DirectedUnweightedOutEdgesGbbsGraph>();
  ABSL_CHECK_OK(directed_graph->PrepareImport(graph.NumNodes()));
  auto transposed_graph =
      std::make_unique<DirectedUnweightedOutEdgesGbbsGraph>();
  ABSL_CHECK_OK(transposed_graph->PrepareImport(graph.NumNodes()));

  for (NodeId node_id = 0; node_id < graph.NumNodes(); ++node_id) {
    DirectedUnweightedOutEdgesGbbsGraph::AdjacencyList adjacency_list;
    DirectedUnweightedOutEdgesGbbsGraph::AdjacencyList
        transposed_adjacency_list;
    adjacency_list.id = node_id;
    transposed_adjacency_list.id = node_id;
    graph.MapNeighbors(node_id, [&](NodeId neighbor_id) {
      if ((*position)[node_id] < (*position)[neighbor_id]) {
        adjacency_list.outgoing_edges.push_back({neighbor_id, {}});
      } else if ((*position)[node_id] > (*position)[neighbor_id]) {
        transposed_adjacency_list.outgoing_edges.push_back({neighbor_id, {}});
      }
    });
    absl::c_sort(adjacency_list.outgoing_edges);
    absl::c_sort(transposed_adjacency_list.outgoing_edges);
    ABSL_CHECK_OK(directed_graph->Import(adjacency_list));
    ABSL_CHECK_OK(transposed_graph->Import(transposed_adjacency_list));
  }
  ABSL_CHECK_OK(directed_graph->FinishImport());
  ABSL_CHECK_OK(transposed_graph->FinishImport());

  return {
      std::make_unique<GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>(
          std::move(directed_graph)),
      std::make_unique<GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>(
          std::move(transposed_graph))};
}

}  // namespace in_memory
}  // namespace graph_mining
