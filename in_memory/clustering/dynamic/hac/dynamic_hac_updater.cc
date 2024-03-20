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

#include "in_memory/clustering/dynamic/hac/dynamic_hac_updater.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"
#include "utils/status/thread_safe_status.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

using graph_mining::in_memory::NodeId;
using Dendrogram = graph_mining::in_memory::Dendrogram;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;

namespace {

// Returns the parent node id if `node_a` and `node_b` merges in
// `dendrogram`. Returns nullopt if `dendrogram` does not
// satisfy its predicate.
std::optional<NodeId> MutualParentId(const DynamicDendrogram& dendrogram,
                                     NodeId node_a, NodeId node_b) {
  if (!dendrogram.HasNode(node_a) || !dendrogram.HasNode(node_b)) {
    return std::nullopt;
  }
  const auto sib_a = dendrogram.Sibling(node_a);
  if (!sib_a.has_value()) return std::nullopt;
  if (sib_a == node_b) {
    return dendrogram.Parent(node_a).value().parent_id;
  }
  return std::nullopt;
}

absl::StatusOr<NodeId> NextRoundId(
    const NodeId global_id,
    const absl::flat_hash_map<NodeId, NodeId>& next_round_node_map) {
  auto itr = next_round_node_map.find(global_id);
  if (itr != next_round_node_map.end()) {
    return itr->second;
  } else {
    return absl::NotFoundError(
        absl::StrCat("node not in next_round_node_map, ", global_id));
  }
}

}  // namespace

absl::Status UpdateDendrogram(
    const std::vector<std::tuple<NodeId, NodeId, double>>& merges,
    std::vector<NodeId>& subgraph_node_map, NextUnusedId& next_unused_id,
    DynamicDendrogram& dynamic_dendrogram) {
  for (NodeId j = 0; j < merges.size(); ++j) {
    const auto& [cluster_u, cluster_v, w] = merges[j];
    ABSL_CHECK_LT(cluster_u, subgraph_node_map.size());
    ABSL_CHECK_LT(cluster_v, subgraph_node_map.size());
    const auto global_u = subgraph_node_map[cluster_u];
    const auto global_v = subgraph_node_map[cluster_v];

    const auto parent_id =
        MutualParentId(dynamic_dendrogram, global_u, global_v);
    const auto uv_merge_in_dendrogram = parent_id.has_value();

    NodeId internal_node_id;
    if (uv_merge_in_dendrogram) {
      internal_node_id = parent_id.value();
    } else {
      internal_node_id = next_unused_id();
      // Remove ancestors of invalid merge.
      ASSIGN_OR_RETURN(
          const auto dirty_ancestors,
          dynamic_dendrogram.RemoveAncestors({global_u, global_v}));
      RETURN_IF_ERROR(dynamic_dendrogram.AddInternalNode(
          internal_node_id, global_u, global_v, w));
    }
    subgraph_node_map.push_back(internal_node_id);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::tuple<NodeId, NodeId, NodeId>>>
CurrentMapNextRound(
    const absl::Span<const NodeId> active_nodes,
    const absl::flat_hash_set<NodeId>& deleted_nodes,
    const absl::flat_hash_set<NodeId>& new_nodes,
    const absl::flat_hash_map<NodeId, NodeId>& next_round_node_map) {
  std::vector<std::tuple<NodeId, NodeId, NodeId>> mapping;
  // Mapping of deleted nodes.
  for (NodeId i : deleted_nodes) {
    ASSIGN_OR_RETURN(const auto mapped_id, NextRoundId(i, next_round_node_map));
    mapping.push_back(std::make_tuple(-1, i, mapped_id));
  }
  // Mapping of active nodes.
  for (NodeId local_id = 0; local_id < active_nodes.size(); ++local_id) {
    NodeId global_id = active_nodes[local_id];
    if (new_nodes.contains(global_id)) {
      mapping.push_back(std::make_tuple(local_id, global_id, -1));
      continue;
    }
    ASSIGN_OR_RETURN(const auto mapped_id,
                     NextRoundId(global_id, next_round_node_map));
    mapping.push_back(std::make_tuple(local_id, global_id, mapped_id));
  }
  return mapping;
}

absl::StatusOr<absl::flat_hash_set<NodeId>> NodesToDelete(
    absl::Span<const std::tuple<NodeId, NodeId, NodeId>> current_mapping,
    const absl::flat_hash_map<NodeId, NodeId>& root_map,
    const absl::flat_hash_set<NodeId>& active_contracted_nodes,
    absl::flat_hash_map<NodeId, NodeId>& next_round_node_map) {
  absl::flat_hash_set<NodeId> nodes_to_delete;
  for (const auto [local_id, global_id, old_map] : current_mapping) {
    if (local_id == -1) {  // node is deleted
      if (active_contracted_nodes.contains(old_map)) continue;
      nodes_to_delete.insert(old_map);
      continue;
    }
    const auto itr = root_map.find(global_id);
    if (itr == root_map.end()) {
      return absl::NotFoundError(
          absl::StrCat("node not in root_map, ", global_id));
    }
    const auto global_root = itr->second;
    if (old_map != -1 && old_map != global_root &&
        !active_contracted_nodes.contains(old_map)) {
      nodes_to_delete.insert(old_map);
    }
    next_round_node_map[global_id] = global_root;
  }
  return nodes_to_delete;
}

absl::StatusOr<absl::flat_hash_map<NodeId, NodeId>> MappingLastRound(
    const absl::Span<const NodeId> active_nodes,
    const absl::flat_hash_map<NodeId, NodeId>& root_map) {
  absl::flat_hash_map<NodeId, NodeId> next_round_node_map;
  std::vector<std::tuple<NodeId, NodeId, NodeId>> mapping;
  ABSL_CHECK_EQ(next_round_node_map.size(), 0);
  for (NodeId local_id = 0; local_id < active_nodes.size(); ++local_id) {
    const NodeId global_id = active_nodes[local_id];
    const auto itr = root_map.find(global_id);
    if (itr == root_map.end()) {
      return absl::NotFoundError(
          absl::StrCat("node not in root_map, ", global_id));
    }
    const auto global_root = itr->second;
    next_round_node_map[global_id] = global_root;
  }
  return next_round_node_map;
}

absl::StatusOr<std::vector<AdjacencyList>> AdjacencyListsOfNewNodes(
    const std::vector<NodeId>& new_nodes,
    const SubgraphClusterId& subgraph_cluster_id,
    const std::unique_ptr<ContractedGraph>& contracted_graph,
    const absl::flat_hash_map<NodeId, NodeId>& next_round_node_map,
    const absl::flat_hash_set<NodeId>& deleted_nodes,
    const DynamicClusteredGraph& graph) {
  std::vector<AdjacencyList> adjacency_lists(new_nodes.size());
  graph_mining::ThreadSafeStatus loop_status;
  parlay::parallel_for(0, new_nodes.size(), [&](size_t i) {
    const NodeId local_id = new_nodes[i];
    NodeId global_id = subgraph_cluster_id(local_id);
    const auto i_size = contracted_graph->NodeWeight(local_id);
    if (deleted_nodes.contains(global_id)) {
      loop_status.Update(absl::InternalError(
          absl::StrCat("node to insert should not be deleted ", global_id)));
      return;
    }
    adjacency_lists[i].id = global_id;
    adjacency_lists[i].weight = i_size;
    // Store sum of edge similarity to inactive nodes. We need this because
    // inactive nodes may be merged outside of the subgraph hac.
    absl::flat_hash_map<NodeId, double> edge_to_inactive_nodes;
    for (auto [sim, neighbor] :
         contracted_graph->UnnormalizedNeighborsSimilarity(local_id)) {
      NodeId global_neigh = subgraph_cluster_id(neighbor);
      NodeId mapped_neigh = global_neigh;
      if (contracted_graph->IsInactive(neighbor)) {
        // Inactive nodes need to be mapped to next round.
        const auto status = NextRoundId(global_neigh, next_round_node_map);
        if (!status.ok()) {
          loop_status.Update(status.status());
          return;
        }
        mapped_neigh = status.value();
        // The neighbor does not have any heavy edge and is ignored. Only
        // inactive nodes can possibly map to -1.
        if (mapped_neigh == -1) {
          continue;
        }
        if (deleted_nodes.contains(mapped_neigh)) {
          loop_status.Update(absl::InternalError(
              absl::StrCat("invalid neighbor after mapping ", mapped_neigh)));
          return;
        }
        // the edge weights in `contracted_graph` is not multiplied by the
        // weight of inactive nodes, because the weight of inactive nodes is
        // -1.
        const auto node_status = graph.ImmutableNode(global_neigh);
        if (!node_status.ok()) {
          loop_status.Update(node_status.status());
          return;
        }
        const auto node = node_status.value();
        const auto neigh_size = node->ClusterSize();
        sim *= neigh_size;
        auto itr = edge_to_inactive_nodes.find(mapped_neigh);
        if (itr != edge_to_inactive_nodes.end()) {
          itr->second += sim;
        } else {
          edge_to_inactive_nodes[mapped_neigh] = sim;
        }
        continue;  // will insert these edges outside of the for loop.

      } else {
        if (deleted_nodes.contains(mapped_neigh)) {
          loop_status.Update(absl::InternalError(absl::StrCat(
              "invalid neighbor in contracted graph ", mapped_neigh)));
          return;
        }
      }
      adjacency_lists[i].outgoing_edges.push_back({mapped_neigh, sim});
    }
    // Insert edges to merged inactive nodes.
    for (auto [mapped_neigh, sim] : edge_to_inactive_nodes) {
      adjacency_lists[i].outgoing_edges.push_back({mapped_neigh, sim});
    }
  });
  if (!loop_status.status().ok()) return loop_status.status();
  return adjacency_lists;
}

}  // namespace graph_mining::in_memory
