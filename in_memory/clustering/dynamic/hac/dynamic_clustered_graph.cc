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

#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parallel_clustered_graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

namespace {

using NodeId = DynamicClusteredGraph::NodeId;
using Subgraph = DynamicClusteredGraph::Subgraph;

// Return a subgraph of size `active_nodes.size() + inactive_nodes.size()`.
// Node i in returned graph corresponds to `active_nodes`[i] in `graph` and has
// the same cluster size as the corresponding node.
// Node (i + `active_nodes`.size()) in returned graph corresponds to
// `inactive_nodes`[i] in `graph` and has the same cluster size as the
// corresponding node. The edges in the returned graph are added according to
// `graph_edges`. `graph_edges` use the node ids in `graph`. Returns error
// status is any node in `active_nodes` or `inactive_nodes` is not in `graph`.
absl::StatusOr<std::unique_ptr<Subgraph>> CreateSubgraphHelper(
    const absl::flat_hash_set<NodeId>& active_nodes,
    const absl::flat_hash_set<NodeId>& inactive_nodes,
    absl::Span<const std::tuple<NodeId, NodeId, double>> graph_edges,
    const DynamicClusteredGraph& graph) {
  auto subgraph = std::make_unique<Subgraph>();

  const std::size_t partition_num_nodes =
      active_nodes.size() + inactive_nodes.size();
  subgraph->num_active_nodes = active_nodes.size();
  // Need this line when the graph has a single node without any edge.
  subgraph->graph =
      std::make_unique<graph_mining::in_memory::SimpleUndirectedGraph>();
  subgraph->graph->SetNumNodes(partition_num_nodes);

  // `subgraph.node_map[partition_node_id]` is the original node_id.
  // Active nodes are in front of the inactive nodes.
  subgraph->node_map = std::vector<NodeId>(partition_num_nodes);
  absl::flat_hash_map<NodeId, gbbs::uintE> node_map_rev;
  // `j` is local node id and the index of `node_map`.
  NodeId j = 0;
  for (const auto& v : active_nodes) {
    subgraph->node_map[j] = v;
    node_map_rev[v] = j;
    ASSIGN_OR_RETURN(const auto& node, graph.ImmutableNode(v));
    subgraph->graph->SetNodeWeight(j, node->ClusterSize());
    j++;
  }
  for (const auto& v : inactive_nodes) {
    subgraph->node_map[j] = v;
    node_map_rev[v] = j;
    subgraph->graph->SetNodeWeight(j, -1);
    j++;
  }

  // If we add adjacency lists in parallel, we need to use PrepareImport.
  for (const auto& [u, v, w] : graph_edges) {
    RETURN_IF_ERROR(
        subgraph->graph->AddEdge(node_map_rev[u], node_map_rev[v], w));
  }
  return std::move(subgraph);
}

}  // namespace

double DynamicClusteredGraph::StableSimilarity(NodeId a, NodeId b) const {
  const NodeId u = std::min(a, b);
  const NodeId v = std::max(a, b);

  const auto& node_u = clusters_.find(u)->second;
  const auto size_u = node_u.ClusterSize();
  const auto weight = node_u.GetImmutableNeighbors().FindValue(v).value();
  const double similarity = weight.Similarity(size_u);
  return similarity;
}

absl::Status DynamicClusteredGraph::AddNodes(
    absl::Span<const AdjacencyList> nodes, bool skip_existing_nodes) {
  absl::flat_hash_set<NodeId> skipped_nodes;
  absl::flat_hash_set<NodeId> new_nodes_set;
  for (const auto& node : nodes) {
    new_nodes_set.insert(node.id);
  }
  // Add an entry for each node to `clusters_`.
  for (const auto& node : nodes) {
    if (node.weight <= 0) {
      return absl::InvalidArgumentError(
          "node weight is non-positive, node id = " + std::to_string(node.id));
    }
    auto node_id = node.id;
    auto neighborhood_size = node.outgoing_edges.size();

    auto cluster_node = ClusterNode(node_id);
    cluster_node.SetClusterSize(node.weight);
    if (neighborhood_size > 0) {
      cluster_node.GetNeighbors()->AdjustSizeForIncoming(neighborhood_size);
    }
    auto result =
        clusters_.insert(std::make_pair(node_id, std::move(cluster_node)));
    if (!result.second) {
      if (skip_existing_nodes) {
        skipped_nodes.insert(node_id);
      } else {
        return absl::AlreadyExistsError("node already exists, node id = " +
                                        std::to_string(node_id));
      }
    }
  }

  // Insert the edges incident to the nodes. For each edge (i,j) we insert both
  // (i,j) and (j,i) because the graph is undirected.
  for (const auto& node : nodes) {
    auto node_id = node.id;
    if (skip_existing_nodes && skipped_nodes.contains(node_id)) continue;
    auto cluster_size = node.weight;
    auto neighbors = clusters_[node_id].GetNeighbors();

    auto update_f_node = [&](Weight* old_weight) {};
    for (auto [v, cut_weight] : node.outgoing_edges) {
      auto node_v_iter = clusters_.find(v);
      if (node_v_iter == clusters_.end()) {
        return absl::FailedPreconditionError("edge to non-existing node " +
                                             std::to_string(v));
      }

      if (node.id == v) {
        return absl::InvalidArgumentError("self edge should not exist " +
                                          std::to_string(v));
      }
      if (new_nodes_set.contains(v) && node_id < v) {
        continue;
      }
      // Add weighted edge from node_id to v
      Weight weight(cut_weight);
      auto node_v_size = node_v_iter->second.ClusterSize();
      weight.UpdateNeighborSize(node_v_size);
      // We should never need to update
      neighbors->AdjustSizeForIncoming(1);
      neighbors->InsertOrUpdate(v, weight, update_f_node);

      // Add weighted edge from v to node_id
      Weight weight_v(cut_weight);
      weight_v.UpdateNeighborSize(cluster_size);
      // We should never need to update
      node_v_iter->second.GetNeighbors()->AdjustSizeForIncoming(1);
      node_v_iter->second.GetNeighbors()->InsertOrUpdate(node_id, weight_v,
                                                         update_f_node);

      // Update auxiliary data for maintaining max edge weight.
      const double similarity = StableSimilarity(node_id, v);
      ++num_edges_;
      if (similarity >= heavy_threshold_) {
        ++num_heavy_edges_;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status DynamicClusteredGraph::RemoveNode(NodeId node_id) {
  auto node_iter = clusters_.find(node_id);
  if (node_iter == clusters_.end())
    return absl::NotFoundError("node not in graph, node id = " +
                               std::to_string(node_id));
  auto neighbors = node_iter->second.GetNeighbors();

  // Update `num_edges_` and `num_heavy_edges_`.
  num_edges_ -= neighbors->Size();
  ABSL_CHECK_GE(num_edges_, 0);

  struct plus : public std::plus<> {
    using T = std::size_t;
    T identity;
    plus() : identity(0) {}
  };
  const auto num_deleted_heavy_edges = neighbors->MapReduce(
      [&](gbbs::uintE v, Weight _) {
        double similarity = StableSimilarity(node_id, v);
        return similarity >= heavy_threshold_ ? static_cast<std::size_t>(1)
                                              : static_cast<std::size_t>(0);
      },
      plus());
  num_heavy_edges_ -= num_deleted_heavy_edges;
  ABSL_CHECK_GE(num_heavy_edges_, 0);

  // Update neighbors.
  neighbors->Map([&](gbbs::uintE v, Weight weight) {
    auto v_neighbors = clusters_[v].GetNeighbors();
    // Remove `node_id` from `v`'s neighbors.
    bool deleted = v_neighbors->Remove(node_id);
    ABSL_CHECK(deleted);
  });
  clusters_.erase(node_iter);
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<DynamicClusteredGraph::Subgraph>>
DynamicClusteredGraph::CreateSubgraph(
    const absl::flat_hash_set<NodeId>& node_ids,
    const absl::flat_hash_map<NodeId, NodeId>& partition_map) const {
  using uintE = gbbs::uintE;
  std::vector<std::tuple<NodeId, NodeId, double>> graph_edges;
  absl::flat_hash_set<NodeId> inactive_nodes;
  // The subset of inactive nodes that are inactive only because they do not
  // have any heavy edges. They are in global id.
  absl::flat_hash_set<NodeId> inactive_nodes_no_heavy;
  // Active nodes. They are in global id.
  absl::flat_hash_set<NodeId> active_nodes;
  for (const auto v : node_ids) {
    if (!HasHeavyEdges(v).value()) {
      inactive_nodes_no_heavy.insert(v);
      inactive_nodes.insert(v);
    } else {
      active_nodes.insert(v);
    }
  }
  absl::Status status = absl::OkStatus();

  // Nodes we need to process (check their neighbors).
  std::queue<NodeId> node_to_process;
  for (const auto node_id : node_ids) {
    node_to_process.push(node_id);
  }

  // Do a BFS over the active parts of the graph. For each active node, add all
  // edges incident to it to the edge list. For each inactive node, add it to
  // inactive_nodes.
  NodeId cluster_size = 0;
  while (!node_to_process.empty()) {
    const auto node_id = node_to_process.front();
    node_to_process.pop();
    // Add any edge that has at least one node in the partition.
    // A node is "active" if it's in `node_ids`.
    auto map_f = [&](uintE v, DynamicClusteredGraph::Weight weight) {
      const double edge_weight = weight.Similarity(cluster_size);
      if (node_id == v) {
        status = absl::FailedPreconditionError(
            absl::StrCat("graph should not contain self edge, v=", v));
        return true;
      }
      const auto it = partition_map.find(v);
      if (it == partition_map.end()) {
        status = absl::NotFoundError(
            absl::StrCat("Node not found in partition_map, id = ", v));
        return true;
      }

      graph_edges.push_back(std::make_tuple(node_id, v, edge_weight));

      const NodeId& v_target = it->second;
      if (node_ids.contains(v_target)) {
        if (!HasHeavyEdges(v).value()) {  // v is inactive only because it has
                                          // no heavy edge.
          inactive_nodes.insert(v);
          inactive_nodes_no_heavy.insert(v);
        } else {  // v is active
          const auto result = active_nodes.insert(v);
          // It's not processed and not in the queue, so we add it.
          if (result.second) node_to_process.push(v);
        }
        // return false;
      } else {  // v is inactive because it is not in the partition.
        inactive_nodes.insert(v);
      }
      return false;
    };
    ASSIGN_OR_RETURN(auto node, ImmutableNode(node_id));
    cluster_size = node->ClusterSize();
    node->GetImmutableNeighbors().IterateUntil(map_f);
    if (!status.ok()) {
      return status;
    }
  }
  ASSIGN_OR_RETURN(
      auto subgraph,
      CreateSubgraphHelper(active_nodes, inactive_nodes, graph_edges, *this));
  for (NodeId i : inactive_nodes_no_heavy) {
    subgraph->ignored_nodes.push_back(i);
  }
  return std::move(subgraph);
}

absl::StatusOr<absl::flat_hash_set<DynamicClusteredGraph::NodeId>>
DynamicClusteredGraph::Neighbors(
    const absl::flat_hash_set<DynamicClusteredGraph::NodeId>& nodes) const {
  absl::flat_hash_set<NodeId> neighbors;
  for (const auto& node_id : nodes) {
    auto map_f = [&](gbbs::uintE v, double _) {
      neighbors.insert(v);
      return false;
    };
    ASSIGN_OR_RETURN(auto node, ImmutableNode(node_id));
    node->IterateUntil(map_f);
  }

  return neighbors;
}

absl::StatusOr<bool> DynamicClusteredGraph::HasHeavyEdges(NodeId i) const {
  bool has_heavy = false;
  auto map_f = [&](gbbs::uintE _, double w) {
    if (w >= heavy_threshold_) {
      has_heavy = true;
      return true;
    }
    return false;
  };
  ASSIGN_OR_RETURN(auto node, ImmutableNode(i));
  node->IterateUntil(map_f);
  return has_heavy;
}

}  // namespace graph_mining::in_memory
