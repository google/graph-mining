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

#include "in_memory/clustering/dynamic/hac/hac_internal.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"
#include "utils/status/thread_safe_status.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

using NodeColor = DynamicHacNodeColor::NodeColor;
using graph_mining::in_memory::NodeId;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using Dendrogram = graph_mining::in_memory::Dendrogram;
using SimpleUndirectedGraph = graph_mining::in_memory::SimpleUndirectedGraph;
using PriorityType = DynamicHacNodeColorBase::PriorityType;
using WeightType = DynamicClusteredGraph::Weight::StoredWeightType;

namespace {

// Return the target of `node_id` in `partition_map`. It returns an error status
// if `node_id` is not in `partition_map`.
absl::StatusOr<NodeId> NodeTarget(
    const NodeId node_id,
    const absl::flat_hash_map<NodeId, NodeId>& partition_map) {
  auto it = partition_map.find(node_id);
  if (it == partition_map.end())
    return absl::FailedPreconditionError("node not in partition map, id = " +
                                         std::to_string(node_id));
  const NodeId& target = it->second;
  return target;
}

// Return true if the tuple `(p1, id1)` is larger than or equal to `(p2, id2)`;
// false otherwise.
inline bool HigherOrEqualSimilarity(const WeightType p1, const WeightType p2,
                                    const NodeId id1, const NodeId id2) {
  return std::tie(p1, id1) >= std::tie(p2, id2);
}

absl::Status ValidateSubgraphHacInput(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    const std::vector<double>& min_merge_similarities, double weight_threshold,
    double epsilon, const std::vector<NodeId>& subgraph_node_map) {
  for (size_t node_id = 0; node_id < min_merge_similarities.size(); ++node_id) {
    if (min_merge_similarities[node_id] >= weight_threshold / (1 + epsilon)) {
      double best_weight = 0;
      NodeId best_neighbor = 0;
      for (const auto& [neighbor_id, similarity] : graph.Neighbors(node_id)) {
        if (best_weight < similarity) {
          best_neighbor = neighbor_id;
          best_weight = similarity;
        }
        if (similarity < 0) {
          return absl::InternalError(absl::StrCat(
              "Edge incident to node_id=", node_id, " to ", neighbor_id,
              " has weight ", similarity, " which is not strictly positive."));
        }
      }
      if (min_merge_similarities[node_id] * (1 + epsilon) + 1e-6 <
          best_weight) {
        return absl::InternalError(absl::StrCat(
            "node_id=", node_id, " min_merge_similarities[node_id]=",
            min_merge_similarities[node_id], " cluster size = ",
            graph.NodeWeight(node_id), " neighbor = ", best_neighbor,
            " cluster size = ", graph.NodeWeight(best_neighbor),
            " min_merge[neighbor] = ", min_merge_similarities[best_neighbor],
            " epsilon=", epsilon, " best_weight=", best_weight,
            " node_id in original graph = ", subgraph_node_map[node_id],
            " neighbor in original graph = ",
            subgraph_node_map[best_neighbor]));
      }
    }
  }
  return absl::OkStatus();
}

// Compute and update the partition of newly inserted nodes and their neighbors.
// Update the `partition_change[i].second` after for i in `new_nodes` and their
// neighbors in `partition_change` and `partition_map`. The neighbors of
// `new_nodes` in `graph` should be the same as the neighbors stored in
// `new_nodes`. For each red node in `new_nodes[i].id`, put it in its own
// partition, and update its blue neighbors' partitions. If this red node's edge
// weight is higher than the blue neighbor's old partition's edge weight, change
// this blue neighbor's partition to this red neighbor. For each blue node in
// `new_nodes[i].id`, set its partition to the red neighbor with highest edge
// weight. If a new blue node has no red neighbor, we put it into its own
// partition. This function updates `partition_map`. Requires that all nodes in
// `new_nodes` exist in `colors`. `partition_map[i]` must also be a node that is
// a key in `partition_map`. Each node is either mapped to a red neighbor or
// itself. Returns error status if any of the `new_nodes` is not in
// `partition_map` or any edge is not in `graph` and also not incident to
// `deleted_nodes`.
absl::Status UpdatePartitionMembershipNewNodesAndNeighbors(
    absl::Span<const AdjacencyList> new_nodes,
    const DynamicClusteredGraph& graph, const DynamicHacNodeColorBase& color,
    absl::flat_hash_map<NodeId, NodeId>& partition_map,
    absl::flat_hash_map<NodeId, std::pair<NodeId, NodeId>>& partition_change) {
  for (const auto& node : new_nodes) {
    // Initialize each node to its own partition.
    ABSL_CHECK_EQ(partition_map.insert({node.id, node.id}).second, 1);
  }

  for (const auto& node : new_nodes) {
    const auto node_size = node.weight;
    if (color.GetNodeColor(node.id) == NodeColor::kRed) {
      // For each red node, compute which blue neighbors change partition
      // membership to it.
      for (const auto& [neighbor_id, weight] : node.outgoing_edges) {
        ASSIGN_OR_RETURN(const auto neighbor_ptr,
                         graph.ImmutableNode(neighbor_id));
        // Check if `neighbor_id` is blue and if so whether it should point to
        // this red neighbor.
        if (color.GetNodeColor(neighbor_id) == NodeColor::kBlue) {
          ASSIGN_OR_RETURN(const auto old_target,
                           NodeTarget(neighbor_id, partition_map));
          // Does not have any red neighbor yet, assign it to node.
          if (old_target == neighbor_id) {
            partition_map[neighbor_id] = node.id;
            auto itr = partition_change.find(neighbor_id);
            itr->second.second = node.id;
            continue;
          }
          const auto old_similarty_value =
              neighbor_ptr->EdgeSimilarity(old_target);
          if (!old_similarty_value.has_value()) {
            return absl::InternalError(absl::StrCat(
                "Edge does not exist, ", neighbor_id, " to ", old_target));
          }
          const auto old_similarity = old_similarty_value.value();
          const auto new_similarity =
              weight / node_size / neighbor_ptr->ClusterSize();

          // We have a higher similarity, update neighbor's partition to
          // `node.id`.
          if (HigherOrEqualSimilarity(new_similarity, old_similarity, node.id,
                                      old_target)) {
            partition_map[neighbor_id] = node.id;
            auto itr = partition_change.find(neighbor_id);
            itr->second.second = node.id;
          }
        }  // end if blue
      }  // end for loop
    } else {
      // For each new blue node, check all new red neighbors, and update
      // partition if a red neighbor has a higher similarity than its current
      // target if the target is not itself. If the target is itself, the
      // partition is updated to the red neighbor. The target of a blue node is
      // the highest similarity node among its red neighbors, and, if the node
      // has no red neighbors, the node itself.
      auto current_target = node.id;
      WeightType current_similarity = std::numeric_limits<WeightType>::lowest();
      for (const auto& [neighbor_id, weight] : node.outgoing_edges) {
        // Update the target to be the red neighbor with smallest priority.
        if (color.GetNodeColor(neighbor_id) == NodeColor::kRed) {
          // Red neighbor has a larger similarity, update the partition
          // candidate of `node` to `neighbor_id`.
          ASSIGN_OR_RETURN(const auto neighbor_ptr,
                           graph.ImmutableNode(neighbor_id));

          // skip dividing node_size since all weights need to divide it.
          auto new_similarity = weight / neighbor_ptr->ClusterSize();
          if (HigherOrEqualSimilarity(new_similarity, current_similarity,
                                      neighbor_id, current_target)) {
            current_similarity = new_similarity;
            current_target = neighbor_id;
          }
        }
      }
      partition_map[node.id] = current_target;
      partition_change[node.id] = std::make_pair(-1, current_target);
    }
  }

  return absl::OkStatus();
}

// Compute and update `partition_map` of `nodes` in `graph`.
// Update the partitions of `nodes` in `partition_change` as well.
absl::Status UpdatePartitionMembership(
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>& nodes,
    const DynamicHacNodeColorBase& colors, const DynamicClusteredGraph& graph,
    absl::flat_hash_map<graph_mining::in_memory::NodeId,
                        graph_mining::in_memory::NodeId>& partition_map,
    absl::flat_hash_map<NodeId, std::pair<NodeId, NodeId>>& partition_change) {
  for (const auto& node_id : nodes) {
    // Red node does not change membership.
    if (colors.GetNodeColor(node_id) ==
        DynamicHacNodeColorBase::NodeColor::kRed) {
      continue;
    }
    auto current_target = node_id;
    WeightType current_similarity = std::numeric_limits<WeightType>::lowest();
    absl::Status status = absl::OkStatus();
    auto map_f = [&](gbbs::uintE neighbor_id, double weight) {
      // Update the target to be the red neighbor with smallest priority.
      if (colors.GetNodeColor(neighbor_id) == NodeColor::kRed) {
        // Red neighbor has a larger similarity, update the partition
        // candidate of `node` to `neighbor_id`.
        const auto neighbor_ptr_status = graph.ImmutableNode(neighbor_id);
        if (!neighbor_ptr_status.ok()) {
          status = neighbor_ptr_status.status();
          return true;
        }

        // the weight here is already the similarity.
        auto new_similarity = weight;
        if (HigherOrEqualSimilarity(new_similarity, current_similarity,
                                    neighbor_id, current_target)) {
          current_similarity = new_similarity;
          current_target = neighbor_id;
        }
      }
      return false;
    };
    ASSIGN_OR_RETURN(auto node, graph.ImmutableNode(node_id));
    node->IterateUntil(map_f);
    if (!status.ok()) return status;
    auto itr = partition_map.find(node_id);
    if (itr == partition_map.end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("node not in partition map, id = ", node_id));
    }
    partition_change[node_id].second = current_target;
    itr->second = current_target;
  }
  return absl::OkStatus();
}

// Returns the initial partition of `new_nodes`, `new_nodes`'s neighbors, and
// `neighbors_deleted` in `partition_map`. The returned result is a mapping from
// node x to x's current partition in `partition_map`, and its initial partition
// assignment before re-partition. If a node is a new node, it's current
// partition is -1, and its initial partition assignment is itself. For all
// other nodes, its initial partition assignment is the same as its current
// partition.
absl::StatusOr<absl::flat_hash_map<NodeId, std::pair<NodeId, NodeId>>>
InitializePartitionChange(
    absl::Span<const AdjacencyList> new_nodes,
    const absl::flat_hash_set<NodeId>& neighbors_deleted,
    const absl::flat_hash_map<NodeId, NodeId>& partition_map) {
  // Stores node id, partition_before, partition_after (placeholder)
  absl::flat_hash_map<NodeId, std::pair<NodeId, NodeId>> partition_change;

  for (const auto& neighbor_id : neighbors_deleted) {
    ASSIGN_OR_RETURN(const auto old_target,
                     NodeTarget(neighbor_id, partition_map));
    partition_change[neighbor_id] = {old_target, old_target};
  }
  absl::flat_hash_set<NodeId> new_nodes_set;
  for (const auto& node : new_nodes) {
    // Initialize each node to its own partition.
    partition_change[node.id] = std::make_pair(-1, node.id);
    new_nodes_set.insert(node.id);
  }
  for (const auto& node : new_nodes) {
    // For each node, store its neighbor's original partition
    for (const auto& [neighbor_id, _] : node.outgoing_edges) {
      if (new_nodes_set.contains(neighbor_id)) {
        continue;
      }
      ASSIGN_OR_RETURN(const auto old_target,
                       NodeTarget(neighbor_id, partition_map));
      partition_change[neighbor_id] = {old_target, old_target};
    }
  }
  return partition_change;
}

}  // namespace

absl::StatusOr<std::vector<std::tuple<NodeId, NodeId, NodeId>>>
UpdatePartitions(const std::vector<AdjacencyList>& new_nodes,
                 const absl::flat_hash_set<NodeId>& neighbors_deleted,
                 const DynamicClusteredGraph& graph,
                 const DynamicHacNodeColorBase& color,
                 absl::flat_hash_map<NodeId, NodeId>& partition_map) {
  ASSIGN_OR_RETURN(
      auto partition_change,
      InitializePartitionChange(new_nodes, neighbors_deleted, partition_map));

  // Re-partitions due to deletions.
  RETURN_IF_ERROR(UpdatePartitionMembership(neighbors_deleted, color, graph,
                                            partition_map, partition_change));

  RETURN_IF_ERROR(UpdatePartitionMembershipNewNodesAndNeighbors(
      new_nodes, graph, color, partition_map, partition_change));

  std::vector<std::tuple<NodeId, NodeId, NodeId>> nodes_changed;
  for (const auto& [k, v] : partition_change) {
    nodes_changed.push_back(std::make_tuple(k, v.first, v.second));
  }

  return nodes_changed;
}

absl::StatusOr<absl::flat_hash_set<NodeId>> DirtyPartitions(
    absl::Span<const std::tuple<NodeId, NodeId, NodeId>> nodes_changed,
    const DynamicClusteredGraph& graph, const DynamicHacNodeColorBase& color) {
  absl::flat_hash_set<NodeId> dirty_partition;

  for (const auto& [u, p_before, p_after] : nodes_changed) {
    if (p_before != -1 && graph.ContainsNode(p_before)) {
      // If a node changes from a singleton blue partition to a red partition,
      // the previous singleton blue partition is not a valid partition anymore
      // and is not returned. But if it stays as a singleton blue partition, it
      // is still dirty.
      if (color.GetNodeColor(p_before) != NodeColor::kBlue ||
          p_before == p_after) {
        dirty_partition.insert(p_before);
      }
    }
    dirty_partition.insert(p_after);
  }

  return dirty_partition;
}

absl::StatusOr<SubgraphHacResults> RunSubgraphHac(
    std::unique_ptr<SimpleUndirectedGraph>& partition_graph,
    const SubgraphMinMergeSimilarity& min_merge_similarities_partition_map,
    const double epsilon) {
  auto partition_num_nodes = partition_graph->NumNodes();

  // Create min_merge_similarities of the partition.
  std::vector<double> min_merge_similarities_partition(partition_num_nodes);
  graph_mining::ThreadSafeStatus status;
  parlay::parallel_for(0, partition_num_nodes, [&](std::size_t j) {
    if (status.status().ok()) {
      auto local_status = min_merge_similarities_partition_map(j);
      if (local_status.ok()) {
        min_merge_similarities_partition[j] = local_status.value();
      } else {
        status.Update(local_status.status());
      }
    }
  });
  RETURN_IF_ERROR(status.status());

  // Run SubGraphHAC.
  RETURN_IF_ERROR(ValidateSubgraphHacInput(
      *partition_graph, min_merge_similarities_partition, 0, epsilon,
      min_merge_similarities_partition_map.NodeMap()));

  return ApproximateSubgraphHac(std::move(partition_graph),
                                std::move(min_merge_similarities_partition),
                                epsilon);
}

absl::StatusOr<std::vector<double>> LocalMinMergeSimilarities(
    absl::Span<const std::tuple<NodeId, NodeId, double>> merges,
    const SubgraphMinMergeSimilarity& min_merge_similarities_partition_map,
    const std::size_t partition_num_nodes) {
  // Initialize to before SubgraphHac.
  std::vector<double> min_merge_similarities(
      2 * partition_num_nodes - 1, std::numeric_limits<double>::infinity());
  graph_mining::ThreadSafeStatus status;
  parlay::parallel_for(0, partition_num_nodes, [&](std::size_t j) {
    auto local_status = min_merge_similarities_partition_map(j);
    if (local_status.ok()) {
      min_merge_similarities[j] = local_status.value();
    } else {
      status.Update(local_status.status());
    }
  });
  RETURN_IF_ERROR(status.status());
  NodeId internal_node_id = partition_num_nodes;
  for (const auto [u, v, w] : merges) {
    min_merge_similarities[internal_node_id] = std::min(
        w, std::min(min_merge_similarities[u], min_merge_similarities[v]));
    ++internal_node_id;
  }

  return min_merge_similarities;
}

std::vector<NodeId> LeafToRootId(
    const graph_mining::in_memory::Dendrogram& dendrogram) {
  const auto num_leaves = dendrogram.NumClusteredNodes();
  const auto& nodes = dendrogram.Nodes();
  std::vector<NodeId> root(nodes.size());

  for (NodeId i = nodes.size() - 1; i >= 0; --i) {
    if (!dendrogram.HasValidParent(i)) {
      // i is a root node
      root[i] = i;
    } else {
      root[i] = root[nodes[i].parent_id];
    }
  }
  root.resize(num_leaves);
  return root;
}

}  // namespace graph_mining::in_memory
