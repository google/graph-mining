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

#include "in_memory/clustering/dynamic/hac/hac.h"

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
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac_updater.h"
#include "in_memory/clustering/dynamic/hac/hac_internal.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"
#include "utils/timer.h"

using graph_mining::in_memory::Clustering;
using graph_mining::in_memory::NodeId;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using UpdateStats = graph_mining::in_memory::DynamicHacClusterer::UpdateStats;

namespace graph_mining::in_memory {

namespace {
// Update `min_merge_similarities_next`. For all local cluster ids i in
// `active_contracted_nodes_local`, update `min_merge_similarities_next` to map
// from global_id of i to `min_merge_similarities_local`[i].
// `min_merge_similarities_local` is in local cluster id space.
// `local_to_global_id` maps from local cluster id space to global id space.
void UpdateMinMergeSimilarities(
    absl::Span<const NodeId> active_contracted_nodes_local,
    absl::Span<const NodeId> local_to_global_id,
    absl::Span<const double> min_merge_similarities_local,
    absl::flat_hash_map<NodeId, double>& min_merge_similarities_next) {
  for (NodeId cluster_i : active_contracted_nodes_local) {
    const auto global_id = local_to_global_id[cluster_i];
    min_merge_similarities_next[global_id] =
        min_merge_similarities_local[cluster_i];
  }
}

// Process `dirty_partition` in `graph`. Update the following:
// * `min_merge_similarities_next`: min merge similarities in the next round
// * `active_nodes`: in global id space. The active nodes that are potentially
//    merged.
// * `ignored_nodes`: in global id space. Nodes that are ignored because they do
//    not have any heavy edge.
// * `dendrogram`: the dynamic dendrogram
// * `next_unused_node_id`: the next available internal node id
// * `active_contracted_nodes`: in global id space. The merged root nodes.
// * `root_map`: in global id space. Map from nodes to their root in the
//    subgraph hac.
absl::StatusOr<std::pair<SubgraphHacResults, SubgraphClusterId>>
ProcessDirtyPartition(
    const DynamicClusteredGraph& graph, const NodeId& dirty_partition,
    const absl::flat_hash_map<NodeId, NodeId>& partition_memberships,
    const absl::flat_hash_map<NodeId, double>& min_merge_similarities,
    absl::flat_hash_map<NodeId, double>& min_merge_similarities_next,
    const double epsilon, std::vector<NodeId>& active_nodes,
    std::vector<NodeId>& ignored_nodes, DynamicDendrogram& dendrogram,
    NextUnusedId& next_unused_node_id,
    absl::flat_hash_set<NodeId>& active_contracted_nodes,
    absl::flat_hash_map<NodeId, NodeId>& root_map,
    std::size_t& total_num_dirty_edges, std::size_t& total_num_dirty_nodes) {
  // Create subgraph and run SubgraphHac.
  ASSIGN_OR_RETURN(auto subgraph, graph.CreateSubgraph({dirty_partition},
                                                       partition_memberships));
  auto& [partition_graph, local_to_global_id, num_active_nodes,
         local_ignored_nodes] = *subgraph;
  total_num_dirty_edges += partition_graph->NumDirectedEdges();
  total_num_dirty_nodes += partition_graph->NumNodes();
  active_nodes.insert(active_nodes.end(), local_to_global_id.begin(),
                      local_to_global_id.begin() + num_active_nodes);
  ignored_nodes.insert(ignored_nodes.end(), local_ignored_nodes.begin(),
                       local_ignored_nodes.end());

  auto const min_merge_similarities_partition_map =
      SubgraphMinMergeSimilarity(local_to_global_id, min_merge_similarities);
  ASSIGN_OR_RETURN(
      auto subgraph_hac_results,
      RunSubgraphHac(partition_graph, min_merge_similarities_partition_map,
                     epsilon));

  // Postprocess subgraph hac results and update dendrogram.
  const auto& to_cluster_ids = LeafToRootId(subgraph_hac_results.dendrogram);
  ASSIGN_OR_RETURN(
      const auto& min_merge_similarities_local,
      LocalMinMergeSimilarities(subgraph_hac_results.merges,
                                min_merge_similarities_partition_map,
                                local_to_global_id.size()));
  RETURN_IF_ERROR(UpdateDendrogram(subgraph_hac_results.merges,
                                   local_to_global_id, next_unused_node_id,
                                   dendrogram));

  std::vector<NodeId>
      active_nodes_local;  // Root nodes of active nodes, in local cluster id.
  const auto& subgraph_cluster_id =
      SubgraphClusterId(std::move(local_to_global_id), to_cluster_ids);

  // Update active nodes and contracted nodes we store.
  for (NodeId i = 0; i < num_active_nodes; ++i) {
    const NodeId global_id = subgraph_cluster_id.LocalToGlobalId()[i];
    const auto local_root = subgraph_cluster_id.LocalClusterId(i);
    const auto global_root = subgraph_cluster_id.LocalToGlobalId()[local_root];
    root_map[global_id] = global_root;
    active_contracted_nodes.insert(global_root);
    active_nodes_local.push_back(local_root);
  }

  // Update min merge similarities
  UpdateMinMergeSimilarities(
      active_nodes_local, subgraph_cluster_id.LocalToGlobalId(),
      min_merge_similarities_local, min_merge_similarities_next);
  return std::make_pair(std::move(subgraph_hac_results), subgraph_cluster_id);
}

// Map ignored nodes to -1 and delete its mapping from future graph snapshots.
absl::Status ProcessIgnoredNodes(
    const std::vector<NodeId>& ignored_nodes,
    const absl::flat_hash_set<NodeId>& new_nodes_set,
    absl::flat_hash_map<NodeId, NodeId>& next_round_node_map,
    absl::flat_hash_set<NodeId>& nodes_to_delete) {
  // `nodes_to_delete` might contain -1 mapped nodes from NodesToDelete().
  nodes_to_delete.erase(-1);
  for (NodeId global_id : ignored_nodes) {
    if (new_nodes_set.contains(global_id)) {
      // We treat `global_id` specially if it's newly inserted. It does not
      // exist in the map yet and it does not have an old mapping.
      next_round_node_map[global_id] = -1;
    } else {
      auto itr = next_round_node_map.find(global_id);
      if (itr == next_round_node_map.end()) {
        return absl::NotFoundError(
            absl::StrCat("node not in next_round_node_map, ", global_id));

      } else {
        const auto old_mapped_id = itr->second;
        if (old_mapped_id != -1) {
          nodes_to_delete.insert(old_mapped_id);
        }
        next_round_node_map[global_id] = -1;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Clustering> DynamicHacClusterer::FlatCluster(
    double cut_threshold) const {
  if (cut_threshold < weight_threshold_) {
    return absl::InvalidArgumentError(
        "cut_threshold must be greater than weight_threshold");
  }
  ASSIGN_OR_RETURN(const auto& result,
                   dendrogram_.ConvertToParallelDendrogram());
  const auto& [dendrogram, node_ids] = result;
  const auto dense_clustering = dendrogram.GetSubtreeClustering(cut_threshold);
  auto clustering = ClusterIdsToClustering(dense_clustering);
  // Map node ids to dendrogram's leaf ids.
  for (auto& cluster : clustering) {
    for (std::size_t i = 0; i < cluster.size(); ++i) {
      cluster[i] = node_ids[cluster[i]];
    }
  }
  return clustering;
}

void DynamicHacClusterer::AddRound() {
  const NodeId round = clustered_graphs_.size();
  clustered_graphs_.push_back(
      DynamicClusteredGraph(weight_threshold_ / (1 + epsilon_)));
  partition_memberships_.push_back(absl::flat_hash_map<NodeId, NodeId>());
  colors_.push_back(
      std::make_unique<DynamicHacNodeColor>(DynamicHacNodeColor(round + 1)));
  min_merge_similarities_.push_back(absl::flat_hash_map<NodeId, double>());
  next_round_node_maps_.push_back(absl::flat_hash_map<NodeId, NodeId>());
}

void DynamicHacClusterer::ClearRounds(std::size_t round) {
  size_t num_rounds = clustered_graphs_.size();
  for (std::size_t i = round + 1; i < num_rounds; ++i) {
    clustered_graphs_.pop_back();
    partition_memberships_.pop_back();
    colors_.pop_back();
    min_merge_similarities_.pop_back();
    next_round_node_maps_.pop_back();
  }
}

absl::StatusOr<UpdateStats> DynamicHacClusterer::Modify(
    const std::vector<AdjacencyList>& inserts,
    absl::Span<const NodeId> deletes) {
  if (!inserts.empty() && !deletes.empty()) {
    return absl::InvalidArgumentError(
        "inserts and deletes cannot happen at the same time.");
  }
  std::size_t total_num_dirty_partitions = 0;
  std::size_t total_num_dirty_nodes = 0;
  std::size_t total_num_dirty_edge = 0;
  std::size_t total_num_nodes_ignored = 0;

  WallTimer total_timer;
  total_timer.Restart();
  WallTimer timer;
  timer.Restart();
  std::vector<AdjacencyList> new_nodes = inserts;
  // We need at least two rounds (they can be empty).
  if (RoundNumber() == 0) {
    AddRound();
  }
  size_t round = 0;  // The round we are processing.
  const size_t last_round = clustered_graphs_.size() - 1;

  // Stores the dirty nodes that need to be deleted from each level.
  absl::flat_hash_set<NodeId> nodes_to_delete(deletes.begin(), deletes.end());

  // Initialize `min_merge_similarties_` and dendrogram.
  for (const auto& node : new_nodes) {
    min_merge_similarities_[round][node.id] =
        std::numeric_limits<double>::infinity();
    RETURN_IF_ERROR(dendrogram_.AddLeafNode(node.id));
  }

  ABSL_VLOG(0) << "Initialize time: " << timer.GetSeconds() << " seconds";
  timer.Restart();

  int empty_rounds = 0;
  int consecutive_empty_rounds = 0;
  auto log_time = [&](std::string msg) {
    ABSL_VLOG(0) << msg << " Time: " << timer.GetSeconds() << " seconds";
    timer.Restart();
  };

  while (true) {
    WallTimer round_timer;
    round_timer.Restart();
    if (round + 1 >= RoundNumber()) {
      AddRound();
    }

    ABSL_VLOG(0) << "=== Running round: " << round;
    ABSL_VLOG(0) << "=== Num inserted nodes: " << new_nodes.size();
    ABSL_VLOG(0) << "=== Num deleted nodes: " << nodes_to_delete.size();
    ABSL_VLOG(0) << "=== Num total nodes: "
                 << clustered_graphs_[round].NumNodes();

    // `neighbors_deleted` are neighbors of deleted nodes that themselves are
    // not deleted. We will re-partition them and compute dirty partitions from
    // them.
    ASSIGN_OR_RETURN(auto neighbors_deleted,
                     clustered_graphs_[round].Neighbors(nodes_to_delete));
    absl::erase_if(neighbors_deleted,
                   [&](auto k) { return nodes_to_delete.contains(k); });

    // Update graph.
    RETURN_IF_ERROR(clustered_graphs_[round].AddNodes(new_nodes, false));
    for (const auto& node : nodes_to_delete) {
      RETURN_IF_ERROR(clustered_graphs_[round].RemoveNode(node));
    }

    log_time("Update Graph");

    // Find partitions that got dirty due to insertions, deletions, and
    // partition membership change. Update partition.
    ASSIGN_OR_RETURN(
        const auto& partition_update,
        UpdatePartitions(new_nodes, neighbors_deleted, clustered_graphs_[round],
                         *colors_[round], partition_memberships_[round]));
    log_time("Update Partitions");

    ASSIGN_OR_RETURN(absl::flat_hash_set<NodeId> dirty_partitions,
                     DirtyPartitions(partition_update, clustered_graphs_[round],
                                     *colors_[round]));

    for (const auto& node : nodes_to_delete) {
      partition_memberships_[round].erase(node);
      min_merge_similarities_[round].erase(node);
    }

    // There is no edge in graph with weight >= `weight_threshold_ /
    // (1+epsilon_)`, so we are done.
    if (!clustered_graphs_[round].HasHeavyEdges()) {
      next_round_node_maps_[round].clear();
      ClearRounds(round);
      // Remove the ancestors of nodes in the last graph and nodes deleted.
      std::vector<NodeId> cleanup_nodes = clustered_graphs_[round].Nodes();
      for (const auto& node : deletes) {
        cleanup_nodes.push_back(node);
      }
      absl::Span<const NodeId> cleanup_nodes_span =
          absl::Span<const NodeId>(cleanup_nodes.data(), cleanup_nodes.size());
      ASSIGN_OR_RETURN(const auto dirty_ancestors,
                       dendrogram_.RemoveAncestors(cleanup_nodes_span));
      for (const auto& node : deletes) {
        RETURN_IF_ERROR(dendrogram_.RemoveSingletonLeafNode(node));
      }
      ABSL_VLOG(0) << "Total time: " << total_timer.GetSeconds() << " seconds";
      ABSL_VLOG(0) << "Empty rounds: " << empty_rounds << "\n";
      break;
    }

    // We are at the last round, but clustering is not finished yet. All nodes
    // need to be inserted into the next round, so all partitions are dirty.
    if (clustered_graphs_[round + 1].NumNodes() == 0) {
      for (const auto& [k, v] : partition_memberships_[round]) {
        if (!clustered_graphs_[round].ContainsNode(k)) {
          return absl::InternalError(absl::StrCat(
              "Node in partition_memberships_ but not in graph, node = ", k));
        }
        dirty_partitions.insert(v);
      }
    }

    // Run HAC on dirty partitions.
    std::vector<std::pair<SubgraphHacResults, SubgraphClusterId>>
        subgraph_hac_result_vec;
    std::vector<NodeId> active_nodes_vec;
    // Nodes ignored because they do not have any heavy edges.
    std::vector<NodeId> ignored_nodes_vec;
    absl::flat_hash_set<NodeId> active_contracted_nodes;
    // Map from active nodes to root node, in global id space.
    absl::flat_hash_map<NodeId, NodeId> root_map;
    int total_num_merges = 0;
    total_num_dirty_partitions += dirty_partitions.size();

    // TODO : parallelize the for loop. All parts are trivially
    // parallel, except UpdateDendrogram.
    for (const auto& dirty_partition : dirty_partitions) {
      ASSIGN_OR_RETURN(
          auto result,
          ProcessDirtyPartition(
              clustered_graphs_[round], dirty_partition,
              partition_memberships_[round], min_merge_similarities_[round],
              min_merge_similarities_[round + 1], epsilon_, active_nodes_vec,
              ignored_nodes_vec, dendrogram_, next_unused_node_id_,
              active_contracted_nodes, root_map, total_num_dirty_edge,
              total_num_dirty_nodes));
      total_num_merges += result.first.merges.size();
      subgraph_hac_result_vec.push_back(std::move(result));
    }
    if (total_num_merges == 0) {
      if (consecutive_empty_rounds++ >
          std::max(static_cast<std::size_t>(50), 2 * last_round)) {
        return absl::InternalError(
            absl::StrCat("too many empty rounds, number of empty rounds: ",
                         consecutive_empty_rounds));
      }
    } else {
      consecutive_empty_rounds = 0;
    }
    log_time("Process Dirty Partitions");
    total_num_nodes_ignored += ignored_nodes_vec.size();

    // Compute nodes to delete from next round and update next_round_node_map.
    absl::Span<const NodeId> active_nodes = absl::Span<const NodeId>(
        active_nodes_vec.data(), active_nodes_vec.size());
    if (round < last_round) {
      absl::flat_hash_set<NodeId> new_nodes_set;
      for (const auto& node : new_nodes) {
        new_nodes_set.insert(node.id);
      }
      ASSIGN_OR_RETURN(
          const auto& current_mapping,
          CurrentMapNextRound(active_nodes, nodes_to_delete, new_nodes_set,
                              next_round_node_maps_[round]));
      for (const auto& node : nodes_to_delete) {
        next_round_node_maps_[round].erase(node);
      }
      nodes_to_delete.clear();
      ASSIGN_OR_RETURN(
          nodes_to_delete,
          NodesToDelete(current_mapping, root_map, active_contracted_nodes,
                        next_round_node_maps_[round]));
      // Map ignored nodes to -1 and delete its mapping from future graph
      // snapshots.
      RETURN_IF_ERROR(ProcessIgnoredNodes(ignored_nodes_vec, new_nodes_set,
                                          next_round_node_maps_[round],
                                          nodes_to_delete));
    } else {
      nodes_to_delete.clear();
      ASSIGN_OR_RETURN(next_round_node_maps_[round],
                       MappingLastRound(active_nodes, root_map));
      for (NodeId global_id : ignored_nodes_vec) {
        next_round_node_maps_[round][global_id] = -1;
      }
    }

    // Compute nodes and edges to insert to next round.
    std::vector<AdjacencyList> new_adjacency_lists;
    for (const auto& [subgraph_hac_results, subgraph_cluster_id] :
         subgraph_hac_result_vec) {
      const auto& contracted_graph = subgraph_hac_results.contracted_graph;
      std::vector<NodeId> local_new_nodes;
      for (NodeId i = 0; i < contracted_graph->NumNodes(); ++i) {
        if (contracted_graph->IsInactive(i)) continue;
        const NodeId global_id = subgraph_cluster_id(i);
        if (clustered_graphs_[round + 1].ContainsNode(global_id)) continue;
        local_new_nodes.push_back(i);
      }
      ASSIGN_OR_RETURN(const auto& new_edges_local,
                       AdjacencyListsOfNewNodes(
                           local_new_nodes, subgraph_cluster_id,
                           contracted_graph, next_round_node_maps_[round],
                           nodes_to_delete, clustered_graphs_[round]));
      new_adjacency_lists.insert(new_adjacency_lists.end(),
                                 new_edges_local.begin(),
                                 new_edges_local.end());
    }
    new_nodes = std::move(new_adjacency_lists);
    log_time("Compute New Edges");

    round += 1;
    ABSL_VLOG(0) << "Round time: " << round_timer.GetSeconds() << " seconds";
    ABSL_VLOG(0) << "======\n";
  }  //  end while loop
  return UpdateStats(total_num_dirty_partitions, total_num_dirty_nodes,
                     total_num_dirty_edge, total_num_nodes_ignored);
}

}  // namespace graph_mining::in_memory
