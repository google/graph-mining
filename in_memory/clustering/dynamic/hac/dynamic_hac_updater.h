/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_HAC_UPDATER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_HAC_UPDATER_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

class NextUnusedId {
 public:
  explicit NextUnusedId(graph_mining::in_memory::NodeId next_unused_node_id)
      : next_unused_node_id_(next_unused_node_id) {}
  graph_mining::in_memory::NodeId operator()(
      graph_mining::in_memory::NodeId num = 1) {
    graph_mining::in_memory::NodeId start = next_unused_node_id_;
    next_unused_node_id_ += num;
    return start;
  }

 private:
  graph_mining::in_memory::NodeId next_unused_node_id_;
};

// Map from local node id to the global node id that it merges to. Local node id
// are some consecutive node ids that corresponds to some non-consecutive global
// node ids. The local cluster ids correspond to the clusters formed by the
// nodes, and are also consecutive with the local node ids.
class SubgraphClusterId {
  using NodeId = graph_mining::in_memory::NodeId;

 public:
  // `local_to_global_id` maps from local node id and local cluster ids to
  // global node id. `to_cluster_id` maps from local node ids to local cluster
  // ids.
  SubgraphClusterId(const std::vector<NodeId> local_to_global_id,
                    const std::vector<NodeId> to_cluster_id)
      : local_to_global_id_(local_to_global_id),
        to_cluster_id_(to_cluster_id) {}

  // Returns `local_to_global_id_[to_cluster_id_[local_id]]`.
  NodeId operator()(NodeId local_id) const {
    const auto local_root = to_cluster_id_[local_id];
    ABSL_CHECK_LT(local_root, local_to_global_id_.size());
    const auto global_root = local_to_global_id_[local_root];
    return global_root;
  }

  NodeId LocalClusterId(NodeId local_id) const {
    return to_cluster_id_[local_id];
  }

  const std::vector<NodeId>& LocalToGlobalId() const {
    return local_to_global_id_;
  }

 private:
  // Maps from local node id to global node id.
  const std::vector<NodeId> local_to_global_id_;
  // Map from local node id to the local cluster id that it merges to.
  const std::vector<NodeId> to_cluster_id_;
};

// The inputs are:
//  * `merges`: The merges we want the updated `dynamic_dendrogram` to have.
//    `merges` are in local cluster id space. The local cluster ids are
//     consecutive and the internal node resulting from the ith merge will have
//     node id `n-1+i`.
//  * `subgraph_node_map`: The mapping from local node id to global node id.
//  * `next_unused_id`: gives an unused global node id available for an internal
//     node to use.
// The function does the following:
//  * Updates `dynamic_dendrogram` according to `merges`
//  * Appends to `subgraph_node_map` the mapping of newly merged nodes in
//    `merges`. Let the input `subgraph_node_map`.size() be n, the
//    `subgraph_node_map[n-1+i]` will correspond to the internal node resulting
//    from the ith merge in `merges`. The new internal nodes will be given node
//    ids based on `next_unused_id`. "New internal nodes" are nodes that result
//    from `merges` that are different from merges in the current dendrogram.
absl::Status UpdateDendrogram(
    const std::vector<std::tuple<graph_mining::in_memory::NodeId,
                                 graph_mining::in_memory::NodeId, double>>&
        merges,
    std::vector<graph_mining::in_memory::NodeId>& subgraph_node_map,
    NextUnusedId& next_unused_id, DynamicDendrogram& dynamic_dendrogram);

// Return current mappings of `deleted_nodes` and `active_nodes` in
// `next_round_node_map`  in (local id, global id, current mapping to
// global id next round) format. Returns error status if any node is not in
// `next_round_node_map`. For deleted nodes, local id is -1. For new nodes,
// current mapping is -1. A node is a new node if it is contained in
// `new_nodes`.
absl::StatusOr<std::vector<
    std::tuple<graph_mining::in_memory::NodeId, graph_mining::in_memory::NodeId,
               graph_mining::in_memory::NodeId>>>
CurrentMapNextRound(
    absl::Span<const graph_mining::in_memory::NodeId> active_nodes,
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>& deleted_nodes,
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>& new_nodes,
    const absl::flat_hash_map<graph_mining::in_memory::NodeId,
                              graph_mining::in_memory::NodeId>&
        next_round_node_map);

// Returns nodes to delete from next round, updates `next_round_node_map`.
// `current_mapping` contains the mapping of node set V in this round, to nodes
// in next round. It has format [local_id, global_id, current_mapped_global_id].
// If a node is deleted, it has local_id -1. `subgraph_cluster_id(i)` gives the
// global cluster id of node i in hac. If a node's current_mapped_global_id
// satisfies both of the two conditions below, it will be in the returned set.
//    1) not equal to subgraph_cluster_id(i). This means i is mapped to a
//       different node or deleted now, so the old mapped nodes
//       need to be deleted.
//    2) is not in `active_contracted_nodes`. If current_mapped_global_id is in
//       `active_contracted_nodes`, it is re-merged in this round, so it should
//       not be deleted.
// We update `next_round_node_map` to map from global_id to
// `subgraph_cluster_id(i)`.
// `root_map` maps from global id to global root in SubgraphHac.
absl::StatusOr<absl::flat_hash_set<graph_mining::in_memory::NodeId>>
NodesToDelete(
    absl::Span<const std::tuple<graph_mining::in_memory::NodeId,
                                graph_mining::in_memory::NodeId,
                                graph_mining::in_memory::NodeId>>
        current_mapping,
    const absl::flat_hash_map<graph_mining::in_memory::NodeId,
                              graph_mining::in_memory::NodeId>& root_map,
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>&
        active_contracted_nodes,
    absl::flat_hash_map<graph_mining::in_memory::NodeId,
                        graph_mining::in_memory::NodeId>& next_round_node_map);

// Returns the mapping of nodes from the current last round to next round.
// `active_nodes` are the global nodes that we want to add to the mapping. For i
// in [1 ... active_nodes.size()], we add the following to
// `next_round_node_map`: `next_round_node_map[active_nodes[i]] = root_map(i)`.
absl::StatusOr<absl::flat_hash_map<graph_mining::in_memory::NodeId,
                                   graph_mining::in_memory::NodeId>>
MappingLastRound(
    absl::Span<const graph_mining::in_memory::NodeId> active_nodes,
    const absl::flat_hash_map<graph_mining::in_memory::NodeId,
                              graph_mining::in_memory::NodeId>& root_map);

// Returns the new edges that should be added to the next snapshot.
// Specifically, it returns all edges of `contracted_graph` incident to
// `new_nodes`. The edge weights to inactive nodes are multiplied by the weight
// of the node in `graph`, because the unnormalized edge weights in
// `contracted_graph` do not take into account the weight of inactive nodes;
// the weight of inactive nodes is -1 in `contracted_graph`. The returned edges
// use node ids mapped by `subgraph_cluster_id`. `new_nodes` are new nodes we
// want to insert to the next round, in local node id.
// If a node is mapped to -1 in `next_round_node_map`, it is ignored and any
// edge adjacent to it is not returned. Only inactive nodes can possibly map to
// -1. `new_nodes` should not contain -1 node ids.
// Returns error status if any of the node in the return value is in
// `delete_nodes`. `delete_nodes` is used for validating the output only.
absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
AdjacencyListsOfNewNodes(
    const std::vector<graph_mining::in_memory::NodeId>& new_nodes,
    const SubgraphClusterId& subgraph_cluster_id,
    const std::unique_ptr<ContractedGraph>& contracted_graph,
    const absl::flat_hash_map<graph_mining::in_memory::NodeId,
                              graph_mining::in_memory::NodeId>&
        next_round_node_map,
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>& deleted_nodes,
    const DynamicClusteredGraph& graph);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_HAC_UPDATER_H_
