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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac.pb.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac_updater.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

class DynamicHacClusterer {
 public:
  // Represents clustering: each element of the vector contains the set of
  // NodeIds in one cluster.
  using Clustering = graph_mining::in_memory::Clustering;

  // Represents the type of node ids
  using NodeId = graph_mining::in_memory::NodeId;

  // Initializes the clustering parameters using `config`. The clusterer can
  // only be initialized once. `weight_threshold` in `config` is required.
  explicit DynamicHacClusterer(const DynamicHacConfig& config) {
    epsilon_ = config.epsilon();
    ABSL_CHECK(config.has_weight_threshold())
        << "weight_threshold is required.";
    weight_threshold_ = config.weight_threshold();
  }

  // Requires weight_threshold to be set.
  DynamicHacClusterer() = delete;

  virtual ~DynamicHacClusterer() = default;

  struct UpdateStats {
    std::size_t dirty_partitions = 0;
    std::size_t dirty_nodes = 0;
    std::size_t dirty_edges = 0;
    std::size_t nodes_ignored = 0;

    UpdateStats() = default;

    UpdateStats(std::size_t dirty_partitions, std::size_t dirty_nodes,
                std::size_t dirty_edges, std::size_t nodes_ignored)
        : dirty_partitions(dirty_partitions),
          dirty_nodes(dirty_nodes),
          dirty_edges(dirty_edges),
          nodes_ignored(nodes_ignored) {}

    void LogStats() const {
      LOG(INFO) << "Num. Dirty Partitions: " << dirty_partitions;
      LOG(INFO) << "Num. Dirty Nodes: " << dirty_nodes;
      LOG(INFO) << "Num. Dirty Edges: " << dirty_edges;
      LOG(INFO) << "Num. Ignored Nodes: " << nodes_ignored;
    }
  };

  // Returns the flat clustering from the current dendrogram.
  absl::StatusOr<Clustering> FlatCluster(double cut_threshold) const;

  // Returns the largest available node id the clusterer can accept.
  NodeId LargestAvailableNodeId() const { return kLargestAvailableNodeId; }

  DynamicDendrogram Dendrogram() const { return dendrogram_; }

  size_t NumRounds() const { return clustered_graphs_.size(); }

  size_t NumEdges() const { return clustered_graphs_[0].NumEdges(); }

  // Inserts new nodes with edges. It returns an error if 1) the incident edges
  // contain node ids not already in the graph and not in the provided `nodes`.
  // 2) any node in `nodes` is already present in the graph 3) any id in `nodes`
  // is too large or negative. The graph will only have node ids that are
  // provided by the user.
  absl::StatusOr<UpdateStats> Insert(
      const std::vector<
          graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>&
          nodes) {
    return Modify(nodes, {});
  }

  // Removes `nodes` from the current graph. It returns an error if any of
  // `nodes` is not already in the graph.
  absl::StatusOr<UpdateStats> Remove(std::vector<NodeId> nodes) {
    return Modify({}, nodes);
  }

  // For debug and test purpose only. Set the `colors_` of the clusterer.
  void SetColors(
      std::vector<std::unique_ptr<DynamicHacNodeColorTest>>& colors) {
    for (auto& color : colors) {
      colors_.push_back(std::move(color));
    }
  }

 private:
  // Inserts new nodes or delete existing nodes. Cannot insert and delete in
  // the same call.
  // TODO : modify to allow both insertions and deletions at the same
  // time.
  absl::StatusOr<UpdateStats> Modify(
      const std::vector<
          graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>&
          inserts = {},
      absl::Span<const NodeId> deletes = {});

  // Returns one available node id and block the next `num` node ids from being
  // returned. `num` is default to 1. Any node id will only be given once. Ids
  // of deleted nodes are not re-used.
  NextUnusedId next_unused_node_id_ = NextUnusedId(kLargestAvailableNodeId + 1);

  // Adds a new round in the clusterer.
  void AddRound();

  // Removes rounds > `round`. Rounds are 0-based.
  void ClearRounds(std::size_t round);

  // Returns the number of rounds stored.
  std::size_t RoundNumber() const { return clustered_graphs_.size(); }

  // The largest available leaf node id.
  static constexpr NodeId kLargestAvailableNodeId =
      std::numeric_limits<NodeId>::max() / 2;

  double epsilon_;

  double weight_threshold_;

  // Stores the graph at each round.
  std::vector<DynamicClusteredGraph> clustered_graphs_;

  // The partition membership of each node in each round.
  // `partition_memberships_[i]` maps nodes of `clustered_graphs_[i]` to
  // partitions of level i.
  std::vector<absl::flat_hash_map<NodeId, NodeId>> partition_memberships_;

  // Stores the color and color priority of each node at each level. The same
  // node may have different color and priority across different rounds.
  // `colors_[i]` gives the color and priority at round i. A blue node will be
  // assigned to the partition of the red neighbor with lowest priority. More
  // detailed description of the color assignment can be found in the comment
  // for ComputePartitionMembership in
  // third_party/graph_mining/in_memory/clustering/dynamic/hac/hac_internal.h.
  std::vector<std::unique_ptr<DynamicHacNodeColorBase>> colors_;

  // Stores the min merge similarities of each round.
  std::vector<absl::flat_hash_map<NodeId, double>> min_merge_similarities_;

  // Stores the mapping from nodes in the current round to which node they merge
  // to in the next round. If a node is ignored because it does not have any
  // heavy edge, it is mapped to -1.
  std::vector<absl::flat_hash_map<NodeId, NodeId>> next_round_node_maps_;

  DynamicDendrogram dendrogram_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_H_
