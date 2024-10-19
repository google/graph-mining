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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_UTIL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_UTIL_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "gbbs/graph.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_graph_utils.h"

namespace graph_mining::in_memory {

// Returns true if all node parts are either 0 or 1. Returns false otherwise.
bool IsValidBipartiteNodeParts(const std::vector<NodePartId>& node_parts);

// This class encapsulates the data needed to compute and maintain the
// correlation clustering objective.
class ClusteringHelper {
 public:
  using ClusterId = gbbs::uintE;

  ClusteringHelper(
      InMemoryClusterer::NodeId num_nodes,
      const graph_mining::in_memory::ClustererConfig& clusterer_config,
      const InMemoryClusterer::Clustering& clustering,
      const std::vector<NodePartId>& node_parts)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(num_nodes, 1) {
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      ABSL_CHECK_EQ(node_parts.size(), num_nodes);
      node_parts_ = node_parts;
      ABSL_CHECK(IsValidBipartiteNodeParts(node_parts_));
      partitioned_cluster_weights_ =
          std::vector<std::array<double, 2>>(num_nodes, {0, 0});
    } else {
      cluster_weights_ = std::vector<double>(num_nodes, 0);
    }
    SetClustering(clustering);
  }

  ClusteringHelper(
      InMemoryClusterer::NodeId num_nodes,
      const graph_mining::in_memory::ClustererConfig& clusterer_config,
      std::vector<double> node_weights,
      const InMemoryClusterer::Clustering& clustering,
      const std::vector<NodePartId>& node_parts)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)) {
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      ABSL_CHECK_EQ(node_parts.size(), num_nodes);
      node_parts_ = node_parts;
      ABSL_CHECK(IsValidBipartiteNodeParts(node_parts_));
      partitioned_cluster_weights_ =
          std::vector<std::array<double, 2>>(num_nodes, {0, 0});
    } else {
      cluster_weights_ = std::vector<double>(num_nodes, 0);
    }
    SetClustering(clustering);
  }

  // Constructor for testing purposes, to outright set ClusteringHelper data
  // members.
  ClusteringHelper(
      size_t num_nodes, std::vector<ClusterId> cluster_ids,
      std::vector<ClusterId> cluster_sizes, std::vector<double> cluster_weights,
      const graph_mining::in_memory::ClustererConfig& clusterer_config,
      std::vector<double> node_weights, std::vector<NodePartId> node_parts = {},
      std::vector<std::array<double, 2>> partitioned_cluster_weights = {})
      : num_nodes_(num_nodes),
        cluster_ids_(std::move(cluster_ids)),
        cluster_sizes_(std::move(cluster_sizes)),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)),
        cluster_weights_(std::move(cluster_weights)),
        node_parts_(std::move(node_parts)),
        partitioned_cluster_weights_(std::move(partitioned_cluster_weights)) {}

  // Contains objective change, which includes:
  //  * A vector of tuples, indicating the objective change for the
  //    corresponding cluster id if a node is moved to said cluster.
  //  * The objective change of a node moving out of its current cluster
  struct ObjectiveChange {
    std::vector<std::tuple<ClusterId, double>> move_to_change;
    double move_from_change;
  };

  // Moves node i from its current cluster to a new cluster moves[i].
  // If moves[i] == null optional, then the corresponding node will not be
  // moved. A move to the number of nodes in the graph means that a new cluster
  // is created. The size of moves should be equal to num_nodes_.
  // Returns an array where the entry is true if the cluster corresponding to
  // the index was modified, and false if the cluster corresponding to the
  // index was not modified. Nodes may not necessarily move if the best move
  // provided is to stay in their existing cluster.
  std::unique_ptr<bool[]> MoveNodesToCluster(
      const std::vector<std::optional<ClusterId>>& moves);

  // Asynchronously moves node moving_node from its current cluster to a new
  // cluster given by move_cluster_id. Consistency guarantees are relaxed, in
  // that cluster weights may not always accurately reflect the actual
  // clusters that vertices participate in.
  void MoveNodeToClusterAsync(InMemoryClusterer::NodeId moving_node,
                              ClusterId move_cluster_id);

  // Asynchronously moves a set of nodes moving_nodes, which must all be in the
  // same initial cluster, to a new cluster given by move_cluster_id.
  // Consistency guarantees are relaxed, in that cluster weights may not always
  // accurately reflect the actual clusters that vertices participate in.
  void MoveNodesToClusterAsync(const std::vector<gbbs::uintE>& moving_nodes,
                               ClusterId move_cluster_id);

  // Returns a tuple of:
  //  * The best cluster to move all of the nodes in moving_nodes to according
  //    to the correlation clustering objective function. An id equal to the
  //    number of nodes in the graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  std::tuple<ClusteringHelper::ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      const std::vector<gbbs::uintE>& moving_nodes);

  // Returns a tuple of:
  //  * The best cluster to move moving_node to according to the correlation
  //    clustering objective function. An id equal to the number of nodes in the
  //    graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  std::tuple<ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      InMemoryClusterer::NodeId moving_node);

  // Compute the objective of the current clustering. See correlation.proto for
  // details on how it's computed.
  double ComputeObjective(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph);

  const std::vector<ClusterId>& ClusterIds() const { return cluster_ids_; }

  const std::vector<ClusterId>& ClusterSizes() const { return cluster_sizes_; }

  const std::vector<double>& ClusterWeights() const { return cluster_weights_; }

  const std::vector<NodePartId>& NodeParts() const { return node_parts_; }

  const std::vector<std::array<double, 2>>& PartitionedClusterWeights() const {
    return partitioned_cluster_weights_;
  }

  const std::vector<double>& NodeWeights() const { return node_weights_; }

  // Returns the weight of the given node, or 1.0 if it has not been set.
  double NodeWeight(InMemoryClusterer::NodeId id) const;

  // Given the number of nodes in the graph, the cluster ids, and the node
  // weights, resets the saved clustering state in the helper (including
  // cluster sizes and weights) to match the inputted clustering.
  void ResetClustering(const std::vector<ClusterId>& cluster_ids,
                       const std::vector<double>& node_weights,
                       const std::vector<NodePartId>& node_parts);

  // Unfolds cluster id space from N (the number of nodes) to N*2 if
  // CorrelationClustererConfig.use_auxiliary_array_for_temp_cluster_id is true.
  void MaybeUnfoldClusterIdSpace();

  // Folds temporary new cluster ids back to the original cluster id space.
  // Moved cluster ids are recorded in moved_clusters.
  void MaybeFoldClusterIdSpace(bool* moved_clusters);

 private:
  std::size_t num_nodes_;
  std::vector<ClusterId> cluster_ids_;
  std::vector<ClusterId> cluster_sizes_;
  graph_mining::in_memory::ClustererConfig clusterer_config_;
  std::vector<double> node_weights_;
  std::vector<double> cluster_weights_;
  std::vector<NodePartId> node_parts_;
  std::vector<std::array<double, 2>> partitioned_cluster_weights_;

  // Initialize cluster_ids_ and cluster_sizes_ given an initial clustering.
  // If clustering is empty, initialize singleton clusters.
  // num_nodes_ must be correctly set before calling this function.
  void SetClustering(const InMemoryClusterer::Clustering& clustering);
};

// Given cluster ids and a graph, compress the graph such that the new
// vertices are the cluster ids and the edges are aggregated by sum.
// Self-loops preserve the total weight of the undirected edges in the clusters.
// The helper is only used to provide the node weights.
absl::StatusOr<graph_mining::in_memory::GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const ClusteringHelper& helper);

// Validates CorrelationClustererConfig configuration.
absl::Status ValidateCorrelationClustererConfigConfig(
    const graph_mining::in_memory::CorrelationClustererConfig& config);

// Metadata used for maintaining the relation regarding node ids, cluster ids,
// and partition information between the current graph and the compressed graph
// in the bipartite case.
//
// Naming convention
//
// Entities
// - `node_id`: Node id
// - `cluster_id`: Cluster id
// - `part`: Partition id
//
// Versions
// - An entity without any prefix refers to the current uncompressed graph
// - An entity with `new_` prefix refers to the new compressed graph
struct BipartiteGraphCompressionMetadata {
  // Map node ids to new node ids.
  //
  // Specifically, for the i-th node in the current graph,
  // node_id_to_new_node_ids[i] is the node id in the new graph.
  std::vector<gbbs::uintE> node_id_to_new_node_ids;

  // Partition information for nodes in the new graph.
  std::vector<NodePartId> new_node_parts;

  // Map new node ids to current cluster ids.
  //
  // Specifically, for the i-th node in the new graph,
  // new_node_id_to_cluster_ids[i] is the cluster id in the current graph.
  std::vector<gbbs::uintE> new_node_id_to_cluster_ids;

  // Map {cluster id, part} pair to new node ids.
  //
  // Specifically, for the i-th cluster id in the current graph,
  // cluster_id_and_part_to_new_node_ids[i][0] is the node id for Partition 0 in
  // the new graph. cluster_id_and_part_to_new_node_ids[i][1] is for
  // Partition 1.
  //
  // If no new nodes map to a {cluster id, part} pair, this means that no nodes
  // in the current graph belong to this pair (either the cluster is empty or
  // the cluster is one-sided). This case is mapped to the invalid node id
  // UINT_E_MAX.
  std::vector<std::array<gbbs::uintE, 2>> cluster_id_and_part_to_new_node_ids;
};

// Prepares metadata in order to call CompressGraph for the bipartite case.
// Speficially, the metadata produced by this function, when applied to the
// graph compression logic, ensures that nodes sharing the same cluster id but
// from different parts are not merged.
//
// For details, refer to the comments for `BipartiteGraphCompressionMetadata`.
BipartiteGraphCompressionMetadata PrepareBipartiteGraphCompression(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<NodePartId>& parts,
    std::vector<std::array<gbbs::uintE, 2>>&&
        cluster_id_and_part_to_new_node_ids);

// For a bipartite graph, given `cluster_ids` the cluster ids for the previous
// iteration, `node_parts` the node part id for the previous iteration, and
// `cluster_id_and_part_to_new_node_ids` the mapping from the previous iteration
// to the graph used in the current iteration, returns a new cluster id
// collection where the i-th element represents the mapping from the i-th node
// in the previous iteration to the node id in the current iteration.
//
// A cluster id of UINT_E_MAX indicates that the node has already been placed
// into a finalized cluster. This is preserved in the remapping.
//
// Intended Use Case:
//
// `FlattenBipartiteClustering` is used for the first of the two-level cluster
// id flattening logic in the bipartite case.
//
// For reference, for a regular (non-bipartite) graph, each layer has
// `cluster_ids` representing the mapping from node id to cluster id. Given that
// the cluster ids in the previous layer are the node ids in the next layer, we
// can flatten the clusters as
//
// flattened_cluster_ids[i] = next_cluster_ids[cluster_ids[i]]
//
// For the bipartite case, a cluster id from the previous level does not
// necessarily equal to the node id at the next level. Thus we need to perform
// an extra translation during cluster flattening to first identify the relation
// from node ids at the previous level to the node ids at the next level.
//
// translation[i] = cluster_id_and_part_to_new_node_ids[cluster_ids[i],
//                                                      node_parts[i]]
//
// flattened_cluster_ids[i] = next_cluster_ids[translation[i]]
//
// The return from `FlattenBipartiteClustering` is the extra translation layer
// `translation` in the above formula.
std::vector<gbbs::uintE> FlattenBipartiteClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<NodePartId>& node_parts,
    const std::vector<std::array<gbbs::uintE, 2>>&
        cluster_id_and_part_to_new_node_ids);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_UTIL_H_
