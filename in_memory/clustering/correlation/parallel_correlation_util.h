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

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/correlation.pb.h"
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

  // Constructor arguments:
  // - num_nodes: The number of nodes in the graph. Must be positive.
  // - clusterer_config: The clusterer configuration.
  // - node_weights: The weights of the nodes in the objective function. Must
  //   have length num_nodes.
  // - clustering: The initial clustering. If nonempty, must have length
  //   num_nodes. If empty, the initial clustering is a collection of singletons
  //   (each node is in its own cluster).
  // - node_parts: The node parts. This is used only if
  //   clusterer_config.correlation_clusterer_config.use_bipartite_objective is
  //   true. In that case, node_parts must have length num_nodes, and each value
  //   must be either 0 or 1.
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
    ABSL_CHECK_GE(num_nodes, 1);
    ABSL_CHECK_EQ(node_weights_.size(), num_nodes);
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      ABSL_CHECK_EQ(node_parts.size(), num_nodes);
      ABSL_CHECK(IsValidBipartiteNodeParts(node_parts));
      node_parts_ = node_parts;
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
        node_parts_(std::move(node_parts)),
        cluster_weights_(std::move(cluster_weights)),
        partitioned_cluster_weights_(std::move(partitioned_cluster_weights)) {
    ABSL_CHECK_GE(num_nodes, 1);
    ABSL_CHECK_EQ(cluster_ids_.size(), num_nodes);
    ABSL_CHECK_EQ(cluster_sizes_.size(), num_nodes);
    ABSL_CHECK_EQ(node_weights_.size(), num_nodes);
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      ABSL_CHECK_EQ(node_parts_.size(), num_nodes);
      ABSL_CHECK(IsValidBipartiteNodeParts(node_parts_));
      ABSL_CHECK_EQ(cluster_weights_.size(), 0);
      ABSL_CHECK_EQ(partitioned_cluster_weights_.size(), num_nodes);
    } else {
      ABSL_CHECK_EQ(node_parts_.size(), 0);
      ABSL_CHECK_EQ(cluster_weights_.size(), num_nodes);
      ABSL_CHECK_EQ(partitioned_cluster_weights_.size(), 0);
    }
  }

  // Moves each node i from its current cluster to a new cluster moves[i].
  // If moves[i] == nullopt, then the corresponding node will not be moved. A
  // move to the number of nodes in the graph means that a new cluster is
  // created. The size of moves should be equal to num_nodes_. Returns an array
  // of size num_nodes_ where the entry is true if the cluster corresponding to
  // the index was modified, and false if the cluster corresponding to the index
  // was not modified.
  // Can be called only if
  // clusterer_config_.correlation_clusterer_config.use_synchronous is true.
  std::unique_ptr<bool[]> MoveNodesToCluster(
      const std::vector<std::optional<ClusterId>>& moves);

  // Asynchronously moves node moving_node from its current cluster to a new
  // cluster given by target_cluster_id. Consistency guarantees are relaxed, in
  // that cluster weights may not always accurately reflect the actual
  // clusters that vertices participate in.
  // Can be called only if
  // clusterer_config_.correlation_clusterer_config.use_synchronous is false.
  void MoveNodeToClusterAsync(InMemoryClusterer::NodeId moving_node,
                              ClusterId target_cluster_id);

  // Asynchronously moves a set of nodes moving_nodes, which must all be in the
  // same initial cluster, to a new cluster given by target_cluster_id.
  // Consistency guarantees are relaxed, in that cluster weights may not always
  // accurately reflect the actual clusters that vertices participate in.
  // moving_nodes must be non-empty. Can be called only if
  // clusterer_config_.correlation_clusterer_config.use_synchronous is false.
  void MoveNodesToClusterAsync(const std::vector<gbbs::uintE>& moving_nodes,
                               ClusterId target_cluster_id);

  // Struct holding information about a potential move of a node (or a set of
  // nodes) to a different cluster.
  struct ClusterMove {
    // Cluster to which the node(s) will be moved if the move is performed.
    ClusteringHelper::ClusterId target_cluster_id;

    // Change in objective function if the move is performed.
    double objective_change = 0.0;
  };

  // Returns the best move for all nodes in moving_nodes, according to the
  // correlation clustering objective function. If the best move is to create a
  // new cluster instead of moving to an existing one, a special cluster ID is
  // returned, defined as:
  //  - the number of nodes in the graph if
  //    use_auxiliary_array_for_temp_cluster_id is false; or
  //  - 2 * the number of nodes in the graph if
  //    use_auxiliary_array_for_temp_cluster_id is true.
  //
  // The number of nodes in graph must be equal to not num_nodes_, and all
  // entries in moving_nodes must be smaller than num_nodes_.
  ClusterMove BestMove(
      const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      const std::vector<gbbs::uintE>& moving_nodes);

  // Computes the objective of the current clustering. See correlation.proto for
  // details on how it's computed. Cannot be called when the cluster ID space is
  // unfolded.
  double ComputeObjective(
      const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph)
      const;

  // Accessor for the ClustererConfig passed to the constructor.
  const graph_mining::in_memory::ClustererConfig& ClustererConfig() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return clusterer_config_;
  }

  const std::vector<ClusterId>& ClusterIds() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return cluster_ids_;
  }

  const std::vector<ClusterId>& ClusterSizes() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return cluster_sizes_;
  }

  const std::vector<double>& ClusterWeights() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return cluster_weights_;
  }

  const std::vector<NodePartId>& NodeParts() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return node_parts_;
  }

  const std::vector<std::array<double, 2>>& PartitionedClusterWeights() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return partitioned_cluster_weights_;
  }

  const std::vector<double>& NodeWeights() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return node_weights_;
  }

  // Returns the weight of the given node. The id must be in the range [0,
  // num_nodes_).
  double NodeWeight(InMemoryClusterer::NodeId id) const;

  // Given the the cluster IDs, the node weights, and possibly the node parts,
  // resets the saved clustering state in the helper (including cluster sizes
  // and weights) to match the inputted clustering.
  //
  // The size of node_weights will become the new value of num_nodes_.
  // cluster_ids must have length node_weights.size().
  //
  // node_parts is used only if
  // clusterer_config_.correlation_clusterer_config.use_bipartite_objective is
  // true. In that case, it must have length node_weights.size(), and each value
  // must be either 0 or 1.
  void ResetClustering(const std::vector<ClusterId>& cluster_ids,
                       std::vector<double> node_weights,
                       std::vector<NodePartId> node_parts);

  // Unfolds cluster id space from N (the number of nodes) to N*2 if
  // CorrelationClustererConfig.use_auxiliary_array_for_temp_cluster_id is true
  // (otherwise, this is a no-op).
  void MaybeUnfoldClusterIdSpace();

  // Folds temporary new cluster IDs back to the original cluster id space if
  // CorrelationClustererConfig.use_auxiliary_array_for_temp_cluster_id is true
  // (otherwise, this is a no-op). Moved cluster ids are recorded in
  // moved_clusters, which must have length num_nodes_.
  void MaybeFoldClusterIdSpace(bool moved_clusters[]);

 private:
  std::size_t num_nodes_;

  // Maps each node ID to the ID of the cluster to which the node belongs.
  //   - If correlation_clusterer_config.use_auxiliary_array_for_temp_cluster_id
  //     in clusterer_config_ is false, then cluster IDs are in the range [0,
  //     num_nodes_) at all times.
  //   - Otherwise, the range of valid cluster IDs alternates between [0,
  //     num_nodes_) (initially, and after each call to MaybeFoldClusterIdSpace)
  //     and [0, 2 * num_nodes_) (after each call to MaybeUnfoldClusterIdSpace).
  std::vector<ClusterId> cluster_ids_;

  // Maps each cluster ID to the number of nodes in the cluster. Initially has
  // length num_nodes_. The length may alternate between num_nodes_ and 2 *
  // num_nodes_ depending on whether the cluster ID space is folded or unfolded
  // (see the documentation of cluster_ids_ above).
  std::vector<ClusterId> cluster_sizes_;

  graph_mining::in_memory::ClustererConfig clusterer_config_;

  // Maps each node ID to the weight of the node in the objective function.
  std::vector<double> node_weights_;

  // Maps each node ID to the part of the node (0 or 1); used only when
  // clusterer_config_.correlation_clusterer_config.use_bipartite_objective is
  // true.
  std::vector<NodePartId> node_parts_;

  // Weights of the clusters.
  // Note: only one of the following two vectors is used, depending on whether
  // the objective function is bipartite or not; the other remains empty at all
  // times. Invariant: the length of the used vector is equal to that of
  // cluster_sizes_.
  std::vector<double> cluster_weights_;  // For non-bipartite objective.
  std::vector<std::array<double, 2>>
      partitioned_cluster_weights_;  // For bipartite objective.

  // Initializes cluster_ids_ and cluster_sizes_ given an initial clustering.
  // If clustering is empty, initialize singleton clusters.
  // num_nodes_ and node_weights_ must be correctly set before calling this
  // function, as well as node_parts_ if
  // clusterer_config_.correlation_clusterer_config.use_bipartite_objective is
  // true.
  void SetClustering(const InMemoryClusterer::Clustering& clustering);
};

// Given cluster ids and a graph, compress the graph such that the new
// vertices are the cluster ids and the edges are aggregated by sum. The cluster
// IDs must be in the range [0, number of distinct elements in cluster_ids).
// Self-loops preserve the total weight of the undirected edges in the clusters.
// The helper is only used to provide the node weights. The number of nodes in
// original_graph must match the lengths of cluster_ids and
// helper.NodeWeights().
absl::StatusOr<graph_mining::in_memory::GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const ClusteringHelper& helper);

// Validates CorrelationClustererConfig configuration.
absl::Status ValidateCorrelationClustererConfig(
    const graph_mining::in_memory::CorrelationClustererConfig& config,
    size_t num_nodes);

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
// Specifically, the metadata produced by this function, when applied to the
// graph compression logic, ensures that nodes sharing the same cluster id but
// from different parts are not merged.
//
// All elements of cluster_ids must be in the range [0, cluster_ids.size()).
// The length of parts must match that of cluster_ids, and all elements of parts
// must be equal to 0 or 1.
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
// All elements of cluster_ids that are not UINT_E_MAX must be in the range [0,
// cluster_id_and_part_to_new_node_ids.size()). The length of node_parts must be
// equal to that of cluster_ids, and all its elements must be equal to 0 or 1.
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
