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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_NODE_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_NODE_H_

#include <numeric>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/types/span.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// This class is a helper class for the graph data structure defined in
// ApproximateSubgraphHacGraph. The class implements the "node" data structure
// used in the graph. The stored edge weights are computed using the
// average-linkage formula.
//
// Each edge is owned by only *one* of its endpoint, i.e., edges can be viewed
// as being "oriented" or "assigned to" one of its two endpoints. Each node only
// stores the goodness estimates for the edges that it owns.
class ApproximateSubgraphHacNode {
 public:
  using NodeId = InMemoryClusterer::NodeId;
  // The goodness and neighbor id corresponding to an edge.
  using GoodnessAndId = std::pair<double, NodeId>;

  // Information stored along with every edge.
  struct NeighborInfo {
    double partial_weight;  // the partial edge-weight
    size_t cluster_size;    // cluster size of the neighbor used to normalize;
                            // storing this field lets us reconstruct the cut
                            // weight exactly.
    double goodness;  // the goodness estimate currently stored with this edge.
  };

  // A default goodness value. Used by GetGoodEdge when no further good edges
  // remain.
  static constexpr double kDefaultGoodness =
      std::numeric_limits<double>::infinity();

  // Construct a node given its cluster size and the approximation factor
  // one_plus_alpha.
  ApproximateSubgraphHacNode(NodeId cluster_size, double one_plus_alpha);

  // Returns the number of edges assigned to this active node. Note that all
  // assigned edges go between active vertices. Exposed only for testing.
  size_t NumAssignedEdges() const;

  // Returns the current cluster size of this node.
  size_t CurrentClusterSize() const;

  // Inserts a new edge to neighbor with average-linkage-weight weight. Note
  // that this call *does not* assign the edge to this node.
  void InsertEdge(NodeId neighbor, size_t neighbor_cluster_size, double weight);

  // Assigns (i.e., orients) the edge from this node to its neighbor. This edge
  // is only owned by *this* endpoint, i.e., the neighbor will not maintain a
  // goodness estimate for this edge.
  void AssignEdge(NodeId neighbor, double goodness_to_neighbor);

  // Returns the exact average-linkage weight of the (u,v) edge. Expects
  // that the edge to neighbor exists and that this node is active.
  double EdgeWeight(NodeId neighbor, size_t neighbor_size) const;

  // Returns true iff neighbor_id is a neighbor of this node.
  bool IsNeighbor(NodeId neighbor_id) const;

  // Returns the stored NeighborInfo struct for the given neighbor_id; this
  // method is exposed to perform assertions in ApproximateSubgraphHacGraph.
  NeighborInfo GetNeighborInfo(NodeId neighbor_id) const;

  // Returns a handle to the neighbors of this node. Useful for extracting the
  // edges incident to an active node, e.g., after completing SubgraphHac.
  const absl::flat_hash_map<NodeId, NeighborInfo>& Neighbors() const;

  // Returns information about the current best edge incident to this node.
  // In particular, returns the best weight and corresponding neighbor.
  // Returns a one_plus_alpha approximation of the true best edge weight.
  std::pair<double, NodeId> ApproximateBestWeightAndId() const;

  // Scan the edges assigned to node node_id and returns an edge with
  // goodness <= threshold (any value s.t. threshold >= 1+eps).
  //
  // The return value is either:
  // (a) {kDefaultGoodness, NodeId::max()} or
  // (b) {goodness_uv, node_v}
  //
  // If all remaining edges have true goodness > threshold/(1+alpha), this
  // function can return case (a). I.e., if threshold = 1+eps, there could
  // still be an edge with true goodness between
  // ((1+epsilon)/(1+alpha), 1+epsilon)] that the algorithm simply ignores.
  // In case (b), goodness_uv <= threshold.
  std::pair<double, NodeId> GetGoodEdge(
      NodeId node_id, double threshold,
      std::function<double(NodeId, NodeId)> get_goodness);

  // If the cluster size increased by a multiplicative factor of one_plus_alpha,
  // then update the partial weights stored in the neighbor's endpoints. Returns
  // true iff the node broadcasted.
  bool MaybeBroadcastClusterSize(NodeId node_id,
                                 absl::Span<ApproximateSubgraphHacNode> nodes,
                                 const std::vector<bool>& is_active);

  // Check if our best value decreased by a one_plus_alpha factor. If it has,
  // then go over all of the incident edges, and potentially reassign them.
  // Returns true iff the best value decreased enough and the node scanned its
  // assigned edges. Any nodes that are assigned an edge are added to the
  // supplied set nodes_to_update_in_pq.
  bool MaybeReassignEdges(NodeId node_id,
                          absl::Span<ApproximateSubgraphHacNode> nodes,
                          const std::vector<bool>& is_active,
                          absl::flat_hash_set<NodeId>* nodes_to_update_in_pq,
                          std::function<double(NodeId, NodeId)> get_goodness);

  // This function merges the neighborhoods of node_from and node_to where
  // node_from's neighbors are assigned to node_to. This function returns a set
  // of edges to reassign, and a set of nodes whose best edge values may be
  // affected. This function sets node_from to be inactive.
  std::pair<std::vector<std::pair<NodeId, NodeId>>, absl::flat_hash_set<NodeId>>
  Merge(NodeId node_from, NodeId node_to,
        absl::Span<ApproximateSubgraphHacNode> nodes,
        std::vector<bool>& is_active);

 private:
  //  Edge weights stored in the graph are partial, i.e., only normalized by
  //  the neighbor endpoints cluster size at the time the edge weight was
  //  computed.
  using PartialWeightAndId = std::pair<double, NodeId>;

  // Changes the last updated cluster size to be equal to the current cluster
  // size.
  void UpdateLastUpdatedClusterSize();

  // Returns true if the cluster size has changed by at least a multiplicative
  // factor of one_plus_alpha.
  bool ClusterSizeChangedEnough() const;

  // Returns true if the best edge weight has changed by a multiplicative factor
  // of one_plus_alpha.
  bool BestWeightChangedEnough() const;

  // Update the cluster size.
  void UpdateClusterSize(size_t new_cluster_size);

  // Update the stored partial_weight of this edge.
  void UpdateEdge(NodeId neighbor, double partial_weight, size_t cluster_size);

  // The best weight when this node last broadcasted to its neighbors.
  double prev_best_weight_;
  // The current cluster size of this node.
  size_t current_cluster_size_;
  // The cluster size of this node when it last updated its neighbors.
  size_t last_updated_cluster_size_;

  // The approximation factor used when broadcasting the node's cluster size
  // to its neighbors, or when checking whether to reassign currently assigned
  // edges to this node.
  const double one_plus_alpha_;

  // All neighbors of the node, stored in the set sorted by
  // decreasing partial weight.
  absl::btree_set<PartialWeightAndId, std::greater<PartialWeightAndId>>
      all_neighbors_;

  // Stores the partial weight of the edge to the neighbor, and the cluster
  // size used in the partial weight. The product of partial_weight *
  // cluster_size is the cut weight for the (node_id, neighbor_id) cut.
  absl::flat_hash_map<NodeId, NeighborInfo> neighbor_info_;

  // Goodness values (estimates) of neighbors assigned to this endpoint.
  absl::btree_set<GoodnessAndId, std::less<GoodnessAndId>> goodness_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_NODE_H_
