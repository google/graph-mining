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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_GRAPH_H_

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "utils/container/fixed_size_priority_queue.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_node.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// This class maintains a graph data structure that maintains an edge weighted
// subgraph of a graph under merges using the average-linkage formula. Note that
// the graph makes a distinction between active nodes (i.e. nodes "owned" by
// this subgraph and eligible to be merged) and inactive nodes (nodes owned by a
// different subgraph, and thus will never be merged from the perspective of
// this subgraph).
// The structure provides efficient access to the "goodness" of the edges as the
// merges occur.
//
// The goodness of an edge (u,v) is defined as:
//   goodness(u,v) = max(WMax(u), WMax(v)) /
//                   min(M(u), M(v), w(u,v))
// where WMax(u) is the max weight edge incident to u, and M(u) is the min-merge
// value of u, i.e., the minimum merge similarity of any merge that forms the
// cluster u.  If we always perform merges of edges that have goodness(u,v) <=
// 1+epsilon, then we can show that the algorithm is (1+epsilon)-approximate.
//
// This class maintains a dynamic graph data structure that efficiently:
// (1) produces edges with goodness <= 1+epsilon (see below for details)
// (2) merges two clusters u, v into a cluster (u U v).
//
// This is a special case of a more general "GoodnessGraph" data structure that
// can query for edges in any range of goodness values.
//
// The properties above are useful for implementing an efficient SubgraphHAC
// algorithm (i.e., merging all (1+epsilon)-good edges in a subgraph).
//
// Our implementation of GoodnessGraph is itself approximate. In particular, if
// Goodness*(u,v) is the true goodness of (u,v), we maintain an approximate
// value, ApxGoodness(u,v) s.t.:
//   Goodness*(u,v) <= ApxGoodness(u,v) <=  (1+alpha) x Goodness*(u,v).
// so [ApxGoodness(u,v) <= 1+epsilon] also implies [Goodness*(u,v) <= 1+epsilon]
//
// Our implementation will emit all edges with ApxGoodness(u,v) <= 1+epsilon, so
// when the algorithm stops returning edges we know that there are no more edges
// remaining with [Goodness*(u,v) <= (1+epsilon)/(1+\alpha)].
//
// See the Appendix of https://arxiv.org/abs/2308.03578 for more details on the
// dynamic graph representation implemented in this file and its guarantees.
//
// Another implementation detail is that weights stored in the graph are
// normalized by the neighbor endpoints cluster size, and implicitly normalized
// by the source's cluster size.
//
// Terminology and Invariants:
// - epsilon: never return an edge with goodness > (1+epsilon).
// - alpha: 0 <= alpha <= epsilon. Controls the frequency with which the
//   algorithm performs "broadcasts", e.g., of goodness reassignment, or
//   broadcasting a node's cluster size to its neighbors.
// - w(u,v): the weight of an edge (u,v). We maintain a (1+alpha) approximation
//   (since cluster sizes are broadcasted only after they grow by a (1+alpha)
//   factor). If w*(u,v) is the true weight, we guarantee that:
//       w*(u,v) <= w(u,v) <= (1+alpha) x w*(u,v)
// - active/inactive node: active nodes are nodes that are *in* this
//   subgraph, and are eligible to be merged. Inactive nodes are not eligible
//   to participate in merges.
// - Best(v): the highest weight incident edge to v (could be either to an
//   active or inactive node). Since we just return the heaviest weight edge we
//   also get a (1+alpha) approximation for Best(v):
//       Best*(v) <= Best(v) <= (1+alpha) x Best*(v)
// - Goodness(u,v): Since the goodness calculation uses one of {Best(u),Best(v)}
//   and (potentially) w(u,v), we also get a (1+alpha) approximation:
//       Goodness*(u,v) <= Goodness(u,v) <=  (1+alpha) x Goodness*(u,v)
//   Note that the goodness being <= (1+epsilon) means that we get a (1+epsilon)
//   approximation.
class ApproximateSubgraphHacGraph {
 public:
  using NodeId = InMemoryClusterer::NodeId;

  // A default goodness value. Used by GetGoodEdge when no further good edges
  // remain.
  static constexpr double kDefaultGoodness =
      ApproximateSubgraphHacNode::kDefaultGoodness;

  // Construct an ApproximateSubgraphHACGraph where the input is:
  // (1) a subgraph containing active and inactive nodes where node weights
  //     correspond to cluster size.
  // (2) the number of nodes in the graph, epsilon, and alpha defined as before,
  // (3) is_active, which indicates whether a node is active or not, and
  // (4) min_merge_similarities, which store the M(v) values per node.
  explicit ApproximateSubgraphHacGraph(
      const SimpleUndirectedGraph& graph, NodeId num_nodes, double epsilon,
      double alpha, std::vector<bool> is_active,
      const std::vector<double>& min_merge_similarities);

  // Number of nodes (clusters) initially present in the graph. This includes
  // both active and inactive nodes.
  size_t NumNodes() const;

  // Either returns an edge as a triple of (node_id, node_id, goodness) that is
  // (1+epsilon)-good, or returns (MaxNodeId, MaxNodeId, kDefaultGoodness), if
  // no such edge is found. If there is an edge with true goodness between
  // [1, (1+/epsilon)/(1+/alpha)], this method is guaranteed to return it.
  // If a valid edge e is returned, we have: std::get<0>(e) < std::get<1>(e).
  //
  // In the first case of an actual edge being returned, both endpoints are
  // guaranteed to be active.
  std::tuple<NodeId, NodeId, double> GetGoodEdge();

  // Merge node_a and node_b together in the graph, and update the given
  // dendrogram. Each merge merges two active nodes and deactivates one of them.
  // The node with fewer neighbors will be merged into the larger neighbor, and
  // the node that is merged into (which is still active) is returned. The
  // similarity of the merge is computed using EdgeWeight(node_a, node_b),
  // which is the exact edge weight. Updates min_merge_similarities of the
  // returned node.
  absl::StatusOr<NodeId> Merge(graph_mining::in_memory::Dendrogram* dendrogram,
                               std::vector<NodeId>* to_cluster_id,
                               std::vector<double>* min_merge_similarities,
                               NodeId node_a, NodeId node_b);

  // Returns the exact average-linkage weight of the (u,v) edge. Expects that
  // the (node_u, node_v) edge exists and that node_u is active. Exposed
  // publicly to allow the implementation to be tested.
  double EdgeWeight(NodeId node_u, NodeId node_v) const;

  // Returns the exact weight of the (u,v) edge unnormalized by cluster size.
  // Expects that the (node_u, node_v) edge exists and that node_u is active.
  // Exposed publicly to allow the implementation to be tested.
  double EdgeWeightUnnormalized(NodeId node_u, NodeId node_v) const;

  // Returns true if and only if node_u is active.
  bool IsActive(NodeId node_u);

  // Returns the exact current cluster size of node_u. Cluster size of inactive
  // nodes are not well defined.
  std::size_t CurrentClusterSize(NodeId node_u) const;

  // Return the neighbors of `id`.
  std::vector<NodeId> Neighbors(NodeId node_id) const;

 private:
  // Used by the Merge routine (see above).
  using Dendrogram = graph_mining::in_memory::Dendrogram;
  // The goodness and neighbor id corresponding to an edge.
  using GoodnessAndId = std::pair<double, NodeId>;
  // Edge weights stored in the graph are partial, i.e., only normalized by
  // the neighbor endpoints cluster size at the time the edge weight was
  // computed.
  using PartialWeightAndId = std::pair<double, NodeId>;

  // Called by ApproximateSubgraphHACGraph's constructor. Initializes the
  // internal data structures of this object based on the input subgraph.
  void Initialize(const SimpleUndirectedGraph& graph);

  // Computes the (approximate) goodness value of the (node_u, node_v) edge.
  // Expects that both node_u and node_v are active and the (node_u, node_v)
  // edge exists.
  //
  // Let g be the true goodness of (node_u, node_v), and ~g be the value
  // returned by this function. We ensure that g <= ~g <= (1+alpha)g.
  double Goodness(NodeId node_u, NodeId node_v);

  // Merge the neighborhoods of node_u and node_v from u --> v
  void MergeNeighborhoods(NodeId node_u, NodeId node_v);

  // Broadcast node_v's cluster size to its neighbors only if the cluster size
  // has changed enough since the last broadcast. Specifically, this procedure
  // will only perform a broadcast if the cluster size has increased by a
  // (1+alpha) factor.
  void MaybeBroadcastClusterSize(NodeId node_v);

  // Perform a broadcast of node_v's cluster size to all of its neighbors. This
  // procedure will update the partial weight in each of its node_v's neighbors
  // data structures to be the exact partial weight.
  void BroadcastClusterSize(NodeId node_v);

  // Updates the node_pq_ value of the active node node_id. If node_id has no
  // more incident active edges, it is removed from node_pq_.
  void UpdateNodePQ(NodeId node_id);

  // Reassign all edges potentially affected this merge to whichever endpoint
  // has larger B(u) value.
  void ReassignChangedEdges(
      std::vector<std::pair<NodeId, NodeId>> edges_to_reassign,
      absl::flat_hash_set<NodeId>& nodes_to_update_in_pq);

  // Whether the node is active nor not.
  std::vector<bool> is_active_;

  // The min-merge value of each node.
  const std::vector<double>& min_merge_similarities_;

  // A priority queue indexed on the nodes. Each node's priority is the edge
  // weight of a neighbor that is (1+epsilon)-good at the time the PQ is
  // updated.
  FixedSizePriorityQueue<double> node_pq_;

  // Stores per-node information.
  std::vector<ApproximateSubgraphHacNode> nodes_;

  // Constants used by the algorithm.
  // one_plus_alpha_ controls how often the algorithm performs a broadcast, as
  // described above.
  double one_plus_alpha_;
  // one_plus_eps_ stores 1 + epsilon (the input parameter). Please see above
  // for more details.
  double one_plus_eps_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_GRAPH_H_
