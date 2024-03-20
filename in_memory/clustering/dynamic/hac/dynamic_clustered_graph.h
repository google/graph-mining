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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_CLUSTERED_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_CLUSTERED_GRAPH_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parallel_clustered_graph.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// A graph that supports adding and deleting nodes. Some nodes might have degree
// 0. The class ensures that the underlying graph has no dangling edges. The
// underlying representation uses parallelism. Hence, it is recommended to hold
// a ParallelSchedulerReference while using it. The stored graph is undirected
// (i.e. for each x -> y edge there is a y->x edge with the same weight).
// The graph also takes a `heavy_threshold_` parameter, and keeps track of the
// number of edges with weight >= `heavy_threshold_`. By default
// `heavy_threshold_`=inf, so there is no heavy edge.
class DynamicClusteredGraph {
 public:
  using Weight = graph_mining::in_memory::AverageLinkageWeight;
  using ClusterNode = graph_mining::in_memory::ClusteredNode<Weight>;
  using NodeId = graph_mining::in_memory::NodeId;
  using StoredWeightType = typename Weight::StoredWeightType;
  using AdjacencyList =
      graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;

  // Represent a subgraph of a DynamicClusteredGraph object. Node ids might
  // be different compared to the ids in the DynamicClusteredGraph.
  struct Subgraph {
    std::unique_ptr<graph_mining::in_memory::SimpleUndirectedGraph> graph;
    // A mapping from the ids in `graph` to the ids in this object.
    std::vector<NodeId> node_map;
    // The number of nodes i that are active.
    NodeId num_active_nodes;
    // Nodes that are inactive only because they do not have any heavy edge. It
    // is in global id.
    std::vector<NodeId> ignored_nodes;
  };

  explicit DynamicClusteredGraph(
      const double heavy_threshold = std::numeric_limits<double>::max())
      : heavy_threshold_(heavy_threshold){};

  std::size_t NumNodes() const { return clusters_.size(); }

  std::size_t NumEdges() const { return 2 * num_edges_; }

  std::vector<NodeId> Nodes() const {
    auto result = std::vector<NodeId>();
    for (const auto& [id, _] : clusters_) {
      result.push_back(id);
    }
    return result;
  }

  // Returns true if there exists edges with normalized weights >=
  //  `heavy_threshold_`.
  bool HasHeavyEdges() const { return num_heavy_edges_ > 0; }

  bool ContainsNode(NodeId i) const { return clusters_.contains(i); }

  // The returned pointer is invalidated when new nodes are added or removed.
  // We do not provide a MutableNode() option because then
  // `max_neighbor_weights_` might not be consistent with the graph edges.
  absl::StatusOr<const ClusterNode*> ImmutableNode(NodeId i) const {
    auto it = clusters_.find(i);
    if (it == clusters_.end())
      return absl::InvalidArgumentError(absl::StrCat("node not in graph ", i));
    return &(it->second);
  }

  // Add new nodes to the graph with edges incident to the nodes.
  // The weight in nodes should be the raw weight (e.g. in the average linkage
  // case, not divided by cluster sizes). Each new node needs to have an
  // AdjacencyList entry in `nodes`. All nodes appeared in
  // `nodes[i].outgoing_edges` need to be either already in the graph or a new
  // node in `nodes`. The size of the added node should be in `nodes[i].weight`.
  // If `skip_existing_nodes` is false, only new nodes should be in
  // `nodes[i].id` for all i. When this returns non-OK status, the
  // DynamicClusteredGraph data structure becomes not usable and no longer keeps
  // the invariants. Since the graph is undirected, if an edge between (i,j) is
  // in `nodes`, edge (j,i) will also be added. If i and j are both new nodes,
  // `nodes` is expected to contain both edge (i,j) and (j,i). Note that the
  // node weights are stored using float type, even though the AdjacencyList
  // struct represents them using doubles. If `skip_existing_nodes` is true,
  // skip adding outgoing edges from nodes that already exist. If a node v is
  // skipped and v is in the neighbor list of another node, this edge incident
  // to v is still added.
  absl::Status AddNodes(absl::Span<const AdjacencyList> nodes,
                        bool skip_existing_nodes = false);

  // Remove a node and its incident edges.
  // It returns ok status if node existed in the graph before deletion, and
  // NotFoundError otherwise. If a node v neighboring `node_id` has no edge
  // after `node_id` is deleted, v is not deleted. We do not delete v because
  // other users might call AddNodes assuming that node v exists.
  absl::Status RemoveNode(NodeId node_id);

  // Returns a subgraph of this DynamicClusteredGraph object. The subgraph's
  // `graph` contains all nodes i such that `partition_map[i]` in `node_ids`
  // and the neighbors of those nodes and all edges between them. The subgraph
  // nodes with ids in [0 ... `num_active_nodes`-1] are the active nodes and
  // have `partition_map[i] in node_ids`. The inactive nodes have node weights
  // -1 in the returned graph. Requires that `node_ids` are nodes in this
  // graph. Returns error status if we processed a node that is not in this
  // graph. Note that two nodes can only be in the same partition if they are at
  // most two-hop away.
  // Nodes without any heavy edge are also inactive.
  absl::StatusOr<std::unique_ptr<Subgraph>> CreateSubgraph(
      const absl::flat_hash_set<NodeId>& node_ids,
      const absl::flat_hash_map<NodeId, NodeId>& partition_map) const;

  // Return the union of all neighbors of `nodes`. If a neighbor is in `nodes`,
  // it will also be returned. Returns error status if any node is not in the
  // graph.
  absl::StatusOr<absl::flat_hash_set<graph_mining::in_memory::NodeId>>
  Neighbors(
      const absl::flat_hash_set<graph_mining::in_memory::NodeId>& nodes) const;

  // Returns the number of edges with normalized weights >= `heavy_threshold_`.
  std::size_t NumHeavyEdges() const { return num_heavy_edges_; }

  // Returns True if node i has at least one edges with normalized weights >=
  // `heavy_threshold_`. Returns InvalidArgumentError if node `i` is not in the
  // graph. Takes O(num_neighbors) time.
  absl::StatusOr<bool> HasHeavyEdges(NodeId i) const;

 private:
  // Returns the similarity of edge (a,b) normalized by node sizes. It always
  // returns the exact same similarity for the edge without numerical precision
  // discrepancy. It also works when the input arguments `a` and `b` are
  // swapped. Requires that `a` and `b` are both nodes in the graph.
  double StableSimilarity(NodeId a, NodeId b) const;

  absl::flat_hash_map<NodeId, ClusterNode> clusters_;

  const double heavy_threshold_;

  // The number of edges with weights (normalized by node sizes) >=
  // than `heavy_threshold_`.
  std::size_t num_heavy_edges_ = 0;

  // The total number of edges.
  std::size_t num_edges_ = 0;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_CLUSTERED_GRAPH_H_
