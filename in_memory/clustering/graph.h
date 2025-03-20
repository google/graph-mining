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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/declare.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "in_memory/clustering/in_memory_clusterer.h"

ABSL_DECLARE_FLAG(bool, lock_free_import);

namespace graph_mining {
namespace in_memory {

// Represents a set of graphs. Does not take ownership of the graphs.
class MultipleGraphs : public InMemoryClusterer::Graph {
 public:
  using Graph = InMemoryClusterer::Graph;
  explicit MultipleGraphs(std::vector<Graph*> graphs)
      : graphs_(std::move(graphs)) {}

  absl::Status PrepareImport(int64_t num_nodes) override;

  absl::Status Import(AdjacencyList adjacency_list) override;

  absl::Status FinishImport() override;

 private:
  std::vector<Graph*> graphs_;
};

// Represents a directed graph that is simple, i.e., has at most one edge from
// x to y for any x and y. Self-loops are allowed.
class SimpleDirectedGraph : public InMemoryClusterer::Graph {
 public:
  absl::Status PrepareImport(int64_t num_nodes) override;

  // Calls SetNodeWeight, followed by AddEdge for each outgoing edge.
  absl::Status Import(AdjacencyList adjacency_list) override;

  absl::Status FinishImport() override;

  // Adds a new node. Returns an id equal to the number of nodes in the graph
  // before the addition. Note that unless you want to create isolated nodes,
  // this function is redundant due to the fact that AddEdge() automatically
  // creates referenced nodes.
  virtual NodeId AddNode();

  // Resizes the graph to the given number of nodes. The graph size may only
  // increase, i.e., the function CHECK-fails if the given parameter is
  // smaller than NumNodes().
  void SetNumNodes(NodeId num_nodes);

  // Returns the number of nodes in the graph. The nodes are numbered 0, ...,
  // NumNodes()-1.
  NodeId NumNodes() const { return adjacency_lists_.size(); }

  // Adds an directed edge of given weight between the nodes.
  // If the endpoints are already connected by an edge, weight is set to the
  // maximum of the existing and new weight.
  // Returns INVALID_ARGUMENT if any of the node indices is negative.
  virtual absl::Status AddEdge(NodeId from_node, NodeId to_node, double weight);

  // Sets the weight of the given edge to <weight>. If the edge did not exist
  // before, it is added to the graph.
  // Returns INVALID_ARGUMENT if any of the node indices is negative.
  virtual absl::Status SetEdgeWeight(NodeId from_node, NodeId to_node,
                                     double weight);

  // Removes an edge from the graph. Returns:
  //   - INVALID_ARGUMENT if any of the node ids is not in [0, NumNodes()).
  //   - NOT_FOUND if the node ids are valid, but the edge does not exist.
  //   - OK if the edge is removed.
  // Calling RemoveEdge invalidates any iterators to the hash map of neighbors
  // of from_node.
  virtual absl::Status RemoveEdge(NodeId from_node, NodeId to_node);

  // Returns a hash map containing neighbors of a given node. Note that
  // the order of iteration may change after each edge insertion.
  const absl::flat_hash_map<NodeId, double>& Neighbors(NodeId id) const {
    ABSL_CHECK_GE(id, 0);
    ABSL_CHECK_LT(id, adjacency_lists_.size());
    return adjacency_lists_.at(id);
  }
  // Returns the weight of the edge connecting the two endpoints, or an empty
  // optional if the edge does not exist.
  // The given ids may be arbitrary (even negative).
  std::optional<double> EdgeWeight(NodeId from_node, NodeId to_node) const;

  // Returns the weight of the given node, or 1.0 if it has not been set.
  // CHECK-fails if id is negative, but returns 1.0 if it is greater than
  // NumNodes()-1.
  double NodeWeight(NodeId id) const;

  // Returns the part of the given node, or 0 if it has not been set.
  // CHECK-fails if id is negative, but returns 0 if it is greater than
  // NumNodes()-1.
  int32_t NodePart(NodeId id) const;

  // Returns true if all nodes in the graph are assigned to part 0 (default).
  bool IsUnipartite() const;

  // Returns the total weight of all outgoing edges of a node.
  double WeightedOutDegree(NodeId id) const;

  // Sets the weight of the given node. Implicitly adds that node if necessary.
  virtual void SetNodeWeight(NodeId id, double weight);

  // Sets the part of the given node. Implicitly adds that node if necessary.
  virtual void SetNodePart(NodeId id, int32_t part);

  // Clears the adjacency list for a node and frees up the memory used by the
  // adjacency list.
  absl::Status ClearNeighbors(NodeId id);

  // Returns the number of directed edges in the graph. Takes O(num nodes) time.
  // Includes self edges if they exist.
  std::size_t NumDirectedEdges() const {
    std::size_t num_edges = 0;
    for (const auto& adjacency_list : adjacency_lists_) {
      num_edges += adjacency_list.size();
    }
    return num_edges;
  }

 protected:
  // Ensures that the graph has the given number of nodes, by adding new nodes
  // if necessary.
  void EnsureSize(NodeId id);

  // Returns INVALID_ARGUMENT iff the id does not correspond to an existing
  // node.
  absl::Status CheckNodeExists(NodeId id) const;

  // Returns INVALID_ARGUMENT iff id is negative.
  absl::Status CheckNodeIdValid(NodeId id) const;

 private:
  // Returns a pointer to the weight of the edge between the given endpoints. If
  // the edge does not exist, it is added to the graph with the given weight.
  // CHECK-fails if any node index is negative.
  double* MutableEdgeWeight(NodeId from_node, NodeId to_node,
                            double default_weight);

  // Implements Import. Delegate lock acquisition/release (if applicable) to
  // caller.
  absl::Status ImportHelper(AdjacencyList adjacency_list);

  // Clears node_weights_ vector if all weights equal to the default weight.
  // Returns true if node_weights_ is cleared.
  bool MaybeClearNodeWeights();

  // Clears node_parts_ vector if all part ids equal to the default id.
  // Returns true if node_parts_ is cleared.
  bool MaybeClearNodeParts();

  absl::Mutex mutex_;
  std::vector<absl::flat_hash_map<NodeId, double>> adjacency_lists_;
  std::vector<double> node_weights_;
  std::vector<int32_t> node_parts_;
  bool lock_free_import_ = false;
};

// A graph that works analogously to SimpleDirectedGraph, but upon inserting an
// edge x -> y, inserts also y -> x. This guarantees that the graph is
// undirected, i.e. the weight of the edge x -> y is the same as the weight of
// the edge y -> x.
class SimpleUndirectedGraph : public SimpleDirectedGraph {
 public:
  absl::Status PrepareImport(int64_t num_nodes) override;

  absl::Status FinishImport() override;

  absl::Status AddEdge(NodeId from_node, NodeId to_node,
                       double weight) override;

  // Removes an edge from the graph. Returns:
  //   - INVALID_ARGUMENT if any of the node ids is not in [0, NumNodes()).
  //   - NOT_FOUND if the node ids are valid, but the edge does not exist.
  //   - OK if the edge is removed.
  // Calling RemoveEdge invalidates any iterators to the hash map of neighbors
  // of both from_node and to_node.
  absl::Status RemoveEdge(NodeId from_node, NodeId to_node) override;

  absl::Status SetEdgeWeight(NodeId from_node, NodeId to_node,
                             double weight) override;

  // Total weight of incident edges. Each self-loop is counted once.
  double WeightedDegree(NodeId node) const { return WeightedOutDegree(node); }

 private:
  // Per-node lock used for Import.
  std::vector<absl::Mutex> per_node_lock_;

  // Lock granularity as defined by the number of nodes per lock.
  static constexpr int kLockGranularity = 256;
};

// Calls out_graph->Import() for each node in in_graph and then calls
// out_graph->FinishImport()
absl::Status CopyGraph(const SimpleDirectedGraph& in_graph,
                       InMemoryClusterer::Graph* out_graph);

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_H_
