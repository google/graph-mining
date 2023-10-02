// Copyright 2010-2023 Google LLC
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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "in_memory/parallel/scheduler.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// A wrapper for GBBS graph types. Should be used with GraphType =
// gbbs::X_ptr_graph<gbbs::X_vertex, Y>, where
//  X is either symmetric or asymmetric
//  Y specifies the weight type
// Setting Y = gbbs::empty gives an unweighted graph.
//
// Note that GBBS doesn't support node weights.
template <typename GraphType>
class GbbsGraphBase : public InMemoryClusterer::Graph {
 public:
  // Edge type of the underneath GBBS graph.
  using Edge = std::tuple<gbbs::uintE, typename GraphType::weight_type>;

  // Prepares graph for node importing.
  //
  // `num_nodes` should be the exact number of nodes in the input graph. Graph
  // building fails if `num_nodes` is smaller than the actual number of nodes.
  // Setting `num_nodes` to a value larger than the actual number of nodes in
  // the graph leads to default-constructed isolated nodes.
  absl::Status PrepareImport(int64_t num_nodes) override;

  // Stores the node and edge information in nodes_ and edges_. Does not
  // automatically symmetrize the graph. If a vertex u is in
  // the adjacency list of a vertex v, then it is not guaranteed that vertex v
  // will appear in the adjacency list of vertex u unless explicitly
  // specified in vertex u's adjacency list.
  absl::Status Import(AdjacencyList adjacency_list) override;

  // Constructs graph_ using nodes_ and edges_, and optionally node_weights_ and
  // node_parts.
  absl::Status FinishImport() override;

  GraphType* Graph() const;

  // Returns the per-node partition id information. Returns an empty vector if
  // no such information exists or if all nodes have the default partition id.
  const std::vector<NodePartId>& GetNodeParts() const;

 protected:
  // Ensures that the graph has the given number of nodes, by adding new nodes
  // if necessary.
  void EnsureSize(NodeId id);

  // Clears node_weights_ vector if all weights equal to the default weight.
  // Returns true if node_weights_ is cleared.
  bool MaybeClearNodeWeights();

  // Clears node_parts_ vector if all part ids equal to the default id.
  // Returns true if node_parts_ is cleared.
  bool MaybeClearNodeParts();

  absl::Mutex mutex_;
  std::unique_ptr<GraphType> graph_;
  std::vector<typename GraphType::vertex> nodes_;
  std::vector<typename GraphType::vertex_weight_type> node_weights_;
  std::vector<NodePartId> node_parts_;
  std::vector<std::unique_ptr<
      std::tuple<gbbs::uintE, typename GraphType::weight_type>[]>>
      edges_;

  // Total number of nodes assuming no dangling edge exists. Used for graph
  // building optimization only.
  gbbs::uintE num_nodes_ = 0;

  // Default node weight.
  static constexpr int kDefaultNodeWeight = 1;

  // Default node part id.
  static constexpr NodePartId kDefaultNodePartId = 0;
};

// Instantiations of the GbbsGraphBase. In each case multiple edges and
// self-loops are allowed.
// Adding a new instantiation may require adding overloads in the internal
// namespace below.

// Weighted undirected graph. WARNING: the caller must ensure an undirected
// graph is loaded (see comment on GbbsGraphBase::Import).
class GbbsGraph : public
    GbbsGraphBase<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>> {
 public:
  // Reweights graph. Must be called after FinishImport.
  //
  // It is the caller's responsibility to synchronize ReweightGraph such that
  // there is no access to the graph (read or write) when the reweighting is in
  // progress. If ReweightGraph returns a non-OK status, then the underlying
  // graph may be partially reweighted and remains in an inconsistent state.
  absl::Status ReweightGraph(
      const std::function<absl::StatusOr<float>(
          gbbs::uintE node_id, gbbs::uintE neighbor_id, std::size_t node_degree,
          std::size_t neighbor_degree, float current_edge_weight)>&
          edge_reweighter);
};

// Directed unweighted graph. The resulting graph has only its out-neighbors
// populated.
using DirectedUnweightedGbbsGraph = GbbsGraphBase<
    gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, gbbs::empty>>;

// Weighted directed graph.
using DirectedGbbsGraph = GbbsGraphBase<
    gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, float>>;

// Unweighted undirected graph.
using UnweightedGbbsGraph = GbbsGraphBase<
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, gbbs::empty>>;

// Weighted undirected graph with sorted neighbors.
class UnweightedSortedNeighborGbbsGraph
    : public GbbsGraphBase<
          gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, gbbs::empty>> {
 public:
  absl::Status Import(AdjacencyList adjacency_list) override;
};

//////////////////////////////////////////////////////////////////////////////
/// END OF PUBLIC API. IMPLEMENTATION ONLY BELOW
//////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename Weight>
inline void SetId(gbbs::asymmetric_vertex<Weight>* vertex,
                  NodeId id) {
  vertex->id = id;
}

template <typename Weight>
inline void SetId(gbbs::symmetric_vertex<Weight>* vertex,
                  NodeId id) {
  vertex->id = id;
}

template <typename Weight>
inline void SetOutDegree(gbbs::asymmetric_vertex<Weight>* vertex,
                         NodeId outdegree) {
  vertex->out_deg = outdegree;
}

template <typename Weight>
inline void SetOutDegree(gbbs::symmetric_vertex<Weight>* vertex,
                         NodeId outdegree) {
  vertex->degree = outdegree;
}

template <typename Weight>
inline void SetOutNeighbors(
    gbbs::asymmetric_vertex<Weight>* vertex,
    typename gbbs::asymmetric_vertex<Weight>::edge_type* out_neighbors) {
  vertex->out_nghs = out_neighbors;
}

// Note that modifying or updating the out-neighbors using SetOutNeighbors will
// make in-neighbors stale (if they are set) or will leave the in-neighbors
// unset, making it unsafe to use direction optimization.
template <typename Weight>
inline void SetOutNeighbors(
    gbbs::symmetric_vertex<Weight>* vertex,
    typename gbbs::symmetric_vertex<Weight>::edge_type* out_neighbors) {
  vertex->neighbors = out_neighbors;
}

inline void SetEdge(std::tuple<gbbs::uintE, gbbs::empty>* edge,
                    std::pair<NodeId, double> in_edge) {
  *edge = std::make_tuple<gbbs::uintE, gbbs::empty>(in_edge.first, {});
}

inline void SetEdge(std::tuple<gbbs::uintE, float>* edge,
                    std::pair<NodeId, double> in_edge) {
  *edge = std::make_tuple<gbbs::uintE, float>(in_edge.first, in_edge.second);
}

}  // namespace internal

template <typename GraphType>
void GbbsGraphBase<GraphType>::EnsureSize(NodeId id) {
  if (nodes_.size() < id) {
    // Create a default vertex to fill in the gap in the range. Note that there
    // is no guarantee on whether the `id` in the default vertex matches the
    // vertex id (i.e., the offset within nodes_). Instead, we set all node ids
    // to their final correct values inside `FinishImport`.
    auto vertex = typename GraphType::vertex();
    internal::SetId(&vertex, id);
    internal::SetOutDegree(&vertex, 0);
    internal::SetOutNeighbors(&vertex, nullptr);
    nodes_.resize(id, vertex);

    // Also resize edges_ and node_weights_.
    edges_.resize(id);
    node_weights_.resize(id, kDefaultNodeWeight);
    node_parts_.resize(id, kDefaultNodePartId);
  }
}

template <typename GraphType>
absl::Status GbbsGraphBase<GraphType>::PrepareImport(const int64_t num_nodes) {
  if (num_nodes > std::numeric_limits<NodeId>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Total number of nodes exceeds limit: ", num_nodes));
  }
  if (num_nodes < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Total number of nodes cannot be negative: ", num_nodes));
  }
  if (!nodes_.empty() || !edges_.empty() || !node_weights_.empty()) {
    return absl::FailedPreconditionError(
        "PrepareImport is called with preexisting nodes or edges.");
  }
  num_nodes_ = static_cast<NodeId>(num_nodes);
  EnsureSize(num_nodes_);
  return absl::OkStatus();
}

template <typename GraphType>
absl::Status GbbsGraphBase<GraphType>::Import(AdjacencyList adjacency_list) {
  using GbbsEdge = std::tuple<gbbs::uintE, typename GraphType::weight_type>;
  auto outgoing_edges_size = adjacency_list.outgoing_edges.size();
  auto out_neighbors = std::make_unique<GbbsEdge[]>(outgoing_edges_size);
  // TODO: Reevaluate using gbbs::parallel_for here once homegrown
  // scheduler changes are complete.  Code was:
  //   gbbs::parallel_for(0, outgoing_edges_size, [&](size_t i) {
  //     out_neighbors[i] ...;
  //   });
  for (size_t i = 0; i < outgoing_edges_size; ++i) {
    internal::SetEdge(&out_neighbors[i], adjacency_list.outgoing_edges[i]);
  }

  auto update_func = [this, &adjacency_list, &out_neighbors,
                      outgoing_edges_size]() {
    internal::SetId(&nodes_[adjacency_list.id], adjacency_list.id);
    internal::SetOutDegree(&nodes_[adjacency_list.id], outgoing_edges_size);
    internal::SetOutNeighbors(&nodes_[adjacency_list.id], out_neighbors.get());

    edges_[adjacency_list.id] = std::move(out_neighbors);
    node_weights_[adjacency_list.id] = adjacency_list.weight;
    if (adjacency_list.part.has_value()) {
      node_parts_[adjacency_list.id] = *adjacency_list.part;
    }
  };

  if (num_nodes_ == 0) {
    // No prior knowledge on the total number of nodes. Use regular update with
    // lock.
    absl::MutexLock lock(&mutex_);
    EnsureSize(adjacency_list.id + 1);
    update_func();
  } else {
    // Apply lock-free optimization. This works only if the total number of
    // nodes are given prior to the graph building stage.
    if (adjacency_list.id >= nodes_.size() ||
        adjacency_list.id >= edges_.size()) {
      return absl::FailedPreconditionError(
          "Lock-free optimization requires that there be no dangling edge in "
          "the input graph. Please verify that the input complies with that "
          "requirement. Contact Graph Mining team (gm-clustering@) for "
          "assistance.");
    }
    update_func();
  }

  return absl::OkStatus();
}

template <typename GraphType>
absl::Status GbbsGraphBase<GraphType>::FinishImport() {
  
  auto degrees = parlay::delayed_seq<std::size_t>(
      nodes_.size(), [this](size_t i) { return nodes_[i].out_degree(); });
  auto num_edges = parlay::reduce(parlay::make_slice(degrees));

  auto neighbors = parlay::delayed_seq<gbbs::uintE>(nodes_.size(), [this](
                                                                       size_t
                                                                           i) {
    if (nodes_[i].out_degree() == 0) return gbbs::uintE{0};
    // TODO: Replace std::max_element with parallel reduce.
    auto max_neighbor = std::max_element(
        nodes_[i].out_neighbors().neighbors,
        nodes_[i].out_neighbors().neighbors + nodes_[i].out_degree(),
        [](const std::tuple<gbbs::uintE, typename GraphType::weight_type>& u,
           const std::tuple<gbbs::uintE, typename GraphType::weight_type>& v) {
          return std::get<0>(u) < std::get<0>(v);
        });
    return std::get<0>(*max_neighbor);
  });

  int64_t max_node =
      neighbors.empty()
          ? -1
          : static_cast<int64_t>(parlay::reduce(parlay::make_slice(neighbors),
                                                parlay::maxm<gbbs::uintE>()));

  if (num_nodes_ != 0) {
    // Note that `max_node` is the largest neighbor node id, not necessarily the
    // largest node id overall (because there may exist isolated nodes). Thus
    // the invariant is that, if a total node hint is given, it must be larger
    // than the largest node id.
    //
    // The invariant that the total node hint is larger than the largest center
    // node id is verified in Import.
    if (num_nodes_ < max_node + 1) {
      return absl::FailedPreconditionError(
          "Total number of nodes provided to PrepareImport does not match the "
          "actual number of nodes.");
    }
  } else {
    EnsureSize(max_node + 1);
  }

  // Set all node ids to the correct final value, because `EnsureSize` uses
  // temporary placeholder node ids.
  parlay::parallel_for(0, nodes_.size(), [this](size_t i) {
    nodes_[i].id = i;
  });

  MaybeClearNodeWeights();
  MaybeClearNodeParts();

  // The GBBS graph takes no ownership of nodes / edges
  graph_ = std::make_unique<GraphType>(
      nodes_.size(), num_edges, nodes_.data(),
      /*_deletion_fn=*/[]() {},  // noop deletion_fn
      (node_weights_.empty() ? nullptr : node_weights_.data()));
  return absl::OkStatus();
}

template <typename GraphType>
GraphType* GbbsGraphBase<GraphType>::Graph() const {
  return graph_.get();
}

template <typename GraphType>
bool GbbsGraphBase<GraphType>::MaybeClearNodeWeights() {
  
  auto has_default_node_weight = parlay::delayed_seq<bool>(
      node_weights_.size(),
      [this](size_t i) { return node_weights_[i] == kDefaultNodeWeight; });
  bool should_clear = parlay::reduce(
      parlay::make_slice(has_default_node_weight),
      parlay::make_monoid([](bool lhs, bool rhs) { return lhs && rhs; }, true));

  if (should_clear) {
    // Clear and free memory allocated to node_weights_.
    std::vector<typename GraphType::vertex_weight_type>().swap(node_weights_);
  }

  return should_clear;
}

template <typename GraphType>
bool GbbsGraphBase<GraphType>::MaybeClearNodeParts() {
  
  auto has_default_node_part_id = parlay::delayed_seq<bool>(
      node_parts_.size(),
      [this](size_t i) { return node_parts_[i] == kDefaultNodePartId; });
  bool should_clear = parlay::reduce(
      parlay::make_slice(has_default_node_part_id),
      parlay::make_monoid([](bool lhs, bool rhs) { return lhs && rhs; }, true));

  if (should_clear) {
    // Clear and free memory allocated to node_parts_.
    std::vector<NodePartId>().swap(node_parts_);
  }

  return should_clear;
}

template <typename GraphType>
const std::vector<NodePartId>& GbbsGraphBase<GraphType>::GetNodeParts() const {
  return node_parts_;
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
