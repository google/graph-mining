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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "gbbs/bridge.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/parallel/streaming_writer.h"
#include "in_memory/status_macros.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

// A wrapper for GBBS graph types. Should be used with GraphType =
// gbbs::X_ptr_graph<gbbs::X_vertex, Y>, where
//  X is either symmetric or asymmetric
//  Y specifies the weight type
// Setting Y = gbbs::empty gives an unweighted graph.
//
// Note that this graph container only materializes the out-edges.
// - If X = symmetric, then the out-edges are the same as the in-edges and so
// both are materialized.
// - If X = asymmetric, then the out-edges and in-edges of each vertex are
// different, and representing both would double the memory usage of the graph
// container. This container chooses to only materialize the out-edges, which
// is sufficient for many applications. If in-edges are also required, see the
// class DirectedGbbsInOutEdgesGraph below.
//
// Note that GBBS doesn't support node weights.
template <typename GraphType>
class GbbsOutEdgesOnlyGraph : public InMemoryClusterer::Graph {
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

  // Returns true if the input is a valid bipartite graph. Returns false
  // otherwise.
  //
  // Note that for validation purposes, we consider only part ids 0 and 1 to be
  // valid for a bipartite graph.
  bool IsValidBipartiteGraph() const;

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

// Instantiations of the GbbsOutEdgesOnlyGraph. In each case multiple edges and
// self-loops are allowed.
// Adding a new instantiation may require adding overloads in the internal
// namespace below.

// Weighted undirected graph. WARNING: the caller must ensure an undirected
// graph is loaded (see comment on GbbsOutEdgesOnlyGraph::Import).
class GbbsGraph
    : public GbbsOutEdgesOnlyGraph<
          gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>> {
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

// Directed graph that also materializes the in-edges when FinishImport() is
// called.
template <class Weight>
class DirectedGbbsInOutEdgesGraph
    : public GbbsOutEdgesOnlyGraph<
          gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, Weight>> {
 public:
  // Constructs graph_ using nodes_ and edges_, and optionally node_weights_ and
  // node_parts. Also materializes the in-edges.
  absl::Status FinishImport() override;

 protected:
  using GraphType = gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, Weight>;
  // Maintains in-edges.
  std::vector<std::unique_ptr<std::tuple<gbbs::uintE, Weight>[]>> in_edges_;
};

using DirectedUnweightedGbbsGraph = DirectedGbbsInOutEdgesGraph<gbbs::empty>;

using DirectedGbbsGraph = DirectedGbbsInOutEdgesGraph<float>;

using DirectedUnweightedOutEdgesGbbsGraph = GbbsOutEdgesOnlyGraph<
    gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, gbbs::empty>>;

using DirectedOutEdgesGbbsGraph = GbbsOutEdgesOnlyGraph<
    gbbs::asymmetric_ptr_graph<gbbs::asymmetric_vertex, float>>;

using UnweightedGbbsGraph = GbbsOutEdgesOnlyGraph<
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, gbbs::empty>>;

// Weighted undirected graph with sorted neighbors.
class UnweightedSortedNeighborGbbsGraph
    : public GbbsOutEdgesOnlyGraph<
          gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, gbbs::empty>> {
 public:
  absl::Status Import(AdjacencyList adjacency_list) override;
};

//////////////////////////////////////////////////////////////////////////////
/// END OF PUBLIC API. IMPLEMENTATION ONLY BELOW
//////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename Weight>
inline void SetId(gbbs::asymmetric_vertex<Weight>* vertex, NodeId id) {
  vertex->id = id;
}

template <typename Weight>
inline void SetId(gbbs::symmetric_vertex<Weight>* vertex, NodeId id) {
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
    typename gbbs::asymmetric_vertex<Weight>::neighbor_type* out_neighbors) {
  vertex->out_nghs = out_neighbors;
}

// Note that modifying or updating the out-neighbors using SetOutNeighbors will
// make in-neighbors stale (if they are set) or will leave the in-neighbors
// unset, making it unsafe to use direction optimization.
template <typename Weight>
inline void SetOutNeighbors(
    gbbs::symmetric_vertex<Weight>* vertex,
    typename gbbs::symmetric_vertex<Weight>::neighbor_type* out_neighbors) {
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
void GbbsOutEdgesOnlyGraph<GraphType>::EnsureSize(NodeId id) {
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
absl::Status GbbsOutEdgesOnlyGraph<GraphType>::PrepareImport(
    const int64_t num_nodes) {
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
absl::Status GbbsOutEdgesOnlyGraph<GraphType>::Import(
    AdjacencyList adjacency_list) {
  if (adjacency_list.id < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Node ID cannot be negative: ", adjacency_list.id));
  }
  using GbbsEdge = std::tuple<gbbs::uintE, typename GraphType::weight_type>;
  auto outgoing_edges_size = adjacency_list.outgoing_edges.size();
  auto out_neighbors = std::make_unique<GbbsEdge[]>(outgoing_edges_size);
  // TODO: Reevaluate using gbbs::parallel_for here once homegrown
  // scheduler changes are complete.  Code was:
  //   gbbs::parallel_for(0, outgoing_edges_size, [&](size_t i) {
  //     out_neighbors[i] ...;
  //   });
  NodeId maximum_node_id_in_adjacency_list = adjacency_list.id;
  for (size_t i = 0; i < outgoing_edges_size; ++i) {
    NodeId neighbor_id = adjacency_list.outgoing_edges[i].first;
    if (neighbor_id < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Neighbor ID cannot be negative: ", neighbor_id));
    }
    maximum_node_id_in_adjacency_list =
        std::max(maximum_node_id_in_adjacency_list, neighbor_id);
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
    EnsureSize(maximum_node_id_in_adjacency_list + 1);
    update_func();
  } else {
    // Apply lock-free optimization. This works only if the total number of
    // nodes are given prior to the graph building stage.
    if (maximum_node_id_in_adjacency_list >= nodes_.size() ||
        maximum_node_id_in_adjacency_list >= edges_.size()) {
      return absl::OutOfRangeError(absl::StrCat(
          "Attempted to insert edge incident to a node with invalid ID: ",
          maximum_node_id_in_adjacency_list,
          " is no smaller than the explicitly provided number of nodes (",
          num_nodes_, ")"));
    }
    update_func();
  }

  return absl::OkStatus();
}

template <typename GraphType>
absl::Status GbbsOutEdgesOnlyGraph<GraphType>::FinishImport() {
  
  auto degrees = parlay::delayed_seq<std::size_t>(
      nodes_.size(), [this](size_t i) { return nodes_[i].out_degree(); });
  auto num_edges = parlay::reduce(parlay::make_slice(degrees));

  // Set all node ids to the correct final value, because `EnsureSize` uses
  // temporary placeholder node ids.
  parlay::parallel_for(0, nodes_.size(),
                       [this](size_t i) { nodes_[i].id = i; });

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
GraphType* GbbsOutEdgesOnlyGraph<GraphType>::Graph() const {
  return graph_.get();
}

template <typename GraphType>
bool GbbsOutEdgesOnlyGraph<GraphType>::MaybeClearNodeWeights() {
  
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
bool GbbsOutEdgesOnlyGraph<GraphType>::MaybeClearNodeParts() {
  
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
const std::vector<NodePartId>& GbbsOutEdgesOnlyGraph<GraphType>::GetNodeParts()
    const {
  return node_parts_;
}

template <typename GraphType>
bool GbbsOutEdgesOnlyGraph<GraphType>::IsValidBipartiteGraph() const {
  auto get_node_part = [&](gbbs::uintE i) {
    return i < node_parts_.size() ? node_parts_[i] : 0;
  };

  auto valid_seq = parlay::delayed_seq<bool>(graph_->n, [&](gbbs::uintE i) {
    auto node_part = get_node_part(i);
    std::atomic<bool> valid = true;
    if (node_part != 0 && node_part != 1) {
      valid = false;
    } else {
      auto expected_neighbor_part = 1 - node_part;
      auto validate_neighbor_parts =
          [&](gbbs::uintE u, gbbs::uintE neighbor,
              typename GraphType::weight_type weight) {
            if (get_node_part(neighbor) != expected_neighbor_part) {
              valid = false;
            }
          };
      graph_->get_vertex(i).out_neighbors().map(validate_neighbor_parts);
    }
    return valid.load();
  });
  return parlay::reduce(
      valid_seq,
      parlay::make_monoid([](bool a, bool b) { return a && b; }, true));
}

template <typename Weight>
absl::Status DirectedGbbsInOutEdgesGraph<Weight>::FinishImport() {
  RETURN_IF_ERROR((GbbsOutEdgesOnlyGraph<gbbs::asymmetric_ptr_graph<
                       gbbs::asymmetric_vertex, Weight>>::FinishImport()));
  if (this->graph_ == nullptr) {
    // This should never happen. The call to
    // `GbbsOutEdgesOnlyGraph::FinishImport` should have created the graph.
    return absl::InternalError("'graph_' is null");
  }

  in_edges_.resize(this->graph_->n);

  
  constexpr int kPerThreadBufferSize = 1024;

  // in-edges buffer.
  using Edge = std::tuple<gbbs::uintE, gbbs::uintE, Weight>;
  graph_mining::in_memory::StreamingWriter<Edge> edge_buffer(
      kPerThreadBufferSize);
  parlay::parallel_for(0, this->graph_->n, [&](gbbs::uintE i) {
    auto neighbors = this->graph_->get_vertex(i).out_neighbors();
    neighbors.map(
        [&](const gbbs::uintE& our_id, const gbbs::uintE& neighbor_id,
            const Weight& weight) {
          edge_buffer.Add({neighbor_id, our_id, weight});
        },
        /*parallel=*/false);
  });

  parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, Weight>> edges =
      graph_mining::in_memory::Flatten(edge_buffer.Build());
  CHECK_EQ(edges.size(), this->graph_->m);
  // TODO: Enable tuples containing gbbs::uintE to be correctly
  // compared using std::less, so that the custom comparator shown below is
  // not required.
  parlay::sample_sort_inplace(
      parlay::make_slice(edges), [&](const Edge& left, const Edge& right) {
        return std::tie(std::get<0>(left), std::get<1>(left)) <
               std::tie(std::get<0>(right), std::get<1>(right));
      });

  auto edge_boundaries =
      graph_mining::in_memory::GetBoundaryIndices<std::size_t>(
          edges.size(), [&edges](std::size_t i, std::size_t j) {
            return std::get<0>(edges[i]) == std::get<0>(edges[j]);
          });
  std::size_t num_edge_boundaries = edge_boundaries.size() - 1;
  parlay::parallel_for(0, num_edge_boundaries, [&](std::size_t i) {
    std::size_t start_edge_index = edge_boundaries[i];
    std::size_t end_edge_index = edge_boundaries[i + 1];
    std::size_t in_edges_size = end_edge_index - start_edge_index;

    auto in_neighbors =
        std::make_unique<std::tuple<gbbs::uintE, Weight>[]>(in_edges_size);
    for (std::size_t j = 0; j < in_edges_size; ++j) {
      in_neighbors[j] = {std::get<1>(edges[start_edge_index + j]),
                         std::get<2>(edges[start_edge_index + j])};
    }

    gbbs::uintE center_node_id = std::get<0>(edges[start_edge_index]);
    in_edges_[center_node_id] = std::move(in_neighbors);
    auto& center_node = this->graph_->vertices[center_node_id];

    // `id`, `out_deg`, and `out_nghs` are all set by FinishImport. We only need
    // to adjust `in_deg` and `in_nghs`.
    center_node.in_deg = in_edges_size;
    center_node.in_nghs = in_edges_[center_node_id].get();
  });
  return absl::OkStatus();
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_H_
