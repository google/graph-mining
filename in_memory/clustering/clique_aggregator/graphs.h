#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_GRAPHS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_GRAPHS_H_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "gbbs/bridge.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/clique_aggregator/bitset.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Two implementations of a graph, with the same API, but not using inheritance.
// The goal is to enable templated code that works with both graph
// implementations. Specifically, in the `CliqueAggregatorClusterer`
// implementation, we use the `BitSetGraph` for recursive calls (since in a
// recursive call the graph typically has <= 200 nodes), and the
// `GbbsGraphWrapper` for the outmost call (when the graphs may be very large).

// Represents the graph as a vector of bitsets. Hence, uses roughly n*n/8 bytes
// of memory to store an n-node graph.
// Note: This implementation does not validate the arguments passed to the
// methods.
class BitSetGraph {
 public:
  BitSetGraph() = delete;

  explicit BitSetGraph(int num_nodes)
      : num_nodes_(num_nodes), adjacency_lists_(num_nodes, BitSet(num_nodes)) {}

  // Adds a directed edge from `node_id1` to `node_id2`. Does nothing if the
  // edge already exists.
  void AddEdge(NodeId node_id1, NodeId node_id2) {
    adjacency_lists_[node_id1].Insert(node_id2);
  }

  int NumNodes() const { return num_nodes_; }

  int Degree(NodeId node_id) const { return adjacency_lists_[node_id].Size(); }

  template <typename F>
  void MapNeighbors(NodeId node_id, F&& f) const {
    adjacency_lists_[node_id].MapElements(f);
  }

  // Calls f(common) for each common neighbor of `node_id1` and `node_id2`.
  template <typename F>
  void MapCommonNeighbors(NodeId node_id1, NodeId node_id2, F&& f) const {
    adjacency_lists_[node_id1].MapCommonElements(adjacency_lists_[node_id2], f);
  }

  // Calls f(common) for each neighbor of `node_id` that is present in `other`.
  // The universe size of other must be equal to `num_nodes_`.
  template <typename F>
  void MapCommonNeighbors(NodeId node_id, const BitSet& other, F&& f) const {
    adjacency_lists_[node_id].MapCommonElements(other, f);
  }

  // Returns true if all neighbors of `node_id` are also in `other`. The
  // universe size of `other` must be equal to `num_nodes_`.
  bool NeighborhoodIsSupersetOf(NodeId node_id, const BitSet& other) const {
    return adjacency_lists_[node_id].IsSupersetOf(other);
  }

 private:
  int num_nodes_;
  std::vector<BitSet> adjacency_lists_;
};

// An owning or not-owning (depending on the constructor used) wrapper around
// `DirectedUnweightedOutEdgesGbbsGraph` or `UnweightedGbbsGraph`.
template <typename GbbsGraphT>
class GbbsGraphWrapper {
 public:
  static_assert(
      std::is_same_v<GbbsGraphT, UnweightedGbbsGraph> ||
          std::is_same_v<GbbsGraphT, DirectedUnweightedOutEdgesGbbsGraph>,
      "GbbsGraphT must be UnweightedGbbsGraph or "
      "DirectedUnweightedOutEdgesGbbsGraph");
  // Construct a wrapper without taking the ownership of `graph`.
  explicit GbbsGraphWrapper(const GbbsGraphT& graph) : graph_(graph) {}

  // Construct a wrapper and takes ownership of `graph`.
  explicit GbbsGraphWrapper(absl_nonnull std::unique_ptr<GbbsGraphT> graph)
      : owned_graph_(std::move(graph)), graph_(*owned_graph_) {}

  int NumNodes() const { return graph_.Graph()->n; }

  int Degree(NodeId node_id) const {
    return graph_.Graph()->get_vertex(node_id).out_degree();
  }

  template <typename F>
  void MapNeighbors(NodeId node_id, F&& f) const {
    graph_.Graph()->get_vertex(node_id).out_neighbors().map(
        [&](const gbbs::uintE& src, const gbbs::uintE& ngh,
            const gbbs::empty& w) { f(ngh); },
        /*parallel=*/false);
  }

  template <typename F>
  void MapCommonNeighbors(NodeId node_id1, NodeId node_id2, F&& f) const {
    auto common_neighbor_found = [&](gbbs::uintE a, gbbs::uintE b,
                                     gbbs::uintE common) { f(common); };
    auto ngh_neighbors = graph_.Graph()->get_vertex(node_id1).out_neighbors();
    graph_.Graph()->get_vertex(node_id2).out_neighbors().intersect_f(
        &ngh_neighbors, common_neighbor_found);
  }

  template <typename F>
  void MapCommonNeighbors(NodeId node_id, const BitSet& other, F&& f) const {
    graph_.Graph()->get_vertex(node_id).out_neighbors().map(
        [&](const gbbs::uintE& src, const gbbs::uintE& ngh,
            const gbbs::empty& w) {
          if (other.Contains(ngh)) f(ngh);
        },
        /*parallel=*/false);
  }

  bool NeighborhoodIsSupersetOf(NodeId node_id, const BitSet& other) const {
    int contained = 0;
    graph_.Graph()->get_vertex(node_id).out_neighbors().map(
        [&](const gbbs::uintE& src, const gbbs::uintE& ngh,
            const gbbs::empty& w) {
          if (other.Contains(ngh)) ++contained;
        },
        /*parallel=*/false);
    return contained == other.Size();
  }

 private:
  // If the wrapper owns the graph, this field is set. Otherwise, it is nullptr.
  absl_nullable std::unique_ptr<GbbsGraphT> owned_graph_;
  const GbbsGraphT& graph_;
};

// Returns the number of *directed* edges in the graph, where `GraphT` is one of
// the graph types defined above.
template <typename GraphT>
int64_t NumEdges(const GraphT& graph) {
  static_assert(
      std::is_same_v<GraphT, BitSetGraph> ||
          std::is_same_v<GraphT, GbbsGraphWrapper<UnweightedGbbsGraph>> ||
          std::is_same_v<GraphT,
                         GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>,
      "");
  int64_t num_edges = 0;
  for (NodeId i = 0; i < graph.NumNodes(); ++i) num_edges += graph.Degree(i);
  return num_edges;
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_GRAPHS_H_
