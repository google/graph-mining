#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_DEGENERACY_ORIENTATION_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_DEGENERACY_ORIENTATION_H_

#include <bit>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/types/span.h"
#include "in_memory/clustering/clique_aggregator/graphs.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"
#include "utils/container/fixed_size_priority_queue.h"

namespace graph_mining {
namespace in_memory {

// Computes a degeneracy ordering of the nodes in the graph.
// A degeneracy ordering is an ordering of the nodes such that each node has
// the minimum possible degree in the subgraph induced by itself and the nodes
// that come after it in the ordering.
// `GraphT` should be one of the types defined in
// https://github.com/google/graph-mining/tree/main/in_memory/clustering/clique_aggregator/graphs.h
// (`BitSetGraph` or `GbbsGraphWrapper`).
template <typename GraphT>
std::vector<NodeId> DegeneracyOrdering(const GraphT& graph) {
  std::vector<NodeId> degeneracy_ordering;
  degeneracy_ordering.reserve(graph.NumNodes());
  std::vector<bool> removed(graph.NumNodes(), false);

  // For each node, we insert it into the queue with priority equal to its
  // out-degree. This ensures that the `Top()` method of the queue returns the
  // node with the smallest out-degree. We use a power-of-two sized queue to
  // ensure that ties are broken by returning the node with the smallest id (see
  // comment on `FixedSizePriorityQueue::Top`).
  FixedSizePriorityQueue<NodeId, NodeId, std::less<NodeId>> queue(std::bit_ceil(
      static_cast<std::make_unsigned_t<NodeId>>(graph.NumNodes())));
  for (NodeId node_id = 0; node_id < graph.NumNodes(); ++node_id) {
    queue.InsertOrUpdate(node_id, graph.Degree(node_id));
  }
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    auto node_id = queue.Top();
    queue.Remove(node_id);
    removed[node_id] = true;
    degeneracy_ordering.push_back(node_id);

    graph.MapNeighbors(node_id, [&](NodeId neighbor_id) {
      if (removed[neighbor_id]) return;
      queue.InsertOrUpdate(neighbor_id, queue.Priority(neighbor_id) - 1);
    });
  }
  return degeneracy_ordering;
}

// Orients the edges of the graph according to the given ordering.
// An edge {u, v} is directed from u to v if u appears before v in the
// ordering. Self-loops are removed.
// Returns a pair of graphs, where the first graph is the directed graph and
// the second graph is the transposed graph.
// If the ordering is invalid, i.e., not a permutation of [0, num_nodes),
// returns {nullptr, nullptr}.
std::pair<absl_nullable std::unique_ptr<BitSetGraph>,
          absl_nullable std::unique_ptr<BitSetGraph>>
DirectGraph(const BitSetGraph& graph, absl::Span<const NodeId> ordering);

// Overload of `DirectGraph` for `GbbsGraphWrapper`.
// The adjacency lists of the returned graphs are sorted in ascending order by
// the target node ID.
std::pair<absl_nullable std::unique_ptr<
              GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>,
          absl_nullable std::unique_ptr<
              GbbsGraphWrapper<DirectedUnweightedOutEdgesGbbsGraph>>>
DirectGraph(const GbbsGraphWrapper<UnweightedGbbsGraph>& graph,
            absl::Span<const NodeId> ordering);

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_DEGENERACY_ORIENTATION_H_
