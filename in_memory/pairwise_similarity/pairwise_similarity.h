#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAIRWISE_SIMILARITY_PAIRWISE_SIMILARITY_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAIRWISE_SIMILARITY_PAIRWISE_SIMILARITY_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"
#include "in_memory/pairwise_similarity/pairwise_similarity.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

class PairwiseSimilarity {
 public:
  using WeightT = DirectedOutEdgesGbbsGraph::Weight;
  using SimilarityMatrix = std::vector<std::vector<WeightT>>;

  explicit PairwiseSimilarity(const PairwiseSimilarityConfig& config)
      : config_(config) {}

  // PairwiseSimilarity is neither copyable nor movable.
  PairwiseSimilarity(const PairwiseSimilarity&) = delete;
  PairwiseSimilarity& operator=(const PairwiseSimilarity&) = delete;

  // Returns a pointer to the underlying graph.
  InMemoryClusterer::Graph* MutableGraph() { return &graph_; }

  // Computes the pairwise similarity between the source and target nodes. The
  // input graph must have at least one node and non-infinite edge weights. The
  // source and target nodes must be valid node ids in the graph.
  absl::StatusOr<SimilarityMatrix> Compute(
      absl::Span<const NodeId> source_nodes,
      absl::Span<const NodeId> target_nodes) const;

 private:
  // Computes the Jaccard similarity between the source and target nodes. For a
  // given source, target node pair, if both the neighbor sets are empty then
  // Jaccard similarity is undefined and we return quiet_NaN(). Otherwise, the
  // similarity is defined as the size of the intersection divided by the size
  // of the union of the two neighbor sets. Edge weights are not used.
  SimilarityMatrix Jaccard(absl::Span<const NodeId> source_nodes,
                           absl::Span<const NodeId> target_nodes) const;

  // Computes the Cosine similarity between the source and target nodes. Views
  // the neighbor sets as vectors of length N, where the weight of the i-th
  // position is the edge weight of the i-th neighbor of the source node if such
  // a neighbor exists, and zero otherwise. The cosine similarity is defined as
  // the dot product of these vectors divided by the L2 norm of both vectors. If
  // the L2 norm of either vector is 0 then the cosine similarity is undefined
  // and we return quiet_NaN(). Edge weights are used.
  SimilarityMatrix Cosine(absl::Span<const NodeId> source_nodes,
                          absl::Span<const NodeId> target_nodes) const;

  // Computes the Common neighbors similarity between the source and target
  // nodes. Returns the number of common nodes between the source and target
  // neighbor sets. Edge weights are not used.
  SimilarityMatrix Common(absl::Span<const NodeId> source_nodes,
                          absl::Span<const NodeId> target_nodes) const;

  // Computes the Total neighbors similarity between the source and target
  // nodes. Returns the number of unique nodes in the source and target neighbor
  // sets. Edge weights are not used. Note that if node degrees are fixed, this
  // behaves as a "dissimilarity" function, i.e., the larger the value, the less
  // similar the nodes are.
  SimilarityMatrix Total(absl::Span<const NodeId> source_nodes,
                         absl::Span<const NodeId> target_nodes) const;

  // Returns true if all edge weights are non-infinite, false otherwise.
  bool CheckEdgeWeights() const;

  // Collects the out-neighbors of a given node.
  absl::flat_hash_set<NodeId> CollectOutNeighbors(NodeId node_id) const;

  // Collects the out-neighbors and their edge weights of a given node.
  absl::flat_hash_map<NodeId, WeightT> CollectOutNeighborsAndWeights(
      NodeId node_id) const;

  // Computes the number of common neighbors between a source set and a target
  // node.
  NodeId IntersectCount(const absl::flat_hash_set<NodeId>& src_set,
                        NodeId node_id) const;

  // Returns the out-degree of a given node.
  NodeId GetOutDegree(NodeId node_id) const;

  // Computes the L2 norm of the edge weights of the out-neighbors of a given
  // node.
  WeightT GetLength(NodeId node_id) const;

  // Computes the product of weights of common neighbors between a source set
  // and a target node.
  WeightT WeightsProduct(const absl::flat_hash_map<NodeId, WeightT>& src_set,
                         NodeId node_id) const;

  DirectedOutEdgesGbbsGraph graph_;
  PairwiseSimilarityConfig config_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAIRWISE_SIMILARITY_PAIRWISE_SIMILARITY_H_
