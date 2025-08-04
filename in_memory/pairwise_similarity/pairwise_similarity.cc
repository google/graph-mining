#include "in_memory/pairwise_similarity/pairwise_similarity.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gbbs/bridge.h"
#include "in_memory/clustering/types.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

absl::StatusOr<PairwiseSimilarity::SimilarityMatrix>
PairwiseSimilarity::Compute(absl::Span<const NodeId> source_nodes,
                            absl::Span<const NodeId> target_nodes) const {
  if (graph_.Graph() == nullptr) {
    return absl::InvalidArgumentError("No graph provided.");
  }

  NodeId num_nodes = graph_.Graph()->n;

  if (num_nodes < 1) {
    return absl::InvalidArgumentError("Graph must have at least 1 node.");
  }

  if (source_nodes.empty()) {
    return absl::InvalidArgumentError("Source nodes must not be empty.");
  }

  if (target_nodes.empty()) {
    return absl::InvalidArgumentError("Target nodes must not be empty.");
  }

  switch (config_.metric()) {
    case PairwiseSimilarityConfig::METRIC_UNSPECIFIED:
      return absl::InvalidArgumentError("Metric unspecified.");
    case PairwiseSimilarityConfig::JACCARD:
      return Jaccard(source_nodes, target_nodes);
    case PairwiseSimilarityConfig::COSINE:
      return Cosine(source_nodes, target_nodes);
    case PairwiseSimilarityConfig::COMMON:
      return Common(source_nodes, target_nodes);
    case PairwiseSimilarityConfig::TOTAL:
      return Total(source_nodes, target_nodes);
    default:
      return absl::InvalidArgumentError("Unknown metric requested.");
  }
}

PairwiseSimilarity::SimilarityMatrix PairwiseSimilarity::Jaccard(
    absl::Span<const NodeId> source_nodes,
    absl::Span<const NodeId> target_nodes) const {
  size_t src_size = source_nodes.size();
  size_t tgt_size = target_nodes.size();

  SimilarityMatrix result(src_size);
  for (auto& v : result) {
    v.resize(tgt_size);
  }

  parlay::parallel_for(0, src_size, [&](size_t src_id) {
    absl::flat_hash_set<NodeId> src_id_neighbors =
        CollectOutNeighbors(source_nodes[src_id]);
    parlay::parallel_for(0, tgt_size, [&](size_t tgt_id) {
      NodeId common_count =
          IntersectCount(src_id_neighbors, target_nodes[tgt_id]);
      NodeId tgt_id_neighbors_count = GetOutDegree(target_nodes[tgt_id]);
      NodeId total_count =
          src_id_neighbors.size() + tgt_id_neighbors_count - common_count;
      WeightT jaccard = 0.0;
      // If `total_count == 0` then both neighbor sets are empty, in which case
      // Jaccard similarity is undefined, so we return NaN.
      if (total_count > 0) {
        jaccard = static_cast<WeightT>(common_count) / total_count;
      } else {
        jaccard = std::numeric_limits<WeightT>::quiet_NaN();
      }
      result[src_id][tgt_id] = jaccard;
    });
  });

  return result;
}

PairwiseSimilarity::SimilarityMatrix PairwiseSimilarity::Cosine(
    absl::Span<const NodeId> source_nodes,
    absl::Span<const NodeId> target_nodes) const {
  size_t src_size = source_nodes.size();
  size_t tgt_size = target_nodes.size();

  SimilarityMatrix result(src_size);
  for (auto& v : result) {
    v.resize(tgt_size);
  }

  parlay::parallel_for(0, src_size, [&](size_t src_id) {
    absl::flat_hash_map<NodeId, WeightT> src_id_neighbors =
        CollectOutNeighborsAndWeights(source_nodes[src_id]);
    WeightT src_id_length = GetLength(source_nodes[src_id]);
    parlay::parallel_for(0, tgt_size, [&](size_t tgt_id) {
      WeightT product = WeightsProduct(src_id_neighbors, target_nodes[tgt_id]);
      WeightT length_product = src_id_length * GetLength(target_nodes[tgt_id]);
      // If the length of either vectors is zero then the cosine similarity is
      // undefined, so we return NaN.
      if (length_product == 0.0) {
        result[src_id][tgt_id] = std::numeric_limits<WeightT>::quiet_NaN();
      } else {
        result[src_id][tgt_id] = product / length_product;
      }
    });
  });

  return result;
}

PairwiseSimilarity::SimilarityMatrix PairwiseSimilarity::Common(
    absl::Span<const NodeId> source_nodes,
    absl::Span<const NodeId> target_nodes) const {
  size_t src_size = source_nodes.size();
  size_t tgt_size = target_nodes.size();

  SimilarityMatrix result(src_size);
  for (auto& v : result) {
    v.resize(tgt_size);
  }

  parlay::parallel_for(0, src_size, [&](size_t src_id) {
    absl::flat_hash_set<NodeId> src_id_neighbors =
        CollectOutNeighbors(source_nodes[src_id]);
    parlay::parallel_for(0, tgt_size, [&](size_t tgt_id) {
      NodeId common_count =
          IntersectCount(src_id_neighbors, target_nodes[tgt_id]);
      result[src_id][tgt_id] = static_cast<WeightT>(common_count);
    });
  });

  return result;
}

PairwiseSimilarity::SimilarityMatrix PairwiseSimilarity::Total(
    absl::Span<const NodeId> source_nodes,
    absl::Span<const NodeId> target_nodes) const {
  size_t src_size = source_nodes.size();
  size_t tgt_size = target_nodes.size();

  SimilarityMatrix result(src_size);
  for (auto& v : result) {
    v.resize(tgt_size);
  }

  parlay::parallel_for(0, src_size, [&](size_t src_id) {
    absl::flat_hash_set<NodeId> src_id_neighbors =
        CollectOutNeighbors(source_nodes[src_id]);
    parlay::parallel_for(0, tgt_size, [&](size_t tgt_id) {
      NodeId common_count =
          IntersectCount(src_id_neighbors, target_nodes[tgt_id]);
      NodeId tgt_id_neighbors_count = GetOutDegree(target_nodes[tgt_id]);
      WeightT total_count =
          src_id_neighbors.size() + tgt_id_neighbors_count - common_count;
      result[src_id][tgt_id] = static_cast<WeightT>(total_count);
    });
  });

  return result;
}

bool PairwiseSimilarity::CheckEdgeWeights() const {
  const auto& graph = *graph_.Graph();
  bool valid = true;

  graph.mapEdges([&valid](auto src_id, auto target_id, WeightT weight) {
    if (weight == std::numeric_limits<WeightT>::infinity()) {
      gbbs::atomic_store(&valid, false);
    }
  });

  return valid;
}

absl::flat_hash_set<NodeId> PairwiseSimilarity::CollectOutNeighbors(
    NodeId node_id) const {
  absl::flat_hash_set<NodeId> result;
  auto neighbors = graph_.Graph()->get_vertex(node_id).out_neighbors();
  for (int loop = 0; loop < neighbors.get_degree(); ++loop) {
    auto neighbor_id = neighbors.get_neighbor(loop);
    result.insert(neighbor_id);
  }
  return result;
}

absl::flat_hash_map<NodeId, PairwiseSimilarity::WeightT>
PairwiseSimilarity::CollectOutNeighborsAndWeights(NodeId node_id) const {
  absl::flat_hash_map<NodeId, WeightT> result;
  auto neighbors = graph_.Graph()->get_vertex(node_id).out_neighbors();
  for (int loop = 0; loop < neighbors.get_degree(); ++loop) {
    auto neighbor_id = neighbors.get_neighbor(loop);
    auto neighbor_weight = neighbors.get_weight(loop);
    result[neighbor_id] = neighbor_weight;
  }
  return result;
}

NodeId PairwiseSimilarity::IntersectCount(
    const absl::flat_hash_set<NodeId>& src_set, NodeId node_id) const {
  NodeId result = 0;
  auto neighbors = graph_.Graph()->get_vertex(node_id).out_neighbors();
  for (int loop = 0; loop < neighbors.get_degree(); ++loop) {
    auto neighbor_id = neighbors.get_neighbor(loop);
    if (src_set.contains(neighbor_id)) {
      ++result;
    }
  }
  return result;
}

NodeId PairwiseSimilarity::GetOutDegree(NodeId node_id) const {
  return graph_.Graph()->get_vertex(node_id).out_degree();
}

PairwiseSimilarity::WeightT PairwiseSimilarity::GetLength(
    NodeId node_id) const {
  WeightT result = 0.0;
  auto neighbors = graph_.Graph()->get_vertex(node_id).out_neighbors();
  for (int loop = 0; loop < neighbors.get_degree(); ++loop) {
    auto neighbor_weight = neighbors.get_weight(loop);
    result += neighbor_weight * neighbor_weight;
  }
  return std::sqrt(result);
}

PairwiseSimilarity::WeightT PairwiseSimilarity::WeightsProduct(
    const absl::flat_hash_map<NodeId, WeightT>& src_set, NodeId node_id) const {
  WeightT result = 0.0;
  auto neighbors = graph_.Graph()->get_vertex(node_id).out_neighbors();
  for (int loop = 0; loop < neighbors.get_degree(); ++loop) {
    NodeId neighbor_id = neighbors.get_neighbor(loop);
    auto iter = src_set.find(neighbor_id);
    if (iter == src_set.end()) {
      continue;
    }
    WeightT src_weight = iter->second;
    auto neighbor_weight = neighbors.get_weight(loop);
    result += neighbor_weight * src_weight;
  }
  return result;
}

}  // namespace graph_mining::in_memory
