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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Contracted graph after HAC. A wrapper class around HacGraph such that
// inactive nodes have node weight -1. Normalized edge weights use node weights
// in the original graph.
class ContractedGraph {
  using NodeId = graph_mining::in_memory::NodeId;

 public:
  explicit ContractedGraph(
      std::unique_ptr<ApproximateSubgraphHacGraph> hac_graph)
      : hac_graph_(std::move(hac_graph)) {}

  inline bool IsInactive(NodeId node_id) const {
    return !hac_graph_->IsActive(node_id);
  }

  int64_t NodeWeight(NodeId node_id) const {
    if (IsInactive(node_id)) {
      return -1;
    }
    return hac_graph_->CurrentClusterSize(node_id);
  }

  // Return the neighbors of `id` with weights not averaged by cluster
  // size.
  std::vector<std::pair<double, NodeId>> UnnormalizedNeighborsSimilarity(
      NodeId id) const {
    auto neighbor_ids = hac_graph_->Neighbors(id);
    std::vector<std::pair<double, NodeId>> neighbors;
    neighbors.reserve(neighbor_ids.size());
    for (const auto& node_v : neighbor_ids) {
      neighbors.push_back(
          {hac_graph_->EdgeWeightUnnormalized(id, node_v), node_v});
    }
    return neighbors;
  }

  // Return the neighbors of `id` with weights averaged by cluster size.
  std::vector<std::pair<double, NodeId>> Neighbors(NodeId id) const {
    auto neighbor_ids = hac_graph_->Neighbors(id);
    std::vector<std::pair<double, NodeId>> neighbors;
    neighbors.reserve(neighbor_ids.size());
    for (const auto& node_v : neighbor_ids) {
      neighbors.push_back({hac_graph_->EdgeWeight(id, node_v), node_v});
    }
    return neighbors;
  }

  inline std::size_t NumNodes() const { return hac_graph_->NumNodes(); }

 private:
  std::unique_ptr<ApproximateSubgraphHacGraph> hac_graph_;
};

struct SubgraphHacResults {
  using NodeId = graph_mining::in_memory::NodeId;

  SubgraphHacResults(std::vector<std::tuple<NodeId, NodeId, double>> merges,
                     graph_mining::in_memory::Dendrogram dendrogram,
                     std::unique_ptr<ApproximateSubgraphHacGraph> hac_graph,
                     InMemoryClusterer::Clustering clustering)
      : merges(std::move(merges)),
        clustering(std::move(clustering)),
        dendrogram(std::move(dendrogram)) {
    contracted_graph = std::make_unique<ContractedGraph>(std::move(hac_graph));
  }

  // Merges in subgraph HAC, in dendrogram node id (i.e. if there are three
  // nodes and two merges, the merges will be [(0,1,w_1), (2,3,w_2)] where 3
  // represents the merged node (0, 1)). The first NodeId is smaller than the
  // second one.
  std::vector<std::tuple<NodeId, NodeId, double>> merges;
  // Clustering produced by subgraph HAC, in local cluster id.
  InMemoryClusterer::Clustering clustering;
  // Dendrogram produced by subgraph HAC, in local cluster id.
  graph_mining::in_memory::Dendrogram dendrogram;
  // Contracted graph after subgraph HAC. Inactive nodes have empty
  // neighborhoods.
  std::unique_ptr<ContractedGraph> contracted_graph;
};

// A sequential subgraph-centric clustering algorithm that produces a (1+eps)
// approximation of HAC on the input subgraph. See go/terahac-paper for more
// details about this algorithm, and why it focuses on a subgraph (and its 1-hop
// neighborhood).
//
// The inputs to this function consist of:
// - min_merge_similarities: The minimum merged similarity of each node with
//   respect to its dendrogram subtree. I.e., for a cluster c, its
//   min_merge_similarity is the smallest similarity over all merges that
//   occurred to create it. For singleton clusters, this value is +infinity.
//   Note that we expect the similarities to satisfy the notion of "goodness"
//   defined in Definition 2 of go/terahac-paper, i.e., for a node v,
//   BestEdge(v) / min_merge_similarities[v] <= (1+epsilon). The algorithm will
//   still run without errors even if the similarities violate this definition.
// - epsilon (the accuracy parameter for subgraph hac)
// - a graph
//
// Vertices in the subgraph have one of two types:
//   1. an active vertex (i.e., a vertex assigned to this subgraph that is
//      eligible for being merged)
//   2. an inactive vertex (a neighbor of some active vertex that is ineligible
//      for clustering)
// A node is considred active iff graph->NodeWeight(v) >= 0.
// Note that min_merge_similarities values of inactive nodes are ignored and can
// be arbitrary.
//
// The return value of this function is a SubgraphHacResults struct, including
// the merges, a Clustering, a dendrogram represented as a
// graph_mining::in_memory::Dendrogram and the contracted graph. The function
// only clusters active nodes. Note that the DendrogramNode values in the
// Dendrogram for inactive nodes are treated the same way as root nodes. Also,
// the output clustering does not cluster the inactive vertices (it does not
// even emit them as singleton clusters).
absl::StatusOr<SubgraphHacResults> ApproximateSubgraphHac(
    std::unique_ptr<SimpleUndirectedGraph> graph,
    std::vector<double> min_merge_similarities, double epsilon);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_SUBGRAPH_APPROXIMATE_SUBGRAPH_HAC_H_
