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

#include "in_memory/clustering/dendrogram_test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "in_memory/clustering/compress_graph.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using NodeId = SimpleUndirectedGraph::NodeId;

namespace internal {

// Given an input graph, graph, and a mapping from the vertices to clusters,
// cluster_ids, this function computes the graph induced by cluster_ids, merging
// edges between two components using the unweighted average-linkage (UPGMA)
// formula (see https://en.wikipedia.org/wiki/UPGMA for more details).
absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>>
GetInducedUnweightedAverageGraph(
    const SimpleUndirectedGraph& graph,
    const std::vector<SimpleUndirectedGraph::NodeId>& cluster_ids) {
  using NodeId = SimpleUndirectedGraph::NodeId;
  // First compress the graph, merging parallel edges using sum aggregation.
  std::unique_ptr<SimpleUndirectedGraph> result;
  ASSIGN_OR_RETURN(
      result,
      graph_mining::in_memory::CompressGraph(
          graph, cluster_ids, [](double compressed_weight, double edge_weight) {
            return compressed_weight + edge_weight;
          }));

  // Compute node weights (the size of each cluster).
  std::vector<NodeId> node_weights(result->NumNodes(), 0);
  for (NodeId i = 0; i < cluster_ids.size(); ++i) {
    node_weights[cluster_ids[i]] += graph.NodeWeight(i);
  }

  // Reweight the edges based on the node weights.
  for (NodeId i = 0; i < result->NumNodes(); ++i) {
    for (const auto& [neighbor_id, similarity] : result->Neighbors(i)) {
      // Process each undirected edge once.
      if (neighbor_id >= i) continue;
      double scaling_factor = node_weights[i] * node_weights[neighbor_id];
      double new_edge_weight = similarity / scaling_factor;
      RETURN_IF_ERROR(result->SetEdgeWeight(i, neighbor_id, new_edge_weight));
      ABSL_CHECK_GT(new_edge_weight, 0);
    }
  }
  return result;
}

double BestNeighborWeight(NodeId node, const SimpleUndirectedGraph& graph) {
  double max_weight = -1;
  for (const auto& [neighbor, weight] : graph.Neighbors(node)) {
    ABSL_CHECK_GT(weight, 0);
    max_weight = std::max(max_weight, weight);
  }
  return max_weight;
}

// Returns the weight of the highest-weight edge in the graph. If the graph only
// contains isolated vertices, a weight of -1 is returned.
double GetBestEdgeWeight(const SimpleUndirectedGraph& graph) {
  using NodeId = SimpleUndirectedGraph::NodeId;
  double max_weight = -1;
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    max_weight = std::max(max_weight, BestNeighborWeight(i, graph));
  }
  return max_weight;
}

}  // namespace internal

double GlobalApproximationFactor(const Dendrogram& dendrogram) {
  double max_global_approx = 1;
  for (size_t node_id = 0; node_id < dendrogram.Nodes().size(); ++node_id) {
    double smallest_similarity = dendrogram.Nodes()[node_id].merge_similarity;
    DendrogramNode node = dendrogram.Nodes()[node_id];
    while (node.parent_id != Dendrogram::kNoParentId) {
      ABSL_CHECK_GT(node.merge_similarity, 0);
      double approx = node.merge_similarity / smallest_similarity;
      if (smallest_similarity != std::numeric_limits<float>::infinity()) {
        max_global_approx = std::max(approx, max_global_approx);
      }
      smallest_similarity =
          std::min(smallest_similarity, node.merge_similarity);

      node = dendrogram.Nodes()[node.parent_id];
    }
  }
  return max_global_approx;
}

double LocalApproximationFactor(const Dendrogram& dendrogram) {
  double max_local_approx = 0;

  // The i-th entry stores the smallest similarity of a merge that creates node
  // i. For leaves, the value is std::numeric_limits<double>::infinity().
  std::vector<double> merge_similarities(
      dendrogram.Nodes().size(), std::numeric_limits<double>::infinity());
  for (const auto& [parent_id, parent_merge] : dendrogram.Nodes()) {
    if (parent_id != Dendrogram::kNoParentId) {
      merge_similarities[parent_id] = parent_merge;
    }
  }

  for (size_t node_id = 0; node_id < dendrogram.Nodes().size(); ++node_id) {
    double our_similarity = merge_similarities[node_id];
    auto node = dendrogram.Nodes()[node_id];
    if (node.parent_id != Dendrogram::kNoParentId) {
      double approx = node.merge_similarity / our_similarity;
      max_local_approx = std::max(approx, max_local_approx);
    }
  }
  return max_local_approx;
}

absl::StatusOr<double> ClosenessApproximationFactor(
    const Dendrogram& dendrogram, const SimpleUndirectedGraph& graph,
    double weight_threshold) {
  NodeId num_nodes = graph.NumNodes();

  // The initial clustering is just the identity clustering, with the cluster
  // ids of internal dendrogram nodes set to -1.
  auto cluster_ids = std::vector<NodeId>(2 * num_nodes - 1, -1);
  for (NodeId node_id = 0; node_id < num_nodes; ++node_id) {
    cluster_ids[node_id] = node_id;
    if (graph.NodeWeight(node_id) != 1) {
      return absl::InvalidArgumentError(
          "ClosenessApproximationFactor does not support graphs with "
          "non-uniform weights.");
    }
  }

  double closeness = 0;
  auto merges = dendrogram.GetMergeSequence();

  int last_relevant_merge = -1;
  for (int i = 0; i < merges.size(); ++i) {
    if (merges[i].merge_similarity >= weight_threshold) last_relevant_merge = i;
  }

  merges.resize(last_relevant_merge + 1);

  for (auto [merge_similarity, node_a, node_b, parent_id, _] : merges) {
    // Check how far from the (current) best similarity in the graph this merge
    // was. This involves computing the graph induced by the current clustering,
    // computing the maximum similarity edge (the best edge) in this graph, and
    // comparing the similarity of the best edge with the edge actually merged
    // by the algorithm.
    std::vector<NodeId> current_clusters(cluster_ids.begin(),
                                         cluster_ids.begin() + num_nodes);
    std::unique_ptr<SimpleUndirectedGraph> current_graph;
    ASSIGN_OR_RETURN(current_graph, internal::GetInducedUnweightedAverageGraph(
                                        graph, current_clusters));
    auto best_edge_weight = internal::GetBestEdgeWeight(*current_graph);
    double local_approx = best_edge_weight / merge_similarity;
    closeness = std::max(closeness, local_approx);

    // Apply the merge and compress cluster_ids.
    ABSL_CHECK_LT(node_a, node_b);
    auto cluster_a = cluster_ids[node_a];
    auto cluster_b = cluster_ids[node_b];

    auto min_cluster = std::min(cluster_a, cluster_b);
    auto max_cluster = std::max(cluster_a, cluster_b);

    for (NodeId i = 0; i < cluster_ids.size(); ++i) {
      if (cluster_ids[i] == max_cluster) {
        cluster_ids[i] = min_cluster;
      }
    }
  }

  return closeness;
}

absl::StatusOr<double> DendrogramGoodness(
    const Dendrogram& dendrogram, const SimpleUndirectedGraph& graph,
    double weight_threshold, std::vector<double> min_merge_similarities) {
  NodeId num_nodes = graph.NumNodes();

  if (min_merge_similarities.size() != num_nodes) {
    return absl::InvalidArgumentError(
        "min_merge_similarities.size != graph.NumNodes()");
  }

  if (dendrogram.NumClusteredNodes() != num_nodes) {
    return absl::InvalidArgumentError(
        "The number of leaf nodes in the dendrogram != graph.NumNodes()");
  }

  SimpleUndirectedGraph unweighted_graph;
  RETURN_IF_ERROR(CopyGraph(graph, &unweighted_graph));

  for (NodeId i = 0; i < num_nodes; ++i) {
    for (const auto [neighbor, weight] : graph.Neighbors(i)) {
      if (neighbor >= i) {
        RETURN_IF_ERROR(unweighted_graph.SetEdgeWeight(
            i, neighbor,
            weight * graph.NodeWeight(i) * graph.NodeWeight(neighbor)));
      }
    }
  }

  // No need to initialize the new entries, as we use this array as a map: each
  // entry is written to before being accessed.
  min_merge_similarities.resize(2 * num_nodes - 1);

  // Counts the number of children of each dendrogram node, which are leaves,
  // i.e., available to merge right away. Once the number of children is 2, we
  // know that a merge corresponding to the dendrogram node can be made.
  std::vector<std::vector<int>> leaf_children(2 * num_nodes - 1);
  for (int i = 0; i < num_nodes; ++i) {
    if (dendrogram.Nodes()[i].parent_id != DendrogramNode::kNoParentId) {
      leaf_children[dendrogram.Nodes()[i].parent_id].push_back(i);
    }
  }

  // The initial clustering is just the identity clustering, with the cluster
  // ids of internal dendrogram nodes set to -1.
  std::vector<NodeId> cluster_ids(num_nodes);
  std::iota(cluster_ids.begin(), cluster_ids.end(), 0);

  double max_goodness = 0;

  std::unique_ptr<SimpleUndirectedGraph> current_graph;
  ASSIGN_OR_RETURN(current_graph, internal::GetInducedUnweightedAverageGraph(
                                      unweighted_graph, cluster_ids));

  while (internal::GetBestEdgeWeight(*current_graph) >= weight_threshold) {
    std::vector<double> max_incident_weight(current_graph->NumNodes(), 0);
    for (NodeId i = 0; i < current_graph->NumNodes(); ++i) {
      max_incident_weight[i] = internal::BestNeighborWeight(i, *current_graph);
    }

    double best_goodness = std::numeric_limits<double>::infinity();
    NodeId best_node_a, best_node_b, best_parent = -1;

    // Iterate through all available merges in the dendrogram to find the one
    // that maximizes goodness.
    for (size_t parent = 0; parent < leaf_children.size(); ++parent) {
      const auto& children = leaf_children[parent];
      if (children.size() != 2) continue;
      auto node_a = children[0];
      auto node_b = children[1];

      double edge_weight_ab = *current_graph->EdgeWeight(node_a, node_b);
      if (edge_weight_ab >= weight_threshold) {
        std::optional<double> reported;
        if (std::abs(dendrogram.Nodes()[node_a].merge_similarity -
                     edge_weight_ab) > 1e-6)
          reported = dendrogram.Nodes()[node_a].merge_similarity;
        if (std::abs(dendrogram.Nodes()[node_b].merge_similarity -
                     edge_weight_ab) > 1e-6)
          reported = dendrogram.Nodes()[node_b].merge_similarity;
        if (reported.has_value()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("The dendrogram reports an invalid merge "
                              "similarity. Reported=%f, actual=%f",
                              *reported, edge_weight_ab));
        }
      }

      double goodness =
          std::max(max_incident_weight[node_a], max_incident_weight[node_b]) /
          std::min(edge_weight_ab, std::min(min_merge_similarities[node_a],
                                            min_merge_similarities[node_b]));
      if (goodness < best_goodness) {
        best_goodness = goodness;
        best_node_a = node_a;
        best_node_b = node_b;
        best_parent = parent;
      }
    }

    if (best_parent < 0) {
      return absl::OutOfRangeError(absl::StrCat(
          "The dendrogram does not merge all graph edges of weight >= ",
          weight_threshold, " There remains an edge of weight ",
          internal::GetBestEdgeWeight(*current_graph)));
    }

    max_goodness = std::max(max_goodness, best_goodness);
    leaf_children[best_parent].clear();

    if (dendrogram.Nodes()[best_parent].parent_id !=
        DendrogramNode::kNoParentId) {
      leaf_children[dendrogram.Nodes()[best_parent].parent_id].push_back(
          best_parent);
    }

    min_merge_similarities[best_parent] =
        std::min(*current_graph->EdgeWeight(best_node_a, best_node_b),
                 std::min(min_merge_similarities[best_node_a],
                          min_merge_similarities[best_node_b]));

    for (NodeId i = 0; i < cluster_ids.size(); ++i) {
      if (cluster_ids[i] == best_node_a || cluster_ids[i] == best_node_b) {
        cluster_ids[i] = best_parent;
      }
    }

    ASSIGN_OR_RETURN(current_graph, internal::GetInducedUnweightedAverageGraph(
                                        unweighted_graph, cluster_ids));
  }

  return max_goodness;
}

}  // namespace graph_mining::in_memory
