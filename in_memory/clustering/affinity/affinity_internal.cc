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

#include "in_memory/clustering/affinity/affinity_internal.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/compress_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/tiebreaking.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ::research_graph::in_memory::AffinityClustererConfig;

std::vector<NodeId> FlattenClustering(
    const std::vector<NodeId>& cluster_ids,
    const std::vector<NodeId>& compressed_cluster_ids) {
  ABSL_CHECK_LE(cluster_ids.size(), std::numeric_limits<NodeId>::max());
  NodeId n = cluster_ids.size();
  std::vector<NodeId> result(n);
  for (NodeId i = 0; i < n; ++i) {
    ABSL_CHECK_GE(cluster_ids[i], -1);

    result[i] =
        cluster_ids[i] == -1 ? -1 : compressed_cluster_ids[cluster_ids[i]];
  }
  return result;
}

absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> CompressGraph(
    const SimpleUndirectedGraph& graph, const std::vector<NodeId>& cluster_ids,
    AffinityClustererConfig clusterer_config) {
  ABSL_CHECK_EQ(cluster_ids.size(), graph.NumNodes());
  const auto edge_aggregation = clusterer_config.edge_aggregation_function();

  std::vector<double> node_weights(graph.NumNodes());
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    ABSL_CHECK_GE(cluster_ids[i], -1);
    ABSL_CHECK_LT(cluster_ids[i], graph.NumNodes());
    if (cluster_ids[i] != -1) ++node_weights[cluster_ids[i]];
  }

  std::unique_ptr<SimpleUndirectedGraph> result;

  // Perform graph compession based on the specified edge aggregation function.
  switch (edge_aggregation) {
    // PERCENTILE and EXPLICIT_AVERAGE edge aggregations require the tracking of
    // all edge weights for each pair of connected clusters.
    case AffinityClustererConfig::PERCENTILE:
      ABSL_CHECK(clusterer_config.has_percentile_linkage_value());
      ABSL_CHECK_GE(clusterer_config.percentile_linkage_value(), 0);
      ABSL_CHECK_LE(clusterer_config.percentile_linkage_value(), 1);
      ABSL_FALLTHROUGH_INTENDED;
    case AffinityClustererConfig::EXPLICIT_AVERAGE: {
      result = std::make_unique<SimpleUndirectedGraph>();
      result->SetNumNodes(graph.NumNodes());

      // NOTE: To ensure determinism we dump all of the edge-weights to a single
      // array, sort it, and then merge parallel edges in the sorted order.
      using edge = std::tuple<NodeId, NodeId, double>;
      std::vector<edge> edge_weights;

      for (NodeId i = 0; i < graph.NumNodes(); ++i) {
        if (cluster_ids[i] == -1) continue;

        for (const auto& [neighbor_id, edge_weight] : graph.Neighbors(i)) {
          // Process each undirected edge once, do not add self loops and ignore
          // deleted vertices.
          if (neighbor_id >= i || cluster_ids[i] == cluster_ids[neighbor_id] ||
              cluster_ids[neighbor_id] == -1) {
            continue;
          }

          // For PERCENTILE edge aggregation, first find all neighbor edge
          // weights before processing.
          auto min_id = std::min(cluster_ids[i], cluster_ids[neighbor_id]);
          auto max_id = std::max(cluster_ids[i], cluster_ids[neighbor_id]);
          edge_weights.emplace_back(edge(min_id, max_id, edge_weight));
        }
      }
      std::sort(edge_weights.begin(), edge_weights.end());

      // Loop over the edge weights.
      for (size_t i = 0; i < edge_weights.size(); ++i) {
        auto [node_u, node_v, uv_weight] = edge_weights[i];
        // Check that this is the first index of this edge. This loop iteration
        // will be responsible for computing the correct edge weight and setting
        // it in the result.
        if (i == 0 || std::tie(std::get<0>(edge_weights[i - 1]),
                               std::get<1>(edge_weights[i - 1])) !=
                          std::tie(node_u, node_v)) {
          if (edge_aggregation == AffinityClustererConfig::EXPLICIT_AVERAGE) {
            double weights_sum = 0.0;
            size_t num_weights = 0;
            for (size_t j = i; j < edge_weights.size(); ++j) {
              auto [node_j_u, node_j_v, j_uv_weight] = edge_weights[j];
              if (std::tie(node_u, node_v) != std::tie(node_j_u, node_j_v)) {
                break;
              }
              weights_sum += j_uv_weight;
              ++num_weights;
            }
            RETURN_IF_ERROR(result->SetEdgeWeight(node_u, node_v,
                                                  weights_sum / num_weights));
          } else {
            ABSL_CHECK_GT(
                clusterer_config.min_edge_count_for_percentile_linkage(), 0);
            std::vector<double> weights;
            for (size_t j = i; j < edge_weights.size(); ++j) {
              auto [node_j_u, node_j_v, j_uv_weight] = edge_weights[j];
              if (std::tie(node_u, node_v) != std::tie(node_j_u, node_j_v)) {
                break;
              }
              weights.push_back(j_uv_weight);
            }
            if (weights.size() <
                clusterer_config.min_edge_count_for_percentile_linkage()) {
              RETURN_IF_ERROR(result->SetEdgeWeight(
                  node_u, node_v,
                  *std::max_element(weights.begin(), weights.end())));
            } else {
              int percentile_index =
                  std::floor(clusterer_config.percentile_linkage_value() *
                             static_cast<float>(weights.size() - 1));
              std::nth_element(weights.begin(),
                               weights.begin() + percentile_index,
                               weights.end());
              RETURN_IF_ERROR(result->SetEdgeWeight(node_u, node_v,
                                                    weights[percentile_index]));
            }
          }
        }
      }
      break;
    }

    // Specify a max reduction function for MAX edge aggregation.
    case AffinityClustererConfig::MAX: {
      ASSIGN_OR_RETURN(
          result,
          CompressGraph(graph, cluster_ids,
                        [](double compressed_weight, double edge_weight) {
                          return std::max(compressed_weight, edge_weight);
                        }));
      break;
    }

    // All other supported aggregation functions sum edge weights during graph
    // compression.
    default: {
      ASSIGN_OR_RETURN(
          result,
          CompressGraph(graph, cluster_ids,
                        [](double compressed_weight, double edge_weight) {
                          return compressed_weight + edge_weight;
                        }));
    }
  }

  // Perform rescaling after graph compression for the relevant edge aggregation
  // functions.
  if (edge_aggregation == AffinityClustererConfig::DEFAULT_AVERAGE ||
      edge_aggregation == AffinityClustererConfig::CUT_SPARSITY) {
    for (NodeId i = 0; i < graph.NumNodes(); ++i) {
      for (const auto& edge : result->Neighbors(i)) {
        // Process each undirected edge once and do not add self loops.
        if (edge.first >= i || cluster_ids[i] == cluster_ids[edge.first])
          continue;
        double scaling_factor;
        if (edge_aggregation == AffinityClustererConfig::DEFAULT_AVERAGE) {
          scaling_factor = node_weights[i] * node_weights[edge.first];
        } else {
          ABSL_CHECK_EQ(edge_aggregation,
                        AffinityClustererConfig::CUT_SPARSITY);
          scaling_factor = std::min(node_weights[i], node_weights[edge.first]);
        }
        RETURN_IF_ERROR(
            result->SetEdgeWeight(i, edge.first, edge.second / scaling_factor));
      }
    }
  }
  return result;
}

std::vector<NodeId> NearestNeighborLinkage(
    const SimpleUndirectedGraph& graph, double weight_threshold,
    std::function<std::string(InMemoryClusterer::NodeId)> get_node_id) {
  SequentialUnionFind<NodeId> cc_finder(graph.NumNodes());

  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    if (graph.Neighbors(i).empty()) continue;

    MaxWeightTiebreaker tiebreaker;
    NodeId best_neighbor_id = -1;
    for (const auto& [neighbor, edge_weight] : graph.Neighbors(i)) {
      if (tiebreaker.IsMaxWeightSoFar(edge_weight, get_node_id(neighbor))) {
        best_neighbor_id = neighbor;
      }
    }

    ABSL_CHECK_GE(best_neighbor_id, 0);
    if (tiebreaker.MaxWeight() >= weight_threshold) {
      cc_finder.Unite(i, best_neighbor_id);
    }
  }
  std::vector<NodeId> abc(graph.NumNodes());
  std::vector<NodeId> final_component_id(graph.NumNodes(), -1);
  std::vector<std::string> smallest_string_id(graph.NumNodes());
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    NodeId component_id = cc_finder.Find(i);
    abc[i] = component_id;
    std::string string_id = get_node_id(i);

    if (final_component_id[component_id] == -1 ||
        smallest_string_id[component_id] > string_id) {
      final_component_id[component_id] = i;
      smallest_string_id[component_id] = string_id;
    }
  }
  return FlattenClustering(abc, final_component_id);
}

InMemoryClusterer::Clustering ComputeClusters(
    const std::vector<NodeId>& cluster_ids) {
  ABSL_CHECK_LE(cluster_ids.size(), std::numeric_limits<NodeId>::max());
  NodeId n = cluster_ids.size();
  std::vector<std::vector<NodeId>> clustering(n);

  for (NodeId i = 0; i < n; ++i) {
    ABSL_CHECK_GE(cluster_ids[i], -1);
    ABSL_CHECK_LT(cluster_ids[i], n);
    if (cluster_ids[i] != -1) clustering[cluster_ids[i]].push_back(i);
  }
  clustering.erase(std::remove_if(clustering.begin(), clustering.end(),
                                  [](const std::vector<NodeId>& cluster) {
                                    return cluster.empty();
                                  }),
                   clustering.end());
  return clustering;
}

ClusterQualityIndicators ComputeClusterQualityIndicators(
    const std::vector<NodeId>& cluster, const SimpleUndirectedGraph& graph,
    double graph_volume) {
  ClusterQualityIndicators result;
  result.density = result.conductance = 0.0;
  double volume = 0.0, inter_cluster_weight = 0.0;

  absl::flat_hash_set<NodeId> cluster_elements(cluster.begin(), cluster.end());

  for (NodeId cluster_member : cluster) {
    ABSL_CHECK_GE(cluster_member, 0);
    ABSL_CHECK_LT(cluster_member, graph.NumNodes());
    for (const auto& [neighbor_id, weight] : graph.Neighbors(cluster_member)) {
      ABSL_CHECK_GE(neighbor_id, 0);
      ABSL_CHECK_LT(neighbor_id, graph.NumNodes());

      volume += weight;

      if (cluster_elements.count(neighbor_id)) {
        if (cluster_member <= neighbor_id) {  // count each undirected edge once
          result.density += weight;
        }
      } else {
        inter_cluster_weight += weight;
      }
    }
  }
  if (cluster.size() >= 2) {
    result.density /=
        static_cast<double>(cluster.size()) * (cluster.size() - 1) / 2.0;
  } else {
    result.density = 0.0;
  }
  double denominator = std::min(volume, graph_volume - volume);
  if (denominator < 1e-6)
    result.conductance = 1.0;
  else
    result.conductance = inter_cluster_weight / denominator;
  return result;
}

bool IsActiveCluster(const std::vector<NodeId>& cluster,
                     const SimpleUndirectedGraph& graph,
                     const AffinityClustererConfig& config,
                     double graph_volume) {
  ABSL_CHECK_GT(cluster.size(), 0);
  if (config.active_cluster_conditions().empty()) return true;
  auto quality = ComputeClusterQualityIndicators(cluster, graph, graph_volume);
  for (const auto& condition : config.active_cluster_conditions()) {
    bool satisfied = true;
    if (condition.has_min_density() &&
        quality.density < condition.min_density())
      satisfied = false;
    if (condition.has_min_conductance() &&
        quality.conductance < condition.min_conductance())
      satisfied = false;
    if (satisfied) return true;
  }
  return false;
}

}  // namespace graph_mining::in_memory
