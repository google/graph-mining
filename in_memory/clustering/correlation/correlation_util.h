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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_CORRELATION_UTIL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_CORRELATION_UTIL_H_

#include <array>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/correlation.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Computes the correlation clustering objective value defined mathematically in
// ../config.proto. Dies if the provided clustering has a node ID that is not
// valid for the graph.
double CorrelationClusteringObjective(
    const SimpleUndirectedGraph& graph,
    const graph_mining::in_memory::CorrelationClustererConfig& config,
    const InMemoryClusterer::Clustering& clustering);

// A helper class that keeps track of the sum of edge weights, accounting
// for missing edges, for best move computations.
class EdgeSum {
 public:
  // The edge weight w should have the edge weight offset subtracted before
  // calling this function.
  void Add(double w) { weight_ += w; }
  // Should be called at most once, after all edges have been Add()ed.
  void RemoveDoubleCounting() { weight_ /= 2; }
  // Retrieve the total weight of all edges seen, correcting for the implicit
  // negative weight of resolution multiplied by the product of the weights of
  // the two nodes incident to each edge.
  double NetWeight(double sum_prod_node_weights,
                   const graph_mining::in_memory::CorrelationClustererConfig&
                       config) const {
    return weight_ - config.resolution() * sum_prod_node_weights;
  }

 private:
  double weight_ = 0.0;
};

// Computes the best move given certain pre-computed sums of edge weights of the
// following classes of vertices in relation to a fixed set of moving_nodes that
// may change clusters:
//  * Class 0: Neither node is moving.
//  * Class 1: Exactly one node is moving.
//  * Class 2: Both nodes are moving.
// where "moving" means in moving_nodes.
//
// Change in objective if we move all moving nodes to cluster i:
//   class_2_currently_separate + class_1_together_after[i] -
//   class_1_currently_together
// where
//   class_2_currently_separate = Weight of edges in class 2 where the endpoints
//       are in different clusters currently
//   class_1_together_after[i] = Weight of edges in class 1 where the non-moving
//       node is in cluster i
//   class_1_currently_together = Weight of edges in class 1 where the endpoints
//       are in the same cluster currently
//
// Two complications:
//   * We need to avoid double-counting pairs in class 2
//   * We need to account for missing edges, which have weight
//     -resolution. To do so we subtract the number of edges we see in each
//     category from the max possible number of edges (i.e. the number of edges
//     we'd have if the graph was complete).
//
// For cluster IDs not present in cluster_moving_weights, we assume the
// corresponding weight is 0.
template <typename ClusterId>
std::pair<std::optional<ClusterId>, double> BestMoveFromStats(
    const graph_mining::in_memory::CorrelationClustererConfig& config,
    const std::function<double(ClusterId)>& get_current_cluster_weight,
    const double moving_nodes_weight,
    const absl::flat_hash_map<ClusterId, double>& cluster_moving_weights,
    const EdgeSum& class_2_currently_separate,
    const EdgeSum& class_1_currently_together,
    const absl::flat_hash_map<ClusterId, EdgeSum>& class_1_together_after) {
  double change_in_objective = 0.0;

  auto half_square = [](const double x) { return x * x / 2.0; };
  double max_edges = half_square(moving_nodes_weight);
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    max_edges -= half_square(moving_nodes_weight);
  }
  change_in_objective +=
      class_2_currently_separate.NetWeight(max_edges, config);

  max_edges = 0.0;
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    max_edges += moving_nodes_weight *
                 (get_current_cluster_weight(cluster) - moving_nodes_weight);
  }
  change_in_objective -=
      class_1_currently_together.NetWeight(max_edges, config);

  std::pair<std::optional<ClusterId>, double> best_move = {std::nullopt,
                                                           change_in_objective};
  for (const auto& [cluster, data] : class_1_together_after) {
    const auto cluster_moving_weight_it = cluster_moving_weights.find(cluster);
    const double cluster_moving_weight =
        cluster_moving_weight_it == cluster_moving_weights.end()
            ? 0.0
            : cluster_moving_weight_it->second;
    max_edges = moving_nodes_weight *
                (get_current_cluster_weight(cluster) - cluster_moving_weight);
    // Change in objective if we move the moving nodes to cluster i.
    double overall_change_in_objective =
        change_in_objective + data.NetWeight(max_edges, config);
    if (overall_change_in_objective > best_move.second ||
        (overall_change_in_objective == best_move.second &&
         cluster < best_move.first)) {
      best_move = {cluster, overall_change_in_objective};
    }
  }
  return best_move;
}

// Computes the best move given certain pre-computed sums of edge weights of the
// following classes of vertices in relation to a fixed set of moving_nodes that
// may change clusters:
//  * Class 0: Neither node is moving.
//  * Class 1: Exactly one node is moving.
//  * Class 2: Both nodes are moving.
// where "moving" means in moving_nodes.
//
// Change in objective if we move all moving nodes to cluster i:
//   class_2_currently_separate + class_1_together_after[i] -
//   class_1_currently_together
// where
//   class_2_currently_separate = Weight of edges in class 2 where the endpoints
//       are in different clusters currently
//   class_1_together_after[i] = Weight of edges in class 1 where the non-moving
//       node is in cluster i
//   class_1_currently_together = Weight of edges in class 1 where the endpoints
//       are in the same cluster currently
//
// Two complications:
//   * We need to avoid double-counting pairs in class 2
//   * We need to account for missing edges, which have weight
//     -resolution. To do so we subtract the number of edges we see in each
//     category from the max possible number of edges (i.e. the number of edges
//     we'd have if the graph was complete).
//
// In addition to the objective computation in `BestMoveFromStats`,
// `BestMoveFromStatsForBipartiteGraph` handles bipartite objective computation.
// Specifically, in the bipartite case, BestMoveFromStatsForBipartiteGraph
// computes same-partition-edge related objective compensation and applies it to
// the baseline objective computed in the same way as BestMoveFromStats.
//
// For cluster IDs not present in cluster_moving_weights, we assume the
// corresponding weight is 0.
//
// Dies if config.use_bipartite_objective is false, or if any key of
// partitioned_cluster_weights or class_1_together_after is not in the range [0,
// partitioned_cluster_weights.size()).
template <typename ClusterId>
std::pair<std::optional<ClusterId>, double> BestMoveFromStatsForBipartiteGraph(
    const graph_mining::in_memory::CorrelationClustererConfig& config,
    const std::function<double(ClusterId)>& get_current_cluster_weight,
    const std::array<double, 2>& moving_nodes_weight,
    const absl::flat_hash_map<ClusterId, std::array<double, 2>>&
        cluster_moving_weights,
    const EdgeSum& class_2_currently_separate,
    const EdgeSum& class_1_currently_together,
    const absl::flat_hash_map<ClusterId, EdgeSum>& class_1_together_after,
    const std::vector<std::array<double, 2>>& partitioned_cluster_weights) {
  double change_in_objective = 0.0;
  ABSL_CHECK(config.use_bipartite_objective());

  auto half_square = [](const double x) { return x * x / 2.0; };
  double total_moving_nodes_weight =
      moving_nodes_weight[0] + moving_nodes_weight[1];
  double max_edges = half_square(total_moving_nodes_weight);
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    max_edges -= half_square(moving_nodes_weight[0] + moving_nodes_weight[1]);
  }
  change_in_objective +=
      class_2_currently_separate.NetWeight(max_edges, config);

  max_edges = 0.0;
  double bipartite_compensation_loss = 0.0;
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    ABSL_CHECK_GE(cluster, 0);
    ABSL_CHECK_LT(cluster, partitioned_cluster_weights.size());
    double total_moving_nodes_weight =
        moving_nodes_weight[0] + moving_nodes_weight[1];
    max_edges +=
        total_moving_nodes_weight *
        (get_current_cluster_weight(cluster) - total_moving_nodes_weight);
    for (NodePartId i : {0, 1}) {
      bipartite_compensation_loss +=
          moving_nodes_weight[i] *
          (partitioned_cluster_weights[cluster][i] - moving_nodes_weight[i]) *
          config.resolution();
    }
  }
  change_in_objective -=
      class_1_currently_together.NetWeight(max_edges, config);
  change_in_objective -= bipartite_compensation_loss;

  std::pair<std::optional<ClusterId>, double> best_move = {std::nullopt,
                                                           change_in_objective};
  for (const auto& [cluster, data] : class_1_together_after) {
    ABSL_CHECK_GE(cluster, 0);
    ABSL_CHECK_LT(cluster, partitioned_cluster_weights.size());
    double total_moving_nodes_weight =
        moving_nodes_weight[0] + moving_nodes_weight[1];
    const auto cluster_moving_weights_it = cluster_moving_weights.find(cluster);
    const std::array<double, 2> cluster_moving_weights_array =
        cluster_moving_weights_it == cluster_moving_weights.end()
            ? std::array<double, 2>{0.0, 0.0}
            : cluster_moving_weights_it->second;
    max_edges =
        total_moving_nodes_weight *
        (get_current_cluster_weight(cluster) - cluster_moving_weights_array[0] -
         cluster_moving_weights_array[1]);
    // Change in objective if we move the moving nodes to cluster i.
    double overall_change_in_objective =
        change_in_objective + data.NetWeight(max_edges, config);

    // Add objective compensation gain for nodes within the same partition.
    for (NodePartId i : {0, 1}) {
      overall_change_in_objective += moving_nodes_weight[i] *
                                     (partitioned_cluster_weights[cluster][i] -
                                      cluster_moving_weights_array[i]) *
                                     config.resolution();
    }

    if (overall_change_in_objective > best_move.second ||
        (overall_change_in_objective == best_move.second &&
         cluster < best_move.first)) {
      best_move = {cluster, overall_change_in_objective};
    }
  }
  return best_move;
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_CORRELATION_UTIL_H_
