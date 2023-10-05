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

#include "in_memory/clustering/correlation/parallel_correlation_util.h"

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "parlay/parallel.h"
#include "gbbs/bridge.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/correlation_util.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/parallel_sequence_ops.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ClusterId = ClusteringHelper::ClusterId;
using ::graph_mining::in_memory::CorrelationClustererConfig;

void ClusteringHelper::ResetClustering(
    const std::vector<ClusterId>& cluster_ids,
    const std::vector<double>& node_weights,
    const std::vector<NodePartId>& node_parts) {
  num_nodes_ = cluster_ids.size();

  auto node_weights_size = node_weights.size();
  ABSL_CHECK(node_weights_size == num_nodes_ || node_weights_size == 0);

  node_weights_.resize(node_weights.size());
  parlay::parallel_for(0, node_weights_size, [&](std::size_t i) {
    node_weights_[i] = node_weights[i];
  });

  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
      partitioned_cluster_weights_[i] = {0, 0};
      cluster_sizes_[i] = 0;
    });
  } else {
    parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_weights_[i] = 0;
      cluster_sizes_[i] = 0;
    });
  }
  node_parts_ = node_parts;

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };
  auto clustering =
      graph_mining::in_memory::OutputIndicesById<ClusterId, NodeId>(
          cluster_ids, get_clusters, num_nodes_);
  SetClustering(clustering);
}

void ClusteringHelper::SetClustering(
    const InMemoryClusterer::Clustering& clustering) {
  if (clustering.empty()) {
    // Keep the following if condition outside the parallel_for to avoid
    // checking this condition for each node.
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
        cluster_sizes_[i] = 1;
        cluster_ids_[i] = i;
        partitioned_cluster_weights_[i][node_parts_[i]] = NodeWeight(i);
      });
    } else {
      parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
        cluster_sizes_[i] = 1;
        cluster_ids_[i] = i;
        cluster_weights_[i] = NodeWeight(i);
      });
    }
  } else {
    // Keep the following if condition outside the parallel_for to avoid
    // checking this condition for each node.
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      parlay::parallel_for(0, clustering.size(), [&](std::size_t i) {
        cluster_sizes_[i] = clustering[i].size();
        for (auto j : clustering[i]) {
          cluster_ids_[j] = i;
          partitioned_cluster_weights_[i][node_parts_[j]] += NodeWeight(j);
        }
      });
    } else {
      parlay::parallel_for(0, clustering.size(), [&](std::size_t i) {
        cluster_sizes_[i] = clustering[i].size();
        for (auto j : clustering[i]) {
          cluster_ids_[j] = i;
          cluster_weights_[i] += NodeWeight(j);
        }
      });
    }
  }
}

double ClusteringHelper::NodeWeight(NodeId id) const {
  return id < node_weights_.size() ? node_weights_[id] : 1.0;
}

double ClusteringHelper::ComputeObjective(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);

  // Compute cluster statistics contributions of each vertex
  parlay::parallel_for(0, graph.n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids_[i];
    auto add_m = parlay::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       float weight) -> double {
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i == cluster_ids_[v])
        return (weight - config.edge_weight_offset()) / 2;
      return 0;
    };
    shifted_edge_weight[i] = graph.get_vertex(i).out_neighbors().reduce(
        intra_cluster_sum_map_f, add_m);
  });
  double objective = graph_mining::in_memory::ReduceAdd(
      absl::Span<const double>(shifted_edge_weight));

  double resolution_seq_result = 0;
  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    auto resolution_seq =
        parlay::delayed_seq<double>(graph.n, [&](std::size_t i) {
          return partitioned_cluster_weights_[i][0] *
                 partitioned_cluster_weights_[i][1];
        });
    resolution_seq_result = parlay::reduce(resolution_seq);
  } else {
    auto resolution_seq =
        parlay::delayed_seq<double>(graph.n, [&](std::size_t i) {
          auto cluster_weight = cluster_weights_[cluster_ids_[i]];
          return node_weights_[i] * (cluster_weight - node_weights_[i]);
        });
    resolution_seq_result = parlay::reduce(resolution_seq) / 2;
  }
  objective -= config.resolution() * resolution_seq_result;
  return objective;
}

void ClusteringHelper::MoveNodeToClusterAsync(NodeId moving_node,
                                              ClusterId move_cluster_id) {
  std::vector<gbbs::uintE> moving_nodes = {
      static_cast<gbbs::uintE>(moving_node)};
  return MoveNodesToClusterAsync(moving_nodes, move_cluster_id);
}

void ClusteringHelper::MoveNodesToClusterAsync(
    const std::vector<gbbs::uintE>& moving_nodes, ClusterId move_cluster_id) {
  // Move moving_nodes from their current cluster
  bool use_bipartite_objective =
      clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective();
  auto current_cluster_id =
      gbbs::atomic_load(&cluster_ids_[moving_nodes.front()]);
  gbbs::write_add(&cluster_sizes_[current_cluster_id],
                  -1 * moving_nodes.size());
  double total_node_weight = 0;
  std::array<double, 2> partitioned_total_node_weight = {0, 0};
  for (const auto& moving_node : moving_nodes) {
    total_node_weight += node_weights_[moving_node];
    if (use_bipartite_objective) {
      partitioned_total_node_weight[node_parts_[moving_node]] +=
          node_weights_[moving_node];
    }
  }

  if (use_bipartite_objective) {
    gbbs::write_add(&partitioned_cluster_weights_[current_cluster_id][0],
                    -1 * partitioned_total_node_weight[0]);
    gbbs::write_add(&partitioned_cluster_weights_[current_cluster_id][1],
                    -1 * partitioned_total_node_weight[1]);
  } else {
    gbbs::write_add(&cluster_weights_[current_cluster_id],
                    -1 * total_node_weight);
  }

  auto use_auxiliary_array_for_temp_cluster_id =
      clusterer_config_.correlation_clusterer_config()
          .use_auxiliary_array_for_temp_cluster_id();

  // If the new cluster move_cluster_id exists, move moving_nodes to the new
  // cluster
  ClusterId end_cluster_id_boundary =
      use_auxiliary_array_for_temp_cluster_id ? num_nodes_ * 2 : num_nodes_;
  if (move_cluster_id != end_cluster_id_boundary) {
    gbbs::write_add(&cluster_sizes_[move_cluster_id], moving_nodes.size());
    if (use_bipartite_objective) {
      gbbs::write_add(&partitioned_cluster_weights_[move_cluster_id][0],
                      partitioned_total_node_weight[0]);
      gbbs::write_add(&partitioned_cluster_weights_[move_cluster_id][1],
                      partitioned_total_node_weight[1]);
    } else {
      gbbs::write_add(&cluster_weights_[move_cluster_id], total_node_weight);
    }
    for (const auto& moving_node : moving_nodes) {
      gbbs::atomic_store(&cluster_ids_[moving_node], move_cluster_id);
    }
    return;
  }

  // If the new cluster move_cluster_id does not exist, find an empty cluster
  // and move moving_nodes to the empty cluster
  if (use_auxiliary_array_for_temp_cluster_id) {
    // When an auxiliary array exists to store temporary new cluster ids, for
    // each center node, there must exist exactly one available direct-mapping
    // slot (i.e., for an index i in [0, num_nodes_), map it to i+num_nodes_).
    // Use CHECK to verify that this is the case and then use the slot.
    std::size_t out_of_bound_id = moving_nodes.front() + num_nodes_;
    ABSL_CHECK(gbbs::atomic_compare_and_swap<ClusterId>(
        &cluster_sizes_[out_of_bound_id], 0, moving_nodes.size()));
    if (use_bipartite_objective) {
      gbbs::write_add(&partitioned_cluster_weights_[out_of_bound_id][0],
                      partitioned_total_node_weight[0]);
      gbbs::write_add(&partitioned_cluster_weights_[out_of_bound_id][1],
                      partitioned_total_node_weight[1]);
    } else {
      gbbs::write_add(&cluster_weights_[out_of_bound_id], total_node_weight);
    }
    for (const auto i : moving_nodes) {
      gbbs::atomic_store(&cluster_ids_[i],
                         static_cast<ClusterId>(out_of_bound_id));
    }
    return;
  }

  std::size_t i = moving_nodes.front();
  while (true) {
    if (gbbs::atomic_compare_and_swap<ClusterId>(&cluster_sizes_[i], 0,
                                                 moving_nodes.size())) {
      if (use_bipartite_objective) {
        gbbs::write_add(&partitioned_cluster_weights_[i][0],
                        partitioned_total_node_weight[0]);
        gbbs::write_add(&partitioned_cluster_weights_[i][1],
                        partitioned_total_node_weight[1]);
      } else {
        gbbs::write_add(&cluster_weights_[i], total_node_weight);
      }
      for (const auto& moving_node : moving_nodes) {
        gbbs::atomic_store(&cluster_ids_[moving_node],
                           static_cast<ClusterId>(i));
      }
      return;
    }
    i++;
    i = i % num_nodes_;
  }
}

std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
    const std::vector<std::optional<ClusterId>>& moves) {
  ABSL_CHECK_EQ(moves.size(), num_nodes_);

  auto modified_cluster = std::make_unique<bool[]>(num_nodes_);
  parlay::parallel_for(0, num_nodes_,
                       [&](std::size_t i) { modified_cluster[i] = false; });

  // We must update cluster_sizes_ and assign new cluster ids to vertices
  // that want to form a new cluster
  // Obtain all nodes that are moving clusters
  auto get_moving_nodes = [&](size_t i) { return i; };
  auto moving_nodes = parlay::filter(
      parlay::delayed_seq<gbbs::uintE>(num_nodes_, get_moving_nodes),
      [&](gbbs::uintE node) -> bool {
        return moves[node].has_value() && moves[node] != cluster_ids_[node];
      });

  if (moving_nodes.empty()) return modified_cluster;

  // Sort moving nodes by original cluster id
  auto sorted_moving_nodes = parlay::stable_sort(
      parlay::make_slice(moving_nodes), [&](gbbs::uintE a, gbbs::uintE b) {
        return cluster_ids_[a] < cluster_ids_[b];
      });

  // The number of nodes moving out of clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> mark_moving_nodes =
      graph_mining::in_memory::GetBoundaryIndices<gbbs::uintE>(
          sorted_moving_nodes.size(), [&](std::size_t i, std::size_t j) {
            return cluster_ids_[sorted_moving_nodes[i]] ==
                   cluster_ids_[sorted_moving_nodes[j]];
          });
  std::size_t num_mark_moving_nodes = mark_moving_nodes.size() - 1;

  bool use_bipartite_objective =
      clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective();

  // Subtract these boundary sizes from cluster_sizes_ in parallel
  parlay::parallel_for(0, num_mark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_moving_nodes[i];
    gbbs::uintE end_id_index = mark_moving_nodes[i + 1];
    auto prev_id = cluster_ids_[sorted_moving_nodes[start_id_index]];
    cluster_sizes_[prev_id] -= (end_id_index - start_id_index);
    modified_cluster[prev_id] = true;
    for (std::size_t j = start_id_index; j < end_id_index; j++) {
      if (use_bipartite_objective) {
        partitioned_cluster_weights_[prev_id]
                                    [node_parts_[sorted_moving_nodes[j]]] -=
            node_weights_[sorted_moving_nodes[j]];
      } else {
        cluster_weights_[prev_id] -= node_weights_[sorted_moving_nodes[j]];
      }
    }
  });

  // Re-sort moving nodes by new cluster id
  auto resorted_moving_nodes = parlay::stable_sort(
      parlay::make_slice(moving_nodes),
      [&](gbbs::uintE a, gbbs::uintE b) { return moves[a] < moves[b]; });

  // The number of nodes moving into clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> remark_moving_nodes =
      graph_mining::in_memory::GetBoundaryIndices<gbbs::uintE>(
          resorted_moving_nodes.size(),
          [&resorted_moving_nodes, &moves](std::size_t i, std::size_t j) {
            return moves[resorted_moving_nodes[i]] ==
                   moves[resorted_moving_nodes[j]];
          });
  std::size_t num_remark_moving_nodes = remark_moving_nodes.size() - 1;

  // Add these boundary sizes to cluster_sizes_ in parallel, excepting
  // those vertices that are forming new clusters
  // Also, excepting those vertices that are forming new clusters, update
  // cluster_ids_
  parlay::parallel_for(0, num_remark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = remark_moving_nodes[i];
    gbbs::uintE end_id_index = remark_moving_nodes[i + 1];
    auto move_id = moves[resorted_moving_nodes[start_id_index]].value();
    if (move_id != num_nodes_) {
      cluster_sizes_[move_id] += (end_id_index - start_id_index);
      modified_cluster[move_id] = true;
      for (std::size_t j = start_id_index; j < end_id_index; j++) {
        cluster_ids_[resorted_moving_nodes[j]] = move_id;
        if (use_bipartite_objective) {
          partitioned_cluster_weights_[move_id]
                                      [node_parts_[resorted_moving_nodes[j]]] +=
              node_weights_[resorted_moving_nodes[j]];
        } else {
          cluster_weights_[move_id] += node_weights_[resorted_moving_nodes[j]];
        }
      }
    }
  });

  // If there are vertices forming new clusters
  if (moves[resorted_moving_nodes[moving_nodes.size() - 1]].value() ==
      num_nodes_) {
    // Filter out cluster ids of empty clusters, so that these ids can be
    // reused for vertices forming new clusters. This is an optimization
    // so that cluster ids do not grow arbitrarily large, when assigning
    // new cluster ids.
    auto get_zero_clusters = [&](std::size_t i) { return i; };
    auto seq_zero_clusters =
        parlay::delayed_seq<gbbs::uintE>(num_nodes_, get_zero_clusters);
    auto zero_clusters = parlay::filter(
        seq_zero_clusters,
        [&](gbbs::uintE id) -> bool { return cluster_sizes_[id] == 0; });

    // Indexing into these cluster ids gives the new cluster ids for new
    // clusters; update cluster_ids_ and cluster_sizes_ appropriately
    gbbs::uintE start_id_index =
        remark_moving_nodes[num_remark_moving_nodes - 1];
    gbbs::uintE end_id_index = remark_moving_nodes[num_remark_moving_nodes];
    parlay::parallel_for(start_id_index, end_id_index, [&](std::size_t i) {
      auto cluster_id = zero_clusters[i - start_id_index];
      cluster_ids_[resorted_moving_nodes[i]] = cluster_id;
      cluster_sizes_[cluster_id] = 1;
      modified_cluster[cluster_id] = true;
      if (use_bipartite_objective) {
        partitioned_cluster_weights_[cluster_id]
                                    [node_parts_[resorted_moving_nodes[i]]] =
                                        node_weights_[resorted_moving_nodes[i]];
      } else {
        cluster_weights_[cluster_id] = node_weights_[resorted_moving_nodes[i]];
      }
    });
  }

  return modified_cluster;
}

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    NodeId moving_node) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  absl::flat_hash_map<ClusterId, std::array<double, 2>>
      partitioned_cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  const ClusterId node_cluster =
      gbbs::atomic_load(&(cluster_ids_[moving_node]));
  double moving_nodes_weight = gbbs::atomic_load(&(node_weights_[moving_node]));
  std::array<double, 2> partitioned_moving_nodes_weight = {0, 0};
  if (config.use_bipartite_objective()) {
    partitioned_moving_nodes_weight[node_parts_[moving_node]] =
        moving_nodes_weight;
  }
  cluster_moving_weights[node_cluster] += moving_nodes_weight;
  if (config.use_bipartite_objective()) {
    partitioned_cluster_moving_weights[node_cluster] =
        partitioned_moving_nodes_weight;
  }
  auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                       double weight) {
    weight -= offset;
    const ClusterId neighbor_cluster =
        gbbs::atomic_load(&(cluster_ids_[neighbor]));
    if (moving_node == neighbor) {
      // Class 2 edge.
      if (node_cluster != neighbor_cluster) {
        class_2_currently_separate.Add(weight);
      }
    } else {
      // Class 1 edge.
      if (node_cluster == neighbor_cluster) {
        class_1_currently_together.Add(weight);
      }
      class_1_together_after[neighbor_cluster].Add(weight);
    }
  };
  graph.get_vertex(moving_node)
      .out_neighbors()
      .map(map_moving_node_neighbors, false);
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return config.use_bipartite_objective()
               ? gbbs::atomic_load(
                     &(partitioned_cluster_weights_[cluster][0])) +
                     gbbs::atomic_load(
                         &(partitioned_cluster_weights_[cluster][1]))
               : gbbs::atomic_load(&(cluster_weights_[cluster]));
  };
  auto best_move =
      config.use_bipartite_objective()
          ? BestMoveFromStatsForBipartiteGraph(
                config, get_cluster_weight, partitioned_moving_nodes_weight,
                partitioned_cluster_moving_weights, class_2_currently_separate,
                class_1_currently_together, class_1_together_after,
                partitioned_cluster_weights_)
          : BestMoveFromStats(
                config, get_cluster_weight, moving_nodes_weight,
                cluster_moving_weights, class_2_currently_separate,
                class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value()
          ? best_move.first.value()
          : (config.use_auxiliary_array_for_temp_cluster_id() ? graph.n * 2
                                                              : graph.n);
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const std::vector<gbbs::uintE>& moving_nodes) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  std::vector<bool> flat_moving_nodes(graph.n, false);
  for (size_t i = 0; i < moving_nodes.size(); i++) {
    flat_moving_nodes[moving_nodes[i]] = true;
  }

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  absl::flat_hash_map<ClusterId, std::array<double, 2>>
      partitioned_cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  std::array<double, 2> partitioned_moving_nodes_weight = {0, 0};
  for (const auto& node : moving_nodes) {
    const ClusterId node_cluster = gbbs::atomic_load(&(cluster_ids_[node]));
    double moving_node_weight = gbbs::atomic_load(&(node_weights_[node]));
    cluster_moving_weights[node_cluster] += moving_node_weight;
    moving_nodes_weight += moving_node_weight;
    if (config.use_bipartite_objective()) {
      partitioned_cluster_moving_weights[node_cluster][node_parts_[node]] +=
          moving_node_weight;
      partitioned_moving_nodes_weight[node_parts_[node]] += moving_node_weight;
    }
    auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                         float weight) {
      weight -= offset;
      const ClusterId neighbor_cluster =
          gbbs::atomic_load(&(cluster_ids_[neighbor]));
      if (flat_moving_nodes[neighbor]) {
        // Class 2 edge.
        if (node_cluster != neighbor_cluster) {
          class_2_currently_separate.Add(weight);
        }
      } else {
        // Class 1 edge.
        if (node_cluster == neighbor_cluster) {
          class_1_currently_together.Add(weight);
        }
        class_1_together_after[neighbor_cluster].Add(weight);
      }
    };
    graph.get_vertex(node).out_neighbors().map(map_moving_node_neighbors,
                                               false);
  }
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return config.use_bipartite_objective()
               ? gbbs::atomic_load(
                     &(partitioned_cluster_weights_[cluster][0])) +
                     gbbs::atomic_load(
                         &(partitioned_cluster_weights_[cluster][1]))
               : gbbs::atomic_load(&(cluster_weights_[cluster]));
  };

  auto best_move =
      config.use_bipartite_objective()
          ? BestMoveFromStatsForBipartiteGraph(
                config, get_cluster_weight, partitioned_moving_nodes_weight,
                partitioned_cluster_moving_weights, class_2_currently_separate,
                class_1_currently_together, class_1_together_after,
                partitioned_cluster_weights_)
          : BestMoveFromStats(
                config, get_cluster_weight, moving_nodes_weight,
                cluster_moving_weights, class_2_currently_separate,
                class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value()
          ? best_move.first.value()
          : (config.use_auxiliary_array_for_temp_cluster_id() ? graph.n * 2
                                                              : graph.n);
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

absl::StatusOr<graph_mining::in_memory::GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    const ClusteringHelper& helper) {
  // Obtain the number of vertices in the new graph
  auto get_cluster_ids = [&](size_t i) { return cluster_ids[i]; };
  auto seq_cluster_ids =
      parlay::delayed_seq<gbbs::uintE>(cluster_ids.size(), get_cluster_ids);
  gbbs::uintE num_compressed_vertices =
      1 + parlay::reduce(seq_cluster_ids, parlay::maxm<gbbs::uintE>());

  // Compute new inter cluster edges using sorting, allowing self-loops
  auto edge_aggregation_func = [](double w1, double w2) { return w1 + w2; };
  auto is_valid_func = [](ClusteringHelper::ClusterId a,
                          ClusteringHelper::ClusterId b) { return true; };
  auto scale_func = [](std::tuple<gbbs::uintE, gbbs::uintE, double> v) {
    return std::get<2>(v);
  };

  graph_mining::in_memory::OffsetsEdges offsets_edges =
      graph_mining::in_memory::ComputeInterClusterEdgesSort(
          original_graph, cluster_ids, num_compressed_vertices,
          edge_aggregation_func, is_valid_func, scale_func);
  const std::vector<std::size_t>& offsets = offsets_edges.offsets;
  std::size_t num_edges = offsets_edges.num_edges;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges =
      std::move(offsets_edges.edges);

  // Obtain cluster ids and node weights of all vertices
  std::vector<std::tuple<ClusterId, double>> node_weights(original_graph.n);
  parlay::parallel_for(0, original_graph.n, [&](std::size_t i) {
    node_weights[i] = std::make_tuple(cluster_ids[i], helper.NodeWeight(i));
  });

  // Initialize new node weights
  std::vector<double> new_node_weights(num_compressed_vertices, 0);

  // Sort weights of neighbors by cluster id
  auto node_weights_sort = graph_mining::in_memory::ParallelSampleSort<
      std::tuple<ClusterId, double>>(
      absl::Span<std::tuple<ClusterId, double>>(node_weights.data(),
                                                node_weights.size()),
      [&](std::tuple<ClusterId, double> a, std::tuple<ClusterId, double> b) {
        return std::get<0>(a) < std::get<0>(b);
      });

  // Obtain the boundary indices where cluster ids differ
  std::vector<gbbs::uintE> mark_node_weights =
      graph_mining::in_memory::GetBoundaryIndices<gbbs::uintE>(
          node_weights_sort.size(),
          [&node_weights_sort](std::size_t i, std::size_t j) {
            return std::get<0>(node_weights_sort[i]) ==
                   std::get<0>(node_weights_sort[j]);
          });
  std::size_t num_mark_node_weights = mark_node_weights.size() - 1;

  // Reset helper to singleton clusters, with appropriate node weights
  parlay::parallel_for(0, num_mark_node_weights, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_node_weights[i];
    gbbs::uintE end_id_index = mark_node_weights[i + 1];
    auto node_weight =
        graph_mining::in_memory::Reduce<std::tuple<ClusterId, double>>(
            absl::Span<const std::tuple<ClusterId, double>>(
                node_weights_sort.begin() + start_id_index,
                end_id_index - start_id_index),
            [&](std::tuple<ClusterId, double> a,
                std::tuple<ClusterId, double> b) {
              return std::make_tuple(std::get<0>(a),
                                     std::get<1>(a) + std::get<1>(b));
            },
            std::make_tuple(std::get<0>(node_weights[start_id_index]),
                            double{0}));
    new_node_weights[std::get<0>(node_weight)] = std::get<1>(node_weight);
  });

  return graph_mining::in_memory::GraphWithWeights(
      graph_mining::in_memory::MakeGbbsGraph<float>(
          offsets, num_compressed_vertices, std::move(edges), num_edges),
      new_node_weights);
}

void ClusteringHelper::MaybeUnfoldClusterIdSpace() {
  if (clusterer_config_.correlation_clusterer_config()
          .use_auxiliary_array_for_temp_cluster_id()) {
    auto new_size = num_nodes_ * 2;
    cluster_sizes_.resize(new_size);
    if (clusterer_config_.correlation_clusterer_config()
            .use_bipartite_objective()) {
      partitioned_cluster_weights_.resize(new_size);
    } else {
      cluster_weights_.resize(new_size);
    }
  }
}

void ClusteringHelper::MaybeFoldClusterIdSpace(bool* moved_clusters) {
  if (!clusterer_config_.correlation_clusterer_config()
           .use_auxiliary_array_for_temp_cluster_id()) {
    return;
  }

  // Step 1: identify all out-of-bound cluster ids.
  auto get_moving_cluster_id = [&](size_t i) { return i + num_nodes_; };
  auto moving_clusters = parlay::filter(
      parlay::delayed_seq<ClusterId>(num_nodes_, get_moving_cluster_id),
      [&](ClusterId node) -> bool { return cluster_sizes_[node] > 0; });

  if (!moving_clusters.empty()) {
    // Step 2: identify all in-bound empty clusters.
    //
    // It is probably more efficient to avoid retrieving all available empty
    // clusters, in particular when the size of the out-of-bound cluster ids is
    // small. We can tune this in the future.
    auto get_node_id = [&](size_t i) { return i; };
    auto empty_clusters = parlay::filter(
        parlay::delayed_seq<ClusterId>(num_nodes_, get_node_id),
        [&](ClusterId node) -> bool { return cluster_sizes_[node] == 0; });

    ABSL_CHECK_GE(empty_clusters.size(), moving_clusters.size());

    // Step 3: fold cluster id mapping for out-of-bound clusters.

    // auxiliary_cluster_ids holds the mapping from out-of-bound id - num_nodes_
    // to an in-bound available empty cluster.
    //
    // For example, suppose
    // * num_nodes_ = 10
    // * an out-of-bound cluster id 14 is mapped to an available empty cluster
    //   id 2
    //
    // Then we have the following entry in auxiliary_cluster_ids
    // * auxiliary_cluster_ids[4] = 2
    //
    // That is, the original out-of-bound cluster id 14 is mapped to 2 and is
    // stored at offset 4 (= 14 - 10) in auxiliary_cluster_ids.
    parlay::sequence<ClusterId> auxiliary_cluster_ids(num_nodes_);

    // Given an id pair representing the mapping between an out-of-bound cluster
    // id and its in-bound (empty cluster) id, rename_cluster is responsible for
    //
    // - adjusting cluster_sizes_[inbound_id]
    // - adjusting cluster_weights_[inbound_id]
    // - marking moved_clusters[inboud_id] for next iteration
    // - adjusting the outbound-inboud cluster id map auxiliary_cluster_ids
    //
    // Notably, rename_cluster does *not* adjust cluster_ids_. That requires a
    // separate step below and happens after all outbound-inboud mapping is
    // established.
    auto rename_cluster = [&](ClusterId out_of_bound_cluster_id,
                              ClusterId empty_cluster_id) {
      cluster_sizes_[empty_cluster_id] =
          cluster_sizes_[out_of_bound_cluster_id];
      if (clusterer_config_.correlation_clusterer_config()
              .use_bipartite_objective()) {
        partitioned_cluster_weights_[empty_cluster_id] =
            partitioned_cluster_weights_[out_of_bound_cluster_id];
      } else {
        cluster_weights_[empty_cluster_id] =
            cluster_weights_[out_of_bound_cluster_id];
      }
      gbbs::CAS<bool>(&moved_clusters[empty_cluster_id], false, true);
      auxiliary_cluster_ids[out_of_bound_cluster_id - num_nodes_] =
          empty_cluster_id;
    };

    parlay::parallel_for(0, moving_clusters.size(), [&](std::size_t i) {
      auto out_of_bound_cluster_id = moving_clusters[i];
      auto empty_cluster_id = empty_clusters[i];
      rename_cluster(out_of_bound_cluster_id, empty_cluster_id);
    });

    // Step 4: fix all out-of-bound cluster ids with the cluster id mapping.
    parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
      if (cluster_ids_[i] >= num_nodes_) {
        cluster_ids_[i] = auxiliary_cluster_ids[cluster_ids_[i] - num_nodes_];
      }
    });
  }

  // Step 5: fold cluster id space.
  cluster_sizes_.resize(num_nodes_);
  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    partitioned_cluster_weights_.resize(num_nodes_);
  } else {
    cluster_weights_.resize(num_nodes_);
  }
}

absl::Status ValidateCorrelationClustererConfigConfig(
    const CorrelationClustererConfig& config) {
  if (config.use_auxiliary_array_for_temp_cluster_id() &&
      config.use_synchronous()) {
    return absl::InvalidArgumentError(
        "use_auxiliary_array_for_temp_cluster_id and use_synchronous cannot "
        "both be set to true.");
  }
  return absl::OkStatus();
}

BipartiteGraphCompressionMetadata PrepareBipartiteGraphCompression(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<NodePartId>& parts,
    std::vector<std::array<gbbs::uintE, 2>>&&
        cluster_id_and_part_to_new_node_ids) {
  BipartiteGraphCompressionMetadata result;
  std::size_t num_original_nodes = cluster_ids.size();

  // `cluster_id_and_part_to_new_node_ids` is resized to match the number of
  // nodes in the current graph and reset to the default value.
  cluster_id_and_part_to_new_node_ids.resize(num_original_nodes);
  parlay::parallel_for(
      0, num_original_nodes,
      [&cluster_id_and_part_to_new_node_ids](std::size_t i) {
        cluster_id_and_part_to_new_node_ids[i] = {UINT_E_MAX, UINT_E_MAX};
      });

  // Step 1: Obtain new node ids for nodes in the original graph. Store
  // result in `node_id_to_new_node_ids`.

  parlay::sequence<gbbs::uintE> cluster_id_and_part_to_new_node_id_prefix_sum(
      num_original_nodes * 2, 0);
  parlay::parallel_for(0, num_original_nodes, [&](std::size_t i) {
    // The contention is benign, because we can only set the value from 0 to 1.
    gbbs::CAS<gbbs::uintE>(
        &cluster_id_and_part_to_new_node_id_prefix_sum[cluster_ids[i] * 2 +
                                                       parts[i]],
        0, 1);
  });

  // Catpure new node id boundaries before computing the prefix sum.
  auto cluster_id_and_part_to_new_node_id_boundaries =
      parlay::pack_index(cluster_id_and_part_to_new_node_id_prefix_sum);

  std::size_t num_new_nodes =
      parlay::scan_inplace(cluster_id_and_part_to_new_node_id_prefix_sum);
  ABSL_CHECK_EQ(cluster_id_and_part_to_new_node_id_boundaries.size(),
                num_new_nodes);

  std::vector<gbbs::uintE> node_id_to_new_node_ids(num_original_nodes);
  parlay::parallel_for(0, num_original_nodes, [&](std::size_t i) {
    gbbs::uintE prefix_idx = cluster_ids[i] * 2 + parts[i];
    node_id_to_new_node_ids[i] =
        cluster_id_and_part_to_new_node_id_prefix_sum[prefix_idx];
  });

  // Steps 2 and 3
  //
  // Step 2: Obtain partition information for nodes in the new compressed graph.
  //
  // Step 3: Obtain the cluster id information for the new compressed graph.
  // Note that in the bipartite case, the condition that each node is in its own
  // cluster no longer holds. It needs to be explicitly maintained.

  // `new_node_parts` maintains the partition information for the compressed
  // graph.
  std::vector<NodePartId> new_node_parts(num_new_nodes);

  // `new_node_id_to_cluster_ids` maintains the cluster id information for the
  // compressed graph. Speficially, it maps each new compressed node id to the
  // original cluster id.
  std::vector<gbbs::uintE> new_node_id_to_cluster_ids(num_new_nodes);

  parlay::parallel_for(0, num_new_nodes, [&](std::size_t i) {
    auto boundary = cluster_id_and_part_to_new_node_id_boundaries[i];
    new_node_parts[i] = boundary % 2;
    new_node_id_to_cluster_ids[i] = boundary / 2;
    cluster_id_and_part_to_new_node_ids[new_node_id_to_cluster_ids[i]]
                                       [new_node_parts[i]] = i;
  });

  result.node_id_to_new_node_ids = std::move(node_id_to_new_node_ids);
  result.new_node_parts = std::move(new_node_parts);
  result.new_node_id_to_cluster_ids = std::move(new_node_id_to_cluster_ids);
  result.cluster_id_and_part_to_new_node_ids =
      std::move(cluster_id_and_part_to_new_node_ids);
  return result;
}

std::vector<gbbs::uintE> FlattenBipartiteClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<NodePartId>& node_parts,
    const std::vector<std::array<gbbs::uintE, 2>>&
        cluster_id_and_part_to_new_node_ids) {
  auto cluster_size = cluster_ids.size();
  ABSL_CHECK_EQ(cluster_ids.size(), cluster_size);
  std::vector<gbbs::uintE> new_cluster_ids(cluster_size);
  parlay::parallel_for(0, cluster_size, [&](std::size_t i) {
    new_cluster_ids[i] =
        (cluster_ids[i] == UINT_E_MAX)
            ? UINT_E_MAX
            : cluster_id_and_part_to_new_node_ids[cluster_ids[i]]
                                                 [node_parts[i]];
  });
  return new_cluster_ids;
}

bool IsValidBipartiteNodeParts(const std::vector<NodePartId>& node_parts) {
  std::atomic<bool> valid = true;
  parlay::parallel_for(0, node_parts.size(), [&](std::size_t i) {
    if (node_parts[i] != 0 && node_parts[i] != 1) {
      valid = false;
    }
  });
  return valid;
}

}  // namespace graph_mining::in_memory
