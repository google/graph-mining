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
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "gbbs/bridge.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/correlation_util.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "parlay/delayed_sequence.h"
#include "parlay/monoid.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ClusterId = ClusteringHelper::ClusterId;
using ::graph_mining::in_memory::CorrelationClustererConfig;

void ClusteringHelper::ResetClustering(
    const std::vector<ClusterId>& cluster_ids, std::vector<double> node_weights,
    std::vector<NodePartId> node_parts) {
  num_nodes_ = cluster_ids.size();

  size_t node_weights_size = node_weights.size();
  ABSL_CHECK(node_weights_size == num_nodes_ || node_weights_size == 0);

  cluster_ids_.resize(num_nodes_);
  node_weights_ = std::move(node_weights);

  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    partitioned_cluster_weights_.resize(num_nodes_);
    cluster_sizes_.resize(num_nodes_);
    parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
      partitioned_cluster_weights_[i] = {0, 0};
      cluster_sizes_[i] = 0;
    });
  } else {
    cluster_weights_.resize(num_nodes_);
    cluster_sizes_.resize(num_nodes_);
    parlay::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_weights_[i] = 0;
      cluster_sizes_[i] = 0;
    });
  }
  node_parts_ = std::move(node_parts);

  std::vector<std::vector<NodeId>> clustering =
      graph_mining::in_memory::OutputIndicesById<ClusterId, NodeId>(
          cluster_ids);
  SetClustering(clustering);
}

void ClusteringHelper::SetClustering(
    const InMemoryClusterer::Clustering& clustering) {
  cluster_sizes_.resize(num_nodes_);
  cluster_ids_.resize(num_nodes_);
  partitioned_cluster_weights_.resize(num_nodes_);
  cluster_weights_.resize(num_nodes_);
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
        for (NodeId j : clustering[i]) {
          cluster_ids_[j] = i;
          partitioned_cluster_weights_[i][node_parts_[j]] += NodeWeight(j);
        }
      });
    } else {
      parlay::parallel_for(0, clustering.size(), [&](std::size_t i) {
        cluster_sizes_[i] = clustering[i].size();
        for (NodeId j : clustering[i]) {
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
    const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph)
    const {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);

  // Compute cluster statistics contributions of each vertex
  parlay::parallel_for(0, graph.n, [&](std::size_t i) {
    ClusterId cluster_id_i = cluster_ids_[i];
    auto add_m = parlay::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       float weight) -> double {
      // Since the graph is undirected, each undirected edge is represented by
      // two directed edges, with the exception of self-loops, which are
      // represented as one edge. Hence, the weight of each intra-cluster edge
      // must be divided by 2, unless it's a self-loop.
      if (u == v)
        return weight;
      if (cluster_id_i == cluster_ids_[v])
        return (weight - config.edge_weight_offset()) / 2.0;
      return 0.0;
    };
    shifted_edge_weight[i] = graph.get_vertex(i).out_neighbors().reduce(
        intra_cluster_sum_map_f, add_m);
  });
  double objective =
      graph_mining::in_memory::ReduceAdd<double>(shifted_edge_weight);

  double resolution_seq_result = 0.0;
  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    // In the bipartite mode, we iterate over the node id space and interpret
    // each node id as a cluster id. The following implementation is equivalent
    // to the uni-partite implementation interpreting the node ids as node ids
    //
    // ====================
    // Equivalent implementation interpreting the node ids as node ids
    // ====================
    // resolution_seq_result = 0
    // For each node id i in [0, graph.n)
    //   resolution_seq_result += node_weights_[i] *
    //     partitioned_cluster_weights_[cluster_ids_[i]][abs(1-node_parts_[i])]
    // resolution_seq_result /= 2
    // ====================
    //
    // The per-node-id and per-cluster-id computation is equivalent because, for
    // a cluster with m nodes in Part 0 (with node weights a_i (i=0..m-1)) and n
    // nodes in Part 1 (with node weights b_j (j=0..n-1)), the per-node-id
    // approach computes the penalty as
    //
    // Sum_{i=0..m-1} Sum_{j=0..n-1} a_i * b_j
    //
    // whereas the per-cluster-id approach computes the penalty as
    //
    // (Sum_{i=0..m-1} a_i) * (Sum_{j=0..n-1} b_j)
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
    resolution_seq_result = parlay::reduce(resolution_seq) / 2.0;
  }
  objective -= config.resolution() * resolution_seq_result;
  return objective;
}

void ClusteringHelper::MoveNodeToClusterAsync(NodeId moving_node,
                                              ClusterId target_cluster_id) {
  std::vector<gbbs::uintE> moving_nodes = {
      static_cast<gbbs::uintE>(moving_node)};
  return MoveNodesToClusterAsync(moving_nodes, target_cluster_id);
}

void ClusteringHelper::MoveNodesToClusterAsync(
    const std::vector<gbbs::uintE>& moving_nodes, ClusterId target_cluster_id) {
  // Move moving_nodes from their current cluster
  bool use_bipartite_objective =
      clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective();
  ClusterId current_cluster_id =
      gbbs::atomic_load(&cluster_ids_[moving_nodes.front()]);
  gbbs::write_add(&cluster_sizes_[current_cluster_id],
                  -1 * moving_nodes.size());
  double total_node_weight = 0.0;
  std::array<double, 2> partitioned_total_node_weight = {0, 0};
  for (const gbbs::uintE moving_node : moving_nodes) {
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

  // Updates partitioned_cluster_weights_, cluster_weights_, cluster_ids_ by
  // moving moving_nodes to cluster_id.
  auto send_nodes_to_cluster = [&](ClusterId cluster_id) {
    if (use_bipartite_objective) {
      gbbs::write_add(&partitioned_cluster_weights_[cluster_id][0],
                      partitioned_total_node_weight[0]);
      gbbs::write_add(&partitioned_cluster_weights_[cluster_id][1],
                      partitioned_total_node_weight[1]);
    } else {
      gbbs::write_add(&cluster_weights_[cluster_id], total_node_weight);
    }
    for (const gbbs::uintE moving_node : moving_nodes) {
      gbbs::atomic_store(&cluster_ids_[moving_node], cluster_id);
    }
  };

  bool use_auxiliary_array_for_temp_cluster_id =
      clusterer_config_.correlation_clusterer_config()
          .use_auxiliary_array_for_temp_cluster_id();
  // If the new cluster move_cluster_id exists, move moving_nodes to the new
  // cluster
  ClusterId end_cluster_id_boundary =
      use_auxiliary_array_for_temp_cluster_id ? num_nodes_ * 2 : num_nodes_;
  if (target_cluster_id != end_cluster_id_boundary) {
    gbbs::write_add(&cluster_sizes_[target_cluster_id], moving_nodes.size());
    send_nodes_to_cluster(target_cluster_id);
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
        &cluster_sizes_[out_of_bound_id], /*oldval=*/0,
        /*newval=*/moving_nodes.size()));
    send_nodes_to_cluster(out_of_bound_id);
    return;
  }

  std::size_t i = moving_nodes.front();
  while (true) {
    if (gbbs::atomic_compare_and_swap<ClusterId>(
            &cluster_sizes_[i], /*oldval=*/0,
            /*newval=*/moving_nodes.size())) {
      send_nodes_to_cluster(i);
      return;
    }
    i = (i + 1) % num_nodes_;
  }
}

std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
    const std::vector<std::optional<ClusterId>>& moves) {
  ABSL_CHECK_EQ(moves.size(), num_nodes_);

  // modified_cluster[i] is true if cluster i has been modified.
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
    ClusterId prev_id = cluster_ids_[sorted_moving_nodes[start_id_index]];
    cluster_sizes_[prev_id] -= (end_id_index - start_id_index);
    modified_cluster[prev_id] = true;
    for (const gbbs::uintE node_id :
         absl::MakeSpan(sorted_moving_nodes.begin() + start_id_index,
                          sorted_moving_nodes.begin() + end_id_index)) {
      if (use_bipartite_objective) {
        partitioned_cluster_weights_[prev_id][node_parts_[node_id]] -=
            node_weights_[node_id];
      } else {
        cluster_weights_[prev_id] -= node_weights_[node_id];
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
  // those vertices that are forming new clusters.
  // Also, excepting those vertices that are forming new clusters, update
  // cluster_ids_.
  parlay::parallel_for(0, num_remark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = remark_moving_nodes[i];
    gbbs::uintE end_id_index = remark_moving_nodes[i + 1];
    ClusterId target_cluster_id = *moves[resorted_moving_nodes[start_id_index]];
    if (target_cluster_id == num_nodes_) return;
    cluster_sizes_[target_cluster_id] += (end_id_index - start_id_index);
    modified_cluster[target_cluster_id] = true;
    for (const gbbs::uintE node_id :
         absl::MakeSpan(resorted_moving_nodes.begin() + start_id_index,
                          resorted_moving_nodes.begin() + end_id_index)) {
      cluster_ids_[node_id] = target_cluster_id;
      if (use_bipartite_objective) {
        partitioned_cluster_weights_[target_cluster_id][node_parts_[node_id]] +=
            node_weights_[node_id];
      } else {
        cluster_weights_[target_cluster_id] += node_weights_[node_id];
      }
    }
  });

  // If there are vertices forming new clusters
  if (*moves[resorted_moving_nodes.back()] == num_nodes_) {
    // Filter out cluster ids of empty clusters, so that these ids can be
    // reused for vertices forming new clusters. This is an optimization
    // so that cluster ids do not grow arbitrarily large, when assigning
    // new cluster ids.
    auto get_zero_clusters = [&](std::size_t i) { return i; };
    auto seq_zero_clusters =
        parlay::delayed_seq<ClusterId>(num_nodes_, get_zero_clusters);
    auto zero_clusters = parlay::filter(
        seq_zero_clusters,
        [&](ClusterId id) -> bool { return cluster_sizes_[id] == 0; });

    // Indexing into these cluster ids gives the new cluster ids for new
    // clusters; update cluster_ids_ and cluster_sizes_ appropriately
    gbbs::uintE start_id_index =
        remark_moving_nodes[num_remark_moving_nodes - 1];
    gbbs::uintE end_id_index = remark_moving_nodes[num_remark_moving_nodes];
    parlay::parallel_for(start_id_index, end_id_index, [&](std::size_t i) {
      ClusterId cluster_id = zero_clusters[i - start_id_index];
      gbbs::uintE moving_node_id = resorted_moving_nodes[i];
      cluster_ids_[moving_node_id] = cluster_id;
      cluster_sizes_[cluster_id] = 1;
      modified_cluster[cluster_id] = true;
      if (use_bipartite_objective) {
        partitioned_cluster_weights_[cluster_id][node_parts_[moving_node_id]] =
            node_weights_[moving_node_id];
      } else {
        cluster_weights_[cluster_id] = node_weights_[moving_node_id];
      }
    });
  }

  return modified_cluster;
}

namespace {

// Returns a function that given a node ID, indicates whether it belongs to
// moving_nodes. All entries in moving_nodes are assumed to be smaller than
// num_nodes.
std::function<bool(gbbs::uintE)> NodeIsMovingFunction(
    const std::vector<gbbs::uintE>& moving_nodes, size_t num_nodes) {
  if (moving_nodes.size() == 1) {
    // Special handling for the case of a single moving node, which is a common
    // case in which we want to avoid using O(num_nodes) memory.
    return [single_moving_node_id = moving_nodes[0]](gbbs::uintE node_id) {
      return node_id == single_moving_node_id;
    };
  } else {
    std::vector<bool> node_is_moving(num_nodes, false);
    for (gbbs::uintE moving_node : moving_nodes) {
      node_is_moving[moving_node] = true;
    }
    return [node_is_moving = std::move(node_is_moving)](gbbs::uintE node_id) {
      return node_is_moving[node_id];
    };
  }
}

}  // namespace

ClusteringHelper::ClusterMove ClusteringHelper::BestMove(
    const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const std::vector<gbbs::uintE>& moving_nodes) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  const std::function<bool(gbbs::uintE)> node_is_moving_fn =
      NodeIsMovingFunction(moving_nodes, graph.n);

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

  double moving_nodes_weight = 0.0;
  std::array<double, 2> partitioned_moving_nodes_weight = {0, 0};
  for (const gbbs::uintE node : moving_nodes) {
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
      if (node_is_moving_fn(neighbor)) {
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
                                               /*parallel=*/false);
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
  auto [cluster_id, objective_change] =
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

  ClusteringHelper::ClusterId target_cluster_id = cluster_id.value_or(
      config.use_auxiliary_array_for_temp_cluster_id() ? graph.n * 2 : graph.n);
  return {.target_cluster_id = target_cluster_id,
          .objective_change = objective_change};
}

namespace {

// Represents the weight of a node in the compressed graph (where each node
// represents a cluster formed by nodes in the original graph).
struct CompressedGraphNodeWeight {
  ClusterId cluster_id = 0;
  double node_weight = 0.0;
};

}  // namespace

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
  std::vector<CompressedGraphNodeWeight> node_weights(original_graph.n);
  parlay::parallel_for(0, original_graph.n, [&](std::size_t i) {
    node_weights[i] = {.cluster_id = cluster_ids[i],
                       .node_weight = helper.NodeWeight(i)};
  });

  // Initialize new node weights
  std::vector<double> new_node_weights(num_compressed_vertices, 0.0);

  // Sort weights of neighbors by cluster id
  auto node_weights_sort =
      graph_mining::in_memory::ParallelSampleSort<CompressedGraphNodeWeight>(
          absl::MakeSpan(node_weights),
          [&](CompressedGraphNodeWeight a, CompressedGraphNodeWeight b) {
            return a.cluster_id < b.cluster_id;
          });

  // Obtain the boundary indices where cluster ids differ
  std::vector<gbbs::uintE> mark_node_weights =
      graph_mining::in_memory::GetBoundaryIndices<gbbs::uintE>(
          node_weights_sort.size(),
          [&node_weights_sort](std::size_t i, std::size_t j) {
            return node_weights_sort[i].cluster_id ==
                   node_weights_sort[j].cluster_id;
          });
  std::size_t num_mark_node_weights = mark_node_weights.size() - 1;

  // Reset helper to singleton clusters, with appropriate node weights
  parlay::parallel_for(0, num_mark_node_weights, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_node_weights[i];
    gbbs::uintE end_id_index = mark_node_weights[i + 1];
    auto node_weight =
        graph_mining::in_memory::Reduce<CompressedGraphNodeWeight>(
            absl::Span<const CompressedGraphNodeWeight>(
                node_weights_sort.begin() + start_id_index,
                end_id_index - start_id_index),
            [&](CompressedGraphNodeWeight a, CompressedGraphNodeWeight b) {
              return CompressedGraphNodeWeight{
                  .cluster_id = a.cluster_id,
                  .node_weight = a.node_weight + b.node_weight};
            },
            {.cluster_id = node_weights_sort[start_id_index].cluster_id,
             .node_weight = 0.0});
    new_node_weights[node_weight.cluster_id] = node_weight.node_weight;
  });

  return graph_mining::in_memory::GraphWithWeights(
      graph_mining::in_memory::MakeGbbsGraph<float>(
          offsets, num_compressed_vertices, std::move(edges), num_edges),
      new_node_weights);
}

void ClusteringHelper::MaybeUnfoldClusterIdSpace() {
  ABSL_CHECK_EQ(cluster_sizes_.size(), num_nodes_);
  if (!clusterer_config_.correlation_clusterer_config()
           .use_auxiliary_array_for_temp_cluster_id()) {
    return;
  }
  size_t new_size = num_nodes_ * 2;
  cluster_sizes_.resize(new_size);
  if (clusterer_config_.correlation_clusterer_config()
          .use_bipartite_objective()) {
    partitioned_cluster_weights_.resize(new_size);
  } else {
    cluster_weights_.resize(new_size);
  }
}

void ClusteringHelper::MaybeFoldClusterIdSpace(bool moved_clusters[]) {
  if (!clusterer_config_.correlation_clusterer_config()
           .use_auxiliary_array_for_temp_cluster_id()) {
    ABSL_CHECK_EQ(cluster_sizes_.size(), num_nodes_);
    return;
  }

  ABSL_CHECK_EQ(cluster_sizes_.size(), 2 * num_nodes_);

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
      gbbs::CAS<bool>(&moved_clusters[empty_cluster_id], /*oldv=*/false,
                      /*newv=*/true);
      auxiliary_cluster_ids[out_of_bound_cluster_id - num_nodes_] =
          empty_cluster_id;
    };

    parlay::parallel_for(0, moving_clusters.size(), [&](std::size_t i) {
      ClusterId out_of_bound_cluster_id = moving_clusters[i];
      ClusterId empty_cluster_id = empty_clusters[i];
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

absl::Status ValidateCorrelationClustererConfig(
    const CorrelationClustererConfig& config, const size_t num_nodes) {
  if (config.use_auxiliary_array_for_temp_cluster_id() &&
      config.use_synchronous()) {
    return absl::InvalidArgumentError(
        "use_auxiliary_array_for_temp_cluster_id and use_synchronous cannot "
        "both be set to true.");
  }
  // In some places, e.g. `FlattenBipartiteClustering`, the maximum value of
  // `ClusterId` has a special meaning and needs to be reserved. Beyond that
  // special value, the algorithm uses cluster IDs in the range [0, num_nodes]
  // or [0, 2 * num_nodes], depending on the value of
  // `use_auxiliary_array_for_temp_cluster_id`.
  static constexpr ClusterId kMaxValidClusterId =
      std::numeric_limits<ClusterId>::max() - 1;
  const ClusterId num_nodes_limit =
      config.use_auxiliary_array_for_temp_cluster_id() ? kMaxValidClusterId / 2
                                                       : kMaxValidClusterId;
  if (num_nodes > num_nodes_limit) {
    return absl::InvalidArgumentError(
        absl::StrCat("Total number of nodes (", num_nodes,
                     ") exceeds the limit: ", num_nodes_limit));
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
  parlay::parallel_for(0, num_original_nodes,
                       [&cluster_id_and_part_to_new_node_ids](std::size_t i) {
                         cluster_id_and_part_to_new_node_ids[i] = {UINT_E_MAX,
                                                                   UINT_E_MAX};
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
        /*oldv=*/0, /*newv=*/1);
  });

  // Capture new node id boundaries before computing the prefix sum.
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
  // compressed graph. Specifically, it maps each new compressed node id to the
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
  std::size_t cluster_size = cluster_ids.size();
  ABSL_CHECK_EQ(node_parts.size(), cluster_size);
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
