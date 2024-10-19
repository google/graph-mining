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

#include "in_memory/clustering/correlation/parallel_correlation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/bridge.h"
#include "gbbs/edge_map_data.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "gbbs/vertex_subset.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/parallel_correlation_util.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

namespace {

using ::graph_mining::in_memory::ClustererConfig;
using ::graph_mining::in_memory::CorrelationClustererConfig;

// This struct is necessary to perform an edge map with GBBS over a vertex
// set. Essentially, all neighbors are valid in this edge map, and this
// map does not do anything except allow for neighbors to be aggregated
// into the next frontier.
struct CorrelationClustererEdgeMap {
  inline bool cond(gbbs::uintE d) { return true; }
  inline bool update(const gbbs::uintE& s, const gbbs::uintE& d, float wgh) {
    return true;
  }
  inline bool updateAtomic(const gbbs::uintE& s, const gbbs::uintE& d,
                           float wgh) {
    return true;
  }
};

// Takes as input a CorrelationClustererConfig, and if the clustering
// moves method is set to the DEFAULT option, returns the LOUVAIN method.
// Otherwise, returns the clustering moves method set by the original config.
CorrelationClustererConfig::ClusteringMovesMethod GetClusteringMovesMethod(
    const CorrelationClustererConfig& clusterer_config) {
  auto clustering_moves_method = clusterer_config.clustering_moves_method();
  if (clustering_moves_method == CorrelationClustererConfig::DEFAULT)
    return CorrelationClustererConfig::LOUVAIN;
  return clustering_moves_method;
}

// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters. Note that if the clusterer_config specifies
// an asynchronous setting for performing vertex moves, then consistency
// guarantees are relaxed and vertices are immediately moved to their
// desired clusters, without locking.
std::unique_ptr<gbbs::vertexSubset> BestMovesForVertexSubset(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* vertices_to_move,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config) {
  std::vector<std::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                std::nullopt);
  auto moved_clusters = std::make_unique<bool[]>(current_graph->n);

  bool use_asynchronous =
      !clusterer_config.correlation_clusterer_config().use_synchronous();

  if (use_asynchronous) {
    // Mark no vertices as having moved yet
    parlay::parallel_for(0, current_graph->n,
                         [&](std::size_t i) { moved_clusters[i] = false; });

    // Find best moves per vertex in moved_subset and asynchronously move
    // vertices. Note that in the asynchronous setting, vertices are
    // immediately moved to their desired clusters after computing said cluster.
    // The new cluster id of the vertex and the updated cluster weights are
    // stored in separate atomic operations without locking, so there is no
    // guarantee of consistency (e.g., a vertex may appear to have moved to
    // a new cluster by its stored cluster id before the cluster weight of its
    // new cluster is properly updated).
    helper->MaybeUnfoldClusterIdSpace();
    gbbs::vertexMap(*vertices_to_move, [&](std::size_t i) {
      auto [target_cluster, objective_change] =
          helper->BestMove(*current_graph, i);
      if (objective_change > 0) {
        gbbs::CAS<bool>(&moved_clusters[helper->ClusterIds()[i]], false, true);
        helper->MoveNodeToClusterAsync(i, target_cluster);
        auto new_cluster_id = helper->ClusterIds()[i];
        if (new_cluster_id < current_graph->n) {
          // If new_cluster_id is out of bound, it is handled by
          // MaybeFoldClusterIdSpace below.
          gbbs::CAS<bool>(&moved_clusters[new_cluster_id], false, true);
        }
      }
    });
    helper->MaybeFoldClusterIdSpace(moved_clusters.get());
  } else {
    // Find best moves per vertex in moved_subset
    gbbs::vertexMap(*vertices_to_move, [&](std::size_t i) {
      std::tuple<ClusteringHelper::ClusterId, double> best_move =
          helper->BestMove(*current_graph, i);

      // If a singleton cluster wishes to move to another singleton cluster,
      // only move if the id of the moving cluster is lower than the id
      // of the cluster it wishes to move to
      auto move_cluster_id = std::get<0>(best_move);
      auto current_cluster_id = helper->ClusterIds()[i];
      if (move_cluster_id < current_graph->n &&
          helper->ClusterSizes()[move_cluster_id] == 1 &&
          helper->ClusterSizes()[current_cluster_id] == 1 &&
          current_cluster_id >= move_cluster_id) {
        best_move = std::make_tuple(current_cluster_id, 0);
      }
      if (std::get<1>(best_move) > 0) moves[i] = std::get<0>(best_move);
    });

    // Compute modified clusters
    moved_clusters = helper->MoveNodesToCluster(moves);
  }

  // Perform cluster moves
  if (GetClusteringMovesMethod(
          clusterer_config.correlation_clusterer_config()) ==
      CorrelationClustererConfig::CLUSTER_MOVES) {
    // Aggregate clusters
    auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
    std::vector<std::vector<gbbs::uintE>> curr_clustering =
        graph_mining::in_memory::OutputIndicesById<ClusteringHelper::ClusterId,
                                                   gbbs::uintE>(
            helper->ClusterIds(), get_clusters, helper->ClusterIds().size());

    auto additional_moved_clusters = std::make_unique<bool[]>(current_graph->n);

    if (use_asynchronous) {
      // Mark no clusters as having moved yet
      parlay::parallel_for(0, current_graph->n, [&](std::size_t i) {
        additional_moved_clusters[i] = false;
      });

      helper->MaybeUnfoldClusterIdSpace();

      // Compute best move per cluster and asynchronously move clusters
      parlay::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
        if (!curr_clustering[i].empty()) {
          auto [target_cluster, objective_change] =
              helper->BestMove(*current_graph, curr_clustering[i]);
          if (objective_change > 0) {
            auto current_cluster_id =
                helper->ClusterIds()[curr_clustering[i].front()];
            helper->MoveNodesToClusterAsync(curr_clustering[i], target_cluster);
            additional_moved_clusters[current_cluster_id] = true;
          }
        }
      });

      helper->MaybeFoldClusterIdSpace(additional_moved_clusters.get());
    } else {
      // Reset moves
      parlay::parallel_for(0, num_nodes,
                           [&](std::size_t i) { moves[i] = std::nullopt; });

      // Compute best move per cluster
      parlay::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
        if (!curr_clustering[i].empty()) {
          std::tuple<ClusteringHelper::ClusterId, double> best_move =
              helper->BestMove(*current_graph, curr_clustering[i]);
          // If a cluster wishes to move to another cluster,
          // only move if the id of the moving cluster is lower than the id
          // of the cluster it wishes to move to
          auto move_cluster_id = std::get<0>(best_move);
          auto current_cluster_id =
              helper->ClusterIds()[curr_clustering[i].front()];
          if (move_cluster_id < current_graph->n &&
              current_cluster_id >= move_cluster_id) {
            best_move = std::make_tuple(current_cluster_id, 0);
          }
          if (std::get<1>(best_move) > 0) {
            for (size_t j = 0; j < curr_clustering[i].size(); j++) {
              moves[curr_clustering[i][j]] = std::get<0>(best_move);
            }
          }
        }
      });

      // Compute modified clusters
      additional_moved_clusters = helper->MoveNodesToCluster(moves);
    }
    parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
      moved_clusters[i] |= additional_moved_clusters[i];
    });
  }

  // Mark vertices adjacent to clusters that have moved; these are
  // the vertices whose best moves must be recomputed.
  auto seq = parlay::sequence<bool>::from_function(
      num_nodes,
      [&](std::size_t i) { return moved_clusters[helper->ClusterIds()[i]]; });
  auto local_moved_subset = std::make_unique<gbbs::vertexSubset>(
      num_nodes, num_nodes, std::move(seq));

  auto edge_map = CorrelationClustererEdgeMap{};
  auto new_moved_subset =
      gbbs::edgeMap(*current_graph, *(local_moved_subset), edge_map);
  return std::make_unique<gbbs::vertexSubset>(std::move(new_moved_subset));
}

// Computes for vertices in the current_graph their desired clusters, based
// on the initial clustering given in helper. Moves vertices to their
// desired clusters, either synchronously or asynchronously as specified by
// clusterer_config, and repeats for num_inner_iterations iterations (or
// until a stable state is achieved, i.e. no vertices desire to change
// clusters). Stores the cluster that gives the maximum objective, as compared
// to max_objective, in local_cluster_ids, and returns the maximum objective
// achieved.
double IterateBestMoves(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    ClusteringHelper* helper, std::vector<gbbs::uintE>& local_cluster_ids,
    double max_objective, int num_inner_iterations,
    const ClustererConfig& clusterer_config) {
  const auto num_nodes = current_graph->n;
  bool local_moved = true;
  auto seq = gbbs::sequence<bool>(num_nodes, true);
  auto moved_subset = std::make_unique<gbbs::vertexSubset>(
      gbbs::vertexSubset(num_nodes, num_nodes, std::move(seq)));

  // Iterate over best moves
  for (int local_iter = 0; local_iter < num_inner_iterations; ++local_iter) {
    ABSL_LOG(INFO) << "Best moves iteration " << local_iter;
    auto new_moved_subset = BestMovesForVertexSubset(
        current_graph, num_nodes, moved_subset.get(), helper, clusterer_config);
    moved_subset.swap(new_moved_subset);
    local_moved = !moved_subset->isEmpty();
    ABSL_LOG(INFO) << "Local movements: " << local_moved;
    if (!local_moved) {
      // Break early to save an extra objective computation.
      break;
    }

    // Compute new objective given by the local moves in this iteration
    double curr_objective = helper->ComputeObjective(*current_graph);
    ABSL_LOG(INFO) << std::fixed << std::setprecision(5)
                   << "Current objective: " << curr_objective
                   << ", previous objective: " << max_objective;

    // Update maximum objective
    if (curr_objective > max_objective) {
      parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
        local_cluster_ids[i] = helper->ClusterIds()[i];
      });
      max_objective = curr_objective;
    }
  }
  return max_objective;
}

}  // namespace

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    ClusteringHelper* initial_helper) const {
  
  using symmetric_ptr_graph =
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;

  const auto& config = clusterer_config.correlation_clusterer_config();
  RETURN_IF_ERROR(ValidateCorrelationClustererConfigConfig(config));

  if (config.use_bipartite_objective() && !graph_.IsValidBipartiteGraph()) {
    return absl::InvalidArgumentError("Invalid bipartite graph.");
  }

  std::unique_ptr<symmetric_ptr_graph> compressed_graph;

  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (GetClusteringMovesMethod(config)) {
    case CorrelationClustererConfig::CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 10;
      break;
    case CorrelationClustererConfig::LOUVAIN:
      num_iterations = config.louvain_config().num_iterations() > 0
                           ? config.louvain_config().num_iterations()
                           : 10;
      num_inner_iterations =
          config.louvain_config().num_inner_iterations() > 0
              ? config.louvain_config().num_inner_iterations()
              : 10;
      break;
    default:
      ABSL_LOG(FATAL) << "Unknown clustering moves method: "
                      << config.clustering_moves_method();
  }

  double max_objective = initial_helper->ComputeObjective(*(graph_.Graph()));

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph_.Graph()->n);
  parlay::parallel_for(0, graph_.Graph()->n,
                       [&](std::size_t i) { cluster_ids[i] = i; });

  // `original_node_id_to_current_node_ids` maintains the mapping of original
  // node ids to the node ids in the graph for the current iteration.
  std::vector<gbbs::uintE> original_node_id_to_current_node_ids = cluster_ids;

  // `cluster_id_and_part_to_new_node_ids` maintains the mapping of {cluster id,
  // part} at the current layer to the node ids in the compressed graph for the
  // next layer.
  //
  // This is needed only for the bipartite case.
  std::vector<std::array<gbbs::uintE, 2>> cluster_id_and_part_to_new_node_ids(
      graph_.Graph()->n, {0, 0});

  std::unique_ptr<ClusteringHelper> current_helper;

  // If performing multi-level refinement, store intermediate clusterings
  // and graphs
  bool use_refinement = config.use_refinement();
  // The index of each of these sequences matches the index of the outer
  // iteration of the correlation clustering algorithm, which represents
  // repeatedly moving vertices / clusters to their desired clusters, and in the
  // case of the Louvain method, compressing the graph. These sequences
  // encapsulate the clustering achieved and the state of the graph at each
  // index.
  gbbs::sequence<std::vector<gbbs::uintE>> recursive_cluster_ids;
  gbbs::sequence<std::vector<double>> recursive_node_weights;
  gbbs::sequence<std::vector<NodePartId>> recursive_node_parts;
  gbbs::sequence<std::vector<std::array<gbbs::uintE, 2>>>
      recursive_cluster_id_and_part_to_new_node_ids;
  gbbs::sequence<std::unique_ptr<symmetric_ptr_graph>> recursive_graphs;

  if (use_refinement) {
    recursive_cluster_ids =
        gbbs::sequence<std::vector<gbbs::uintE>>::from_function(
            num_iterations,
            [](std::size_t i) { return std::vector<gbbs::uintE>(); });
    recursive_node_weights = gbbs::sequence<std::vector<double>>::from_function(
        num_iterations, [](std::size_t i) { return std::vector<double>(); });
    recursive_node_parts =
        gbbs::sequence<std::vector<NodePartId>>::from_function(
            num_iterations,
            [](std::size_t i) { return std::vector<NodePartId>(); });
    recursive_cluster_id_and_part_to_new_node_ids =
        gbbs::sequence<std::vector<std::array<gbbs::uintE, 2>>>::from_function(
            num_iterations, [](std::size_t i) {
              return std::vector<std::array<gbbs::uintE, 2>>();
            });
    recursive_graphs =
        gbbs::sequence<std::unique_ptr<symmetric_ptr_graph>>::from_function(
            num_iterations, [](std::size_t i) {
              return std::unique_ptr<symmetric_ptr_graph>(nullptr);
            });
  }

  int iter;
  for (iter = 0; iter < num_iterations; ++iter) {
    ABSL_LOG(INFO) << "Clustering iteration " << iter;
    symmetric_ptr_graph* current_graph =
        (iter == 0) ? graph_.Graph() : compressed_graph.get();
    auto* helper = (iter == 0) ? initial_helper : current_helper.get();

    // Iterate over best moves.
    // TODO: refactor local_cluster_ids to be a return value of
    // IterateBestMoves.
    auto new_objective =
        IterateBestMoves(current_graph, helper, local_cluster_ids,
                         max_objective, num_inner_iterations, clusterer_config);

    // If no moves can be made at all, exit.
    // Note that the objective is comparable across different levels, as the
    // implementation uses the following trick: when contracting a graph and we
    // contract a cluster C_v into a node v, we add a self-loop at v, whose
    // weight is the total weight of all undirected edges within C. This ensures
    // that once we contract some clustering C of G, and obtain a graph H, the
    // objective of clustering C in G is equal to the objective of
    // singleton-cluster clustering of H.
    if (new_objective <= max_objective) {
      // Number of iterations used must be decremented for multi-level
      // refinement, so that refinement does not occur on a level with no
      // further vertex moves
      --iter;
      break;
    }

    max_objective = new_objective;

    // Compress cluster ids in initial_helper based on helper
    cluster_ids = graph_mining::in_memory::FlattenClustering(
        config.use_bipartite_objective() ? original_node_id_to_current_node_ids
                                         : cluster_ids,
        local_cluster_ids);

    if (iter == num_iterations - 1) {
      if (use_refinement) {
        recursive_cluster_ids[iter] = local_cluster_ids;
        recursive_node_weights[iter] = helper->NodeWeights();
        recursive_node_parts[iter] = helper->NodeParts();
        recursive_cluster_id_and_part_to_new_node_ids[iter] = {};
        recursive_graphs[iter] = std::move(compressed_graph);
      }
      break;
    }

    // TODO: May want to compress out size 0 clusters when compressing
    // the graph

    BipartiteGraphCompressionMetadata bipartite_metadata;
    if (config.use_bipartite_objective()) {
      bipartite_metadata = PrepareBipartiteGraphCompression(
          local_cluster_ids, helper->NodeParts(),
          std::move(cluster_id_and_part_to_new_node_ids));
      cluster_id_and_part_to_new_node_ids =
          std::move(bipartite_metadata.cluster_id_and_part_to_new_node_ids);
      original_node_id_to_current_node_ids =
          graph_mining::in_memory::FlattenClustering(
              original_node_id_to_current_node_ids,
              bipartite_metadata.node_id_to_new_node_ids);
    }

    graph_mining::in_memory::GraphWithWeights new_compressed_graph;
    ASSIGN_OR_RETURN(
        new_compressed_graph,
        CompressGraph(*current_graph,
                      config.use_bipartite_objective()
                          ? bipartite_metadata.node_id_to_new_node_ids
                          : local_cluster_ids,
                      *helper));
    compressed_graph.swap(new_compressed_graph.graph);
    if (use_refinement) {
      recursive_cluster_ids[iter] = local_cluster_ids;
      recursive_node_weights[iter] = helper->NodeWeights();
      recursive_node_parts[iter] = helper->NodeParts();
      recursive_cluster_id_and_part_to_new_node_ids[iter] =
          cluster_id_and_part_to_new_node_ids;
      recursive_graphs[iter] = std::move(new_compressed_graph.graph);
    }

    InMemoryClusterer::Clustering new_clustering;
    if (config.use_bipartite_objective()) {
      // Convert flat bipartite clustering ids for the compressed graph to
      // InMemoryClusterer::Clustering for bipartite case.
      auto get_clusters = [&](NodeId i) -> NodeId { return i; };
      new_clustering =
          graph_mining::in_memory::OutputIndicesById<ClusterId, NodeId>(
              bipartite_metadata.new_node_id_to_cluster_ids, get_clusters,
              bipartite_metadata.new_node_id_to_cluster_ids.size());
    }

    current_helper = std::make_unique<ClusteringHelper>(
        compressed_graph->n, clusterer_config,
        std::move(new_compressed_graph.node_weights), new_clustering,
        bipartite_metadata.new_node_parts);

    // Prepare for the next iteration.
    local_cluster_ids.resize(compressed_graph->n);
  }

  // Perform multi-level refinement
  if (use_refinement && iter > 0) {
    cluster_ids = recursive_cluster_ids[iter];
    for (int i = iter - 1; i >= 0; --i) {
      symmetric_ptr_graph* current_graph =
          (i == 0) ? graph_.Graph() : recursive_graphs[i].get();

      // For bipartite graph, first create the linkage between the node ids from
      // the previous level to the current level.
      std::vector<gbbs::uintE> recursive_node_id_to_new_node_ids;
      if (config.use_bipartite_objective()) {
        recursive_node_id_to_new_node_ids = FlattenBipartiteClustering(
            recursive_cluster_ids[i], recursive_node_parts[i],
            recursive_cluster_id_and_part_to_new_node_ids[i]);
      }

      auto flattened_cluster_ids = graph_mining::in_memory::FlattenClustering(
          config.use_bipartite_objective() ? recursive_node_id_to_new_node_ids
                                           : recursive_cluster_ids[i],
          cluster_ids);

      initial_helper->ResetClustering(flattened_cluster_ids,
                                      recursive_node_weights[i],
                                      recursive_node_parts[i]);

      max_objective = IterateBestMoves(current_graph, initial_helper,
                                       flattened_cluster_ids, max_objective,
                                       num_inner_iterations, clusterer_config);

      cluster_ids = std::move(flattened_cluster_ids);

      // TODO: Update this comment with a description of why
      // calling the destructor on the i-th recursive graph is necessary.
      auto to_del = std::move(recursive_graphs[i]);
    }
    // TODO: Update this comment with a description of why
    // calling the destructor on the iter-th recursive graph is necessary.
    auto to_del = std::move(recursive_graphs[iter]);
  }

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };
  *initial_clustering =
      graph_mining::in_memory::OutputIndicesById<ClusterId, NodeId>(
          cluster_ids, get_clusters, cluster_ids.size());

  return absl::OkStatus();
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  
  // Initialize clustering helper
  ClusteringHelper helper{static_cast<NodeId>(graph_.Graph()->n),
                          clusterer_config, *initial_clustering,
                          graph_.GetNodeParts()};
  return RefineClusters(clusterer_config, initial_clustering, &helper);
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelCorrelationClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  parlay::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<int32_t>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace graph_mining::in_memory
