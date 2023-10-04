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

#include "in_memory/clustering/affinity/parallel_affinity.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/affinity/parallel_affinity_internal.h"
#include "in_memory/clustering/affinity/weight_threshold.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

void AddNewClusters(ParallelAffinityClusterer::Clustering new_clusters,
                    ParallelAffinityClusterer::Clustering* clustering) {
  clustering->reserve(clustering->size() + new_clusters.size());
  std::move(new_clusters.begin(), new_clusters.end(),
            std::back_inserter(*clustering));
}

absl::StatusOr<std::vector<ParallelAffinityClusterer::Clustering>>
ParallelAffinityClusterer::HierarchicalFlatCluster(
    const ClustererConfig& config) const {
  
  ABSL_CHECK(graph_.Graph() != nullptr);
  const AffinityClustererConfig& affinity_config =
      config.affinity_clusterer_config();

  std::size_t n = graph_.Graph()->n;
  std::vector<gbbs::uintE> cluster_ids(n);
  parlay::parallel_for(0, n, [&](std::size_t i) { cluster_ids[i] = i; });

  // `result` contains per-level clustering results. For each level, it contains
  // both finished clusters and active clusters.
  std::vector<ParallelAffinityClusterer::Clustering> result;

  // `clustering` contains finished clusters only.
  ParallelAffinityClusterer::Clustering clustering;

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;
  std::vector<double> node_weights;
  if (graph_.Graph()->vertex_weights != nullptr &&
      affinity_config.use_node_weight_for_cluster_size()) {
    node_weights = {graph_.Graph()->vertex_weights,
                    graph_.Graph()->vertex_weights + n};
  }
  internal::SizeConstraintConfig size_constraint_config{
      affinity_config.size_constraint(), node_weights};
  for (int i = 0; i < affinity_config.num_iterations(); ++i) {
    double weight_threshold = 0;
    ASSIGN_OR_RETURN(
        weight_threshold,
        graph_mining::in_memory::AffinityWeightThreshold(affinity_config, i));

    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (i == 0) ? graph_.Graph() : compressed_graph.get();

    std::vector<gbbs::uintE> compressed_cluster_ids;
    ASSIGN_OR_RETURN(
        compressed_cluster_ids,
        NearestNeighborLinkage(*current_graph, weight_threshold,
                               affinity_config.has_size_constraint()
                                   ? std::make_optional(size_constraint_config)
                                   : std::nullopt));

    cluster_ids = FlattenClustering(cluster_ids, compressed_cluster_ids);

    // TODO: Performance can be improved by not finding finished
    // clusters on the last round
    auto new_clusters =
        FindFinishedClusters(*(graph_.Graph()), affinity_config, cluster_ids,
                             compressed_cluster_ids);
    AddNewClusters(std::move(new_clusters), &clustering);

    // Copy current clustering with finished clusters
    result.emplace_back(clustering);

    auto is_active = [&](gbbs::uintE i) {
      return cluster_ids[i] != UINT_E_MAX;
    };
    auto current_clusters = ComputeClusters(cluster_ids, is_active);
    AddNewClusters(std::move(current_clusters), &(result.back()));

    // Exit if all clusters are finished
    auto exit_seq = parlay::delayed_seq<bool>(
        cluster_ids.size(),
        [&](std::size_t i) { return (cluster_ids[i] == UINT_E_MAX); });
    bool to_exit = parlay::reduce(
        exit_seq,
        parlay::make_monoid([](bool a, bool b) { return a && b; }, true));
    if (to_exit || i == affinity_config.num_iterations() - 1) break;

    // Compress graph
    GraphWithWeights new_compressed_graph;
    ASSIGN_OR_RETURN(new_compressed_graph,
                     CompressGraph(*current_graph, node_weights,
                                   compressed_cluster_ids, affinity_config));
    compressed_graph.swap(new_compressed_graph.graph);
    node_weights = new_compressed_graph.node_weights;
  }

  if (result.empty()) {
    ParallelAffinityClusterer::Clustering trivial_clustering(graph_.Graph()->n);
    parlay::parallel_for(0, trivial_clustering.size(), [&](NodeId i) {
      trivial_clustering[i] = std::vector<NodeId>{i};
    });
    result.emplace_back(trivial_clustering);
  }

  return result;
}

absl::StatusOr<ParallelAffinityClusterer::Clustering>
ParallelAffinityClusterer::Cluster(const ClustererConfig& config) const {
  
  std::vector<ParallelAffinityClusterer::Clustering> clustering_hierarchy;
  ASSIGN_OR_RETURN(clustering_hierarchy, HierarchicalFlatCluster(config));

  if (clustering_hierarchy.empty()) {
    ParallelAffinityClusterer::Clustering trivial_clustering(graph_.Graph()->n);
    parlay::parallel_for(0, trivial_clustering.size(), [&](NodeId i) {
      trivial_clustering[i] = std::vector<NodeId>{i};
    });
    return trivial_clustering;
  } else {
    return clustering_hierarchy.back();
  }
}

}  // namespace graph_mining::in_memory
