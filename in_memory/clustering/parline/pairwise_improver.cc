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

#include "in_memory/clustering/parline/pairwise_improver.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parline/fm_base.h"
#include "in_memory/clustering/parline/pairing_scheme.h"
#include "in_memory/clustering/parline/parline.pb.h"
#include "in_memory/clustering/types.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace graph_mining::in_memory {
namespace {

using ClusterPairingMethod = PairwiseImproverConfig::ClusterPairingMethod;

std::unique_ptr<PairingScheme> ConstructPairingScheme(
    int num_ids, const ClusterPairingMethod& pairing_method, int num_clusters) {
  switch (pairing_method.name()) {
    case ClusterPairingMethod::DEFAULT_ODD_EVEN:
      return std::make_unique<OddEvenPairingScheme>(num_ids, num_clusters);
    case ClusterPairingMethod::DISTANCE:
      return std::make_unique<DistancePairingScheme>(num_ids, num_clusters,
                                                     pairing_method.distance());
  }
  ABSL_LOG(FATAL) << "Unsupported cluster id pairing method: "
                  << pairing_method.name();
}

void UpdateClusters(const absl::flat_hash_set<NodeId>& cluster1_to_cluster2,
                    const absl::flat_hash_set<NodeId>& cluster2_to_cluster1,
                    absl::flat_hash_set<NodeId>& cluster1,
                    absl::flat_hash_set<NodeId>& cluster2) {
  for (const auto& node : cluster1_to_cluster2) {
    cluster1.erase(node);
    cluster2.insert(node);
  }
  for (const auto& node : cluster2_to_cluster1) {
    cluster2.erase(node);
    cluster1.insert(node);
  }
}

// TODO: Move it to a util.
double ComputeTotalNodeWeight(const LinePartitionerConfig& config,
                              const GbbsGraph& gbbs_graph) {
  auto graph = gbbs_graph.Graph();
  if (!config.use_node_weights() || graph->vertex_weights == nullptr) {
    return graph->num_vertices();
  }
  auto weight_seq = parlay::delayed_seq<double>(
      graph->n, [&](size_t i) { return graph->vertex_weights[i]; });
  return parlay::reduce(weight_seq, parlay::addm<double>());
}

}  // namespace

InMemoryClusterer::Clustering ImproveClustersPairwise(
    const GbbsGraph& graph,
    const InMemoryClusterer::Clustering& initial_clustering,
    const LinePartitionerConfig& line_config) {
  if (initial_clustering.empty() || !line_config.has_local_search_config() ||
      !line_config.local_search_config().has_pairwise_improver_config())
    return initial_clustering;
  auto pairwise_improver_config =
      line_config.local_search_config().pairwise_improver_config();
  if (pairwise_improver_config.num_improvement_iterations() <= 0)
    return initial_clustering;
  int num_clusters = initial_clustering.size();
  auto pairing_scheme = ConstructPairingScheme(
      num_clusters, pairwise_improver_config.cluster_pairing_method(),
      num_clusters);

  const double max_cluster_weight = (1 + line_config.imbalance()) *
                                    ComputeTotalNodeWeight(line_config, graph) /
                                    initial_clustering.size();
  // Create individual clusters that will be improved pairwise in parallel.
  std::vector<absl::flat_hash_set<NodeId>> clusters(num_clusters);
  parlay::parallel_for(0, num_clusters, [&](size_t i) {
    clusters[i].insert(initial_clustering[i].begin(),
                       initial_clustering[i].end());
  });

  int total_iterations = pairwise_improver_config.num_improvement_iterations() *
                         pairing_scheme->CycleSize();
  // Each iteration improves clusters pairwise in parallel.
  for (int i = 0; i < total_iterations; ++i) {
    std::vector<std::pair<int, int>> cluster_id_pairs = pairing_scheme->Next();
    parlay::parallel_for(0, cluster_id_pairs.size(), [&](size_t idx) {
      auto& cluster1 = clusters[cluster_id_pairs[idx].first];
      auto& cluster2 = clusters[cluster_id_pairs[idx].second];
      absl::flat_hash_set<NodeId> cluster1_to_cluster2;
      absl::flat_hash_set<NodeId> cluster2_to_cluster1;
      FMBase fm;
      fm.Improve(graph, cluster1, cluster2, max_cluster_weight,
                 cluster1_to_cluster2, cluster2_to_cluster1);
      UpdateClusters(cluster1_to_cluster2, cluster2_to_cluster1, cluster1,
                     cluster2);
    });
  }

  // Convert the final clusters to the in-memory format.
  InMemoryClusterer::Clustering improved_clustering;
  improved_clustering.reserve(num_clusters);
  for (const auto& cluster : clusters) {
    improved_clustering.push_back(
        std::vector<NodeId>(cluster.begin(), cluster.end()));
  }
  return improved_clustering;
}

}  // namespace graph_mining::in_memory
