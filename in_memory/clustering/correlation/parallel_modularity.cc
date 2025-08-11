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

#include "in_memory/clustering/correlation/parallel_modularity.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gbbs/graph.h"
#include "gbbs/helpers/progress_reporting.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/correlation/parallel_correlation.h"
#include "in_memory/clustering/correlation/parallel_correlation_util.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {

namespace {

using graph_mining::in_memory::ClustererConfig;

struct EdgeWeightStats {
  // The total weight of edges incident to each node.
  std::vector<double> weighted_node_degrees;
  // Sum of all edge weights.
  double total_edge_weight;
};

// Computes the stats of the edge weights in the graph. Returns an error if any
// edge has a negative weight.
absl::StatusOr<EdgeWeightStats> ComputeEdgeWeightStats(
    const gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph) {
  double total_edge_weight = 0.0;
  std::vector<double> weighted_node_degrees(graph.n, 0.0);
  absl::Status status = absl::OkStatus();
  for (std::size_t i = 0; i < graph.n; ++i) {
    auto map_weight = [&](gbbs::uintE vertex, gbbs::uintE neighbor,
                          double edge_weight) {
      if (edge_weight < 0.0) {
        status = absl::InvalidArgumentError(
            absl::StrFormat("An edge with negative weight %f was found between "
                            "nodes %d and %d.",
                            edge_weight, vertex, neighbor));
      }
      weighted_node_degrees[i] += edge_weight;
    };
    graph.get_vertex(i).out_neighbors().map(map_weight, /*parallel=*/false);
    if (!status.ok()) {
      return status;
    }
    total_edge_weight += weighted_node_degrees[i];
  }
  return EdgeWeightStats{std::move(weighted_node_degrees), total_edge_weight};
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelModularityClusterer::ClusterWithProgressReporting(
    const ClustererConfig& config,
    std::optional<gbbs::ReportProgressCallback> report_progress) const {
  if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "'Cluster' cannot be called before 'FinishImport' is called for the "
        "graph");
  }
  if (graph_.Graph()->n == 0) {
    if (report_progress.has_value()) (*report_progress)(1.0);
    return InMemoryClusterer::Clustering();
  }
  
  InMemoryClusterer::Clustering clustering =
      AllSingletonsClustering(graph_.Graph()->n);
  RETURN_IF_ERROR(
      ParallelModularityClusterer::RefineClustersWithProgressReporting(
          config, &clustering, std::move(report_progress)));
  return clustering;
}

namespace {

absl::StatusOr<ClusteringHelper> GetClusteringHelper(
    const GbbsGraph& graph,
    const InMemoryClusterer::Clustering& initial_clustering,
    const ClustererConfig& clusterer_config) {
  // Set modularity clustering config.
  ClustererConfig modularity_config;
  *modularity_config.mutable_correlation_clusterer_config() =
      clusterer_config.modularity_clusterer_config().correlation_config();
  ASSIGN_OR_RETURN(EdgeWeightStats edge_weight_stats,
                   ComputeEdgeWeightStats(*graph.Graph()));
  modularity_config.mutable_correlation_clusterer_config()->set_resolution(
      edge_weight_stats.total_edge_weight == 0.0
          ? 0.0
          : clusterer_config.modularity_clusterer_config().resolution() /
                edge_weight_stats.total_edge_weight);

  modularity_config.mutable_correlation_clusterer_config()
      ->set_edge_weight_offset(0.0);
  return ClusteringHelper(static_cast<NodeId>(graph.Graph()->n),
                          modularity_config,
                          std::move(edge_weight_stats.weighted_node_degrees),
                          initial_clustering, graph.GetNodeParts());
}

}  // namespace

absl::StatusOr<std::vector<NodeId>>
ParallelModularityClusterer::ClusterAndReturnClusterIdsWithProgressReporting(
    const graph_mining::in_memory::ClustererConfig& config,
    std::optional<gbbs::ReportProgressCallback> report_progress) const {
  
  if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "'ClusterAndReturnClusterIds' cannot be called before 'FinishImport' "
        "is called for the graph");
  }
  if (graph_.Graph()->n == 0) {
    if (report_progress.has_value()) (*report_progress)(1.0);
    return std::vector<NodeId>();
  }
  InMemoryClusterer::Clustering clustering =
      AllSingletonsClustering(graph_.Graph()->n);
  ASSIGN_OR_RETURN(ClusteringHelper helper,
                   GetClusteringHelper(graph_, clustering, config));
  ASSIGN_OR_RETURN(
      std::vector<ClusterId> cluster_ids,
      ParallelCorrelationClusterer::RefineClustersWithProgressReporting(
          clustering, helper, std::move(report_progress)));
  // TODO: b/399828374 - Avoid this conversion by changing the definition of
  // `ClusterId`.
  std::vector<NodeId> cluster_ids_converted(cluster_ids.size());
  parlay::parallel_for(0, cluster_ids.size(), [&](std::size_t i) {
    cluster_ids_converted[i] = static_cast<NodeId>(cluster_ids[i]);
  });
  return cluster_ids_converted;
}

absl::Status ParallelModularityClusterer::RefineClustersWithProgressReporting(
    const ClustererConfig& clusterer_config, Clustering* initial_clustering,
    std::optional<gbbs::ReportProgressCallback> report_progress) const {
  if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "'RefineClusters' cannot be called before 'FinishImport' is called for "
        "the graph");
  }
  ASSIGN_OR_RETURN(
      ClusteringHelper helper,
      GetClusteringHelper(graph_, *initial_clustering, clusterer_config));
  ASSIGN_OR_RETURN(
      std::vector<ClusterId> cluster_ids,
      ParallelCorrelationClusterer::RefineClustersWithProgressReporting(
          *initial_clustering, helper, std::move(report_progress)));
  *initial_clustering =
      graph_mining::in_memory::OutputIndicesById<ClusterId, NodeId>(
          cluster_ids);
  return absl::OkStatus();
}

}  // namespace graph_mining::in_memory
