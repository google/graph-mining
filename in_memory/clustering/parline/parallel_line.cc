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

#include "in_memory/clustering/parline/parallel_line.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parline/affinity_hierarchy_embedder.h"
#include "in_memory/clustering/parline/linear_embedder.h"
#include "in_memory/clustering/parline/pairwise_improver.h"
#include "in_memory/clustering/parline/parline.pb.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

using graph_mining::in_memory::AffinityClustererConfig;
using graph_mining::in_memory::ClustererConfig;

namespace graph_mining::in_memory {
namespace {

// TODO: This function should be in a util.
std::unique_ptr<LinearEmbedder> CreateEmbedder(
    const LinePartitionerConfig& line_config) {
  // Currently we support only one embedding method.
  const EmbedderConfig& embedder_config = line_config.embedder_config();
  switch (embedder_config.embedder_config_case()) {
    case EmbedderConfig::kAffinityConfig: {
      return std::make_unique<AffinityHierarchyEmbedder>(
          embedder_config.affinity_config());
    }
    default: {
      ABSL_LOG(INFO)
          << "No embedder_config is specified. Using affinity_config "
             "embedding with default values.";
      return std::make_unique<AffinityHierarchyEmbedder>(
          AffinityClustererConfig());
    }
  }
}

parlay::sequence<int> ComputeClusterSizePrefixSum(
    const std::vector<gbbs::uintE>& embedding, int num_clusters) {
  int num_nodes = embedding.size();
  int cluster_size = num_nodes / num_clusters;
  int remainder = num_nodes % num_clusters;
  std::vector<int> cluster_sizes(num_clusters);
  parlay::parallel_for(0, num_clusters, [&](std::size_t i) {
    cluster_sizes[i] = cluster_size;
    // If num_nodes is not divisible by num_clusters then the excess size is
    // distributed to the first 'remainder' clusters (one for each), which
    // guarantees that all the cluster sizes are within +-1 of each other.
    if (i < remainder) ++cluster_sizes[i];
  });
  return parlay::scan_inclusive(cluster_sizes);
}

// Divides a linear embedding of nodes where a cluster size is the number of
// nodes in it.
// TODO: Move this and the SliceEmbeddingWeighted function below into
// a separate file with more unit tests.
absl::StatusOr<InMemoryClusterer::Clustering> SliceEmbedding(
    const GbbsGraph& graph, int num_clusters,
    const LinePartitionerConfig& line_config) {
  std::unique_ptr<LinearEmbedder> embedder = CreateEmbedder(line_config);
  ASSIGN_OR_RETURN(std::vector<gbbs::uintE> embedding,
                   embedder->EmbedGraph(graph));
  ABSL_VLOG(1) << "Done with (node-unweighted) embedding";
  auto cluster_size_prefix_sum =
      ComputeClusterSizePrefixSum(embedding, num_clusters);
  InMemoryClusterer::Clustering clustering(num_clusters);
  parlay::parallel_for(0, num_clusters, [&](std::size_t i) {
    int start = i == 0 ? 0 : cluster_size_prefix_sum[i - 1];
    int end = cluster_size_prefix_sum[i];
    std::vector<int>& cluster = clustering[i];
    cluster.reserve(end - start);
    std::copy(embedding.cbegin() + start, embedding.cbegin() + end,
              std::back_inserter(cluster));
  });
  ABSL_VLOG(1) << "Done with (node-unweighted) initial slicing";
  return clustering;
}

double ComputeTotalNodeWeight(const LinePartitionerConfig& config,
                              const GbbsGraph& gbbs_graph) {
  auto graph = gbbs_graph.Graph();
  if (!config.use_node_weights() || graph->vertex_weights == nullptr) {
    return graph->num_vertices();
  }
  auto weight_seq = parlay::delayed_tabulate<double>(
      graph->n, [&](size_t i) { return graph->vertex_weights[i]; });
  return parlay::reduce(weight_seq, parlay::addm<double>());
}

// Divides a weighted linear embedding of nodes where a cluster size is the sum
// of node weights in it.
absl::StatusOr<InMemoryClusterer::Clustering> SliceEmbeddingWeighted(
    const GbbsGraph& graph, int num_clusters,
    const LinePartitionerConfig& line_config) {
  const double cluster_weight =
      ComputeTotalNodeWeight(line_config, graph) / num_clusters;
  std::unique_ptr<LinearEmbedder> embedder = CreateEmbedder(line_config);
  std::vector<std::pair<gbbs::uintE, double>> embedding;
  ASSIGN_OR_RETURN(embedding, embedder->EmbedGraphWeighted(graph));
  ABSL_VLOG(1) << "Done with (node-weighted) embedding";
  auto clustering_seq =
      parlay::delayed_tabulate<std::pair<gbbs::uintE, gbbs::uintE>>(
          embedding.size(), [&](size_t i) {
            return std::make_pair(
                std::floor(embedding[i].second / cluster_weight),
                embedding[i].first);
          });
  auto grouped = parlay::group_by_key(clustering_seq);
  InMemoryClusterer::Clustering clustering(grouped.size());
  parlay::parallel_for(0, grouped.size(), [&](std::size_t i) {
    auto part = grouped[i].second;
    clustering[i] =
        std::vector<InMemoryClusterer::NodeId>(part.begin(), part.end());
  });
  ABSL_VLOG(1) << "Done with (node-weighted) initial slicing";
  return clustering;
}

// TODO: Refactor this function into a common util.
absl::StatusOr<int> GetNumberOfClusters(const LinePartitionerConfig& config,
                                        const GbbsGraph& gbbs_graph) {
  if (!config.has_num_clusters() && !config.has_cluster_weight()) {
    return absl::InvalidArgumentError(
        "Either line_config.num_clusters or line_config.cluster_size must be "
        "specified.");
  }
  if (config.has_num_clusters()) {
    const int num_clusters = config.num_clusters();
    if (num_clusters <= 1) {
      return absl::InvalidArgumentError(
          "line_config.num_clusters must be at least 2");
    }
    return num_clusters;
  } else {
    const double cluster_size = config.cluster_weight();
    if (cluster_size <= 0) {
      return absl::InvalidArgumentError(
          "line_config.cluster_size must be a positive non-zero value");
    }
    double total_node_weight = ComputeTotalNodeWeight(config, gbbs_graph);
    if (total_node_weight <= cluster_size) {
      return absl::InvalidArgumentError(
          "line_config.cluster_size must be less than total node weight");
    }
    return std::ceil(total_node_weight / cluster_size);
  }
}

InMemoryClusterer::Clustering ImproveClusters(
    const GbbsGraph& graph,
    const InMemoryClusterer::Clustering& initial_clusters,
    const LinePartitionerConfig& line_config) {
  if (initial_clusters.empty()) return initial_clusters;
  if (!line_config.has_local_search_config()) {
    ABSL_LOG(INFO) << "No local_search_config set, ignoring post processing.";
  } else if (line_config.local_search_config().has_pairwise_improver_config()) {
    return ImproveClustersPairwise(graph, initial_clusters, line_config);
  }
  return initial_clusters;
}

absl::StatusOr<InMemoryClusterer::Clustering> ComputeInitialClusters(
    const GbbsGraph& graph, const LinePartitionerConfig& line_config) {
  ASSIGN_OR_RETURN(const int num_clusters,
                   GetNumberOfClusters(line_config, graph));
  return line_config.use_node_weights()
             ? SliceEmbeddingWeighted(graph, num_clusters, line_config)
             : SliceEmbedding(graph, num_clusters, line_config);
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering> ParallelLinePartitioner::Cluster(
    const ClustererConfig& config) const {
  
  ABSL_CHECK(graph_.Graph() != nullptr) << "Input graph not specified";
  const LinePartitionerConfig& line_config = config.line_partitioner_config();
  double* node_weights = nullptr;
  if (!line_config.use_node_weights()) {
    std::swap(node_weights, graph_.Graph()->vertex_weights);
  }
  ASSIGN_OR_RETURN(InMemoryClusterer::Clustering initial_clusters,
                   ComputeInitialClusters(graph_, line_config));
  InMemoryClusterer::Clustering improved_clusters =
      ImproveClusters(graph_, initial_clusters, line_config);
  ABSL_VLOG(1) << "Done with post processing to improve clusters";
  if (!line_config.use_node_weights()) {
    graph_.Graph()->vertex_weights = node_weights;
  }
  return improved_clusters;
}

}  // namespace graph_mining::in_memory
