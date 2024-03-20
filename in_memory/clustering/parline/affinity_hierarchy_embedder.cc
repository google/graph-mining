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

#include "in_memory/clustering/parline/affinity_hierarchy_embedder.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/affinity/parallel_affinity_internal.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

using graph_mining::in_memory::AffinityClustererConfig;

namespace graph_mining::in_memory {
namespace {

using WeightedUndirectedGraph =
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;

const int kMaxAffinityLevels = 40;
const int kDefaultTargetClusterSize = 2;

// Extends the path {C_0, C_1, ..., C_k} at each entry to
// {C_0, C_1, ..., C_k, C_{k+1}} where next_level_cluster_ids[C_k] = C_{k+1}.
// This is basically adding the next level of cluster ids to the paths.
// If a node is isolated then its path is not extended.
void ExtendHierarchyPaths(
    const std::vector<gbbs::uintE>& next_level_cluster_ids,
    const WeightedUndirectedGraph& graph, int level,
    std::vector<std::vector<gbbs::uintE>>* hierarchy_paths) {
  parlay::parallel_for(0, hierarchy_paths->size(), [&](std::size_t i) {
    auto& path = hierarchy_paths->at(i);
    // Check if top was already reached previously.
    if (path.size() < level) return;
    auto node_id = path.back();
    // Only extend the path for non-isolated nodes.
    if (graph.vertices[node_id].out_degree() > 0) {
      path.push_back(next_level_cluster_ids[node_id]);
    }
  });
}

AffinityClustererConfig AffinityConfigWithDefaults(
    AffinityClustererConfig affinity_config) {
  // Set some defaults if needed.
  // TODO: Investigate if dynamic edge weight thresholding can improve
  // the embedding. NOTE: for a related study.
  if (!affinity_config.has_edge_aggregation_function()) {
    affinity_config.set_edge_aggregation_function(AffinityClustererConfig::SUM);
  }
  if (!affinity_config.has_size_constraint()) {
    affinity_config.mutable_size_constraint()->set_target_cluster_size(
        kDefaultTargetClusterSize);
  }
  return affinity_config;
}

}  // namespace

AffinityHierarchyEmbedder::AffinityHierarchyEmbedder(
    const AffinityClustererConfig& config) {
  affinity_config_ = AffinityConfigWithDefaults(config);
}

absl::StatusOr<std::vector<std::vector<gbbs::uintE>>>
AffinityHierarchyEmbedder::ComputeAffinityHierarchyPaths(
    const GbbsGraph& graph) {
  
  std::size_t num_nodes = graph.Graph()->n;
  std::vector<std::vector<gbbs::uintE>> hierarchy_paths(num_nodes);
  // Hierarchy path for each node starts with its own id.
  parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
    hierarchy_paths[i] = {static_cast<gbbs::uintE>(i)};
  });
  std::unique_ptr<WeightedUndirectedGraph> compressed_graph;

  // We use empty node weights vector because we want the default node weight of
  // 1 for nodes since we are only interested in the "number of nodes" as the
  // cluster size for every level. The purpose is to limit the degree of the
  // nodes in the affinity hierarchy which in general produces better embedding
  // for smaller degrees.
  // TODO: A much cleaner way of doing this is to introduce a
  // "NodeAggregationFunction" field in AffinityClustererConfig which is used
  // when compressing graph after each level.
  internal::SizeConstraintConfig size_constraint_config{
      affinity_config_.size_constraint(), {}};

  bool top_reached = false;
  // TODO: We should be able to use the clustering hierarchy returned by
  // ParallelAffinityClusterer::HierarchicalCluster function (once max-size
  // constrained clustering and similar stopping condition is supported there.)
  for (int level = 1; !top_reached; ++level) {
    ABSL_LOG(INFO) << "Affinity Clustering level " << level;
    WeightedUndirectedGraph* current_graph =
        (level == 1) ? graph.Graph() : compressed_graph.get();

    std::vector<gbbs::uintE> compressed_cluster_ids;
    ASSIGN_OR_RETURN(
        compressed_cluster_ids,
        NearestNeighborLinkage(*current_graph,
                               /*weight_threshold=*/0.0,
                               std::make_optional(size_constraint_config)));

    ExtendHierarchyPaths(compressed_cluster_ids, *current_graph, level,
                         &hierarchy_paths);

    GraphWithWeights new_compressed_graph;
    ASSIGN_OR_RETURN(new_compressed_graph,
                     CompressGraph(*current_graph, {},
                                   compressed_cluster_ids, affinity_config_));
    compressed_graph.swap(new_compressed_graph.graph);

    // Top is reached if all the nodes in the compressed_graph are isolated.
    parlay::sequence<bool> isolated_nodes =
        parlay::sequence<bool>::from_function(
            compressed_graph->n, [&](std::size_t i) {
              return compressed_graph->get_vertex(i).out_degree() == 0;
            });
    top_reached = parlay::reduce(
        isolated_nodes,
        parlay::make_monoid([](bool a, bool b) { return a && b; }, true));

    if (top_reached) {
      ABSL_LOG(INFO) << absl::StrFormat("Top reached after %d levels", level);
    } else if (level >= kMaxAffinityLevels) {
      ABSL_LOG(WARNING) << absl::StrFormat(
          "Have not reached the top after %d levels.", level);
      break;
    }
  }

  return hierarchy_paths;
}

absl::StatusOr<std::vector<gbbs::uintE>>
AffinityHierarchyEmbedder::EmbedGraph(const GbbsGraph& graph) {
  
  std::vector<std::vector<gbbs::uintE>> paths;
  ASSIGN_OR_RETURN(paths, ComputeAffinityHierarchyPaths(graph));
  std::vector<gbbs::uintE> embedding(paths.size());
  parlay::parallel_for(0, paths.size(), [&](std::size_t i) {
    embedding[i] = {static_cast<gbbs::uintE>(i)};
  });
  // Sort the nodes lexicographically according to their hierarchy paths.
  parlay::sort_inplace(embedding, [&paths](std::size_t i, std::size_t j) {
    const auto& p1 = paths[i];
    const auto& p2 = paths[j];
    if (p1.size() != p2.size()) {
      return p1.size() < p2.size();
    }
    return std::lexicographical_compare(p1.rbegin(), p1.rend(), p2.rbegin(),
                                        p2.rend());
  });
  return embedding;
}

absl::StatusOr<std::vector<std::pair<gbbs::uintE, double>>>
AffinityHierarchyEmbedder::EmbedGraphWeighted(const GbbsGraph& graph) {
  
  const auto& g = graph.Graph();
  std::size_t num_nodes = g->num_vertices();
  if (g->vertex_weights == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input graph does not have any node weights."));
  }
  const auto* node_weights = g->vertex_weights;
  std::vector<gbbs::uintE> embedding;
  ASSIGN_OR_RETURN(embedding, EmbedGraph(graph));
  std::vector<double> weight_prefix_sums(num_nodes);
  parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
    weight_prefix_sums[i] = node_weights[embedding[i]];
  });
  parlay::scan_inplace(weight_prefix_sums);
  std::vector<std::pair<gbbs::uintE, double>> weighted_embedding(num_nodes);
  parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
    weighted_embedding[i] = std::make_pair(embedding[i], weight_prefix_sums[i]);
  });
  return weighted_embedding;
}

}  // namespace graph_mining::in_memory
