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

#include "in_memory/clustering/coconductance/coconductance.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/coconductance/coconductance.pb.h"
#include "in_memory/clustering/coconductance/coconductance_internal.h"
#include "in_memory/clustering/compress_graph.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"

namespace graph_mining {
namespace in_memory {
namespace {

using NodeId = InMemoryClusterer::NodeId;
using graph_mining::in_memory::CoconductanceConfig;

std::vector<NodeId> FlattenClustering(
    const std::vector<NodeId>& cluster_ids,
    const std::vector<NodeId>& compressed_cluster_ids) {
  ABSL_CHECK_EQ(compressed_cluster_ids.size(), cluster_ids.size());
  ABSL_CHECK_LE(cluster_ids.size(), std::numeric_limits<NodeId>::max());
  NodeId n = cluster_ids.size();
  std::vector<NodeId> result(n);
  for (NodeId i = 0; i < n; ++i) {
    ABSL_CHECK_GE(cluster_ids[i], 0);
    ABSL_CHECK_LT(cluster_ids[i], n);

    result[i] = compressed_cluster_ids[cluster_ids[i]];
  }
  return result;
}

std::vector<NodeId> EmptyClusters(const std::vector<NodeId> cluster_ids) {
  std::vector<int> cluster_sizes(cluster_ids.size());
  std::vector<NodeId> empty_clusters;
  for (NodeId node = 0; node < cluster_ids.size(); ++node) {
    ++cluster_sizes[cluster_ids[node]];
  }
  for (NodeId cluster = 0; cluster < cluster_sizes.size(); ++cluster) {
    if (cluster_sizes[cluster] == 0) empty_clusters.push_back(cluster);
  }
  return empty_clusters;
}

bool SingleNodeMove(const SimpleUndirectedGraph& graph, double exponent,
                    std::vector<NodeId>* cluster_ids) {
  ClusteringState state = InitialState(graph);
  bool move_made = true;

  double total_gain = 0;
  absl::BitGen bitgen;
  while (move_made) {
    move_made = false;

    std::vector<NodeId> permutation(graph.NumNodes());
    for (int i = 0; i < permutation.size(); ++i) permutation[i] = i;
    std::shuffle(permutation.begin(), permutation.end(), bitgen);

    // We find empty clusters - these are the clusters that nodes which want to
    // leave their clusters may join.
    std::vector<NodeId> empty_clusters = EmptyClusters(state.cluster_ids);

    for (NodeId node : permutation) {
      absl::flat_hash_map<int, double> edges_to_cluster;
      for (const auto& [neighbor_id, weight] : graph.Neighbors(node)) {
        edges_to_cluster[state.cluster_ids.at(neighbor_id)] += weight;
      }
      double edges_to_current_cluster =
          edges_to_cluster[state.cluster_ids.at(node)];

      double best_delta = 0;
      absl::optional<NodeId> best_cluster;
      double best_cluster_edges;

      auto try_move = [&](NodeId new_cluster, double edges_to_new_cluster) {
        double obj_delta = ObjectiveChangeAfterMove(
            node, new_cluster, graph, state, edges_to_current_cluster,
            edges_to_new_cluster, exponent);

        if (obj_delta > best_delta) {
          best_delta = obj_delta;
          best_cluster = new_cluster;
          best_cluster_edges = edges_to_new_cluster;
          return true;
        }
        return false;
      };

      for (const auto& [new_cluster, edges_to_new_cluster] : edges_to_cluster) {
        try_move(new_cluster, edges_to_new_cluster);
      }
      // Try forming a new cluster
      if (!empty_clusters.empty()) {
        auto new_cluster = empty_clusters.back();
        state.cluster_edges[new_cluster] = 0;
        state.cluster_weight[new_cluster] = 0;
        if (try_move(new_cluster, 0.0)) empty_clusters.pop_back();
      }

      if (best_cluster.has_value()) {
        total_gain += best_delta;
        move_made = true;
        MoveNodeAndUpdateState(state, node, *best_cluster, graph,
                               edges_to_current_cluster, best_cluster_edges);
      }
    }
  }
  *cluster_ids = std::move(state.cluster_ids);
  return total_gain > 0;
}

// Louvain heuristic. In each round iterate through nodes and perform single
// node moves until convergence. A single node move moves a node to a cluster
// of its neighbor, chosen to maximize the gain in the objective. Once no good
// single node move exists, contract the clusters to nodes and repeat.
// Note that the clustering objective on the contracted graph is equal to the
// objective on the corresponding clustering of the original graph. This
// relies on the fact that a node representing cluster C has a self-loop of
// weight being equal to the total weight of edges inside cluster C.
absl::StatusOr<std::vector<NodeId>> LouvainCoconductance(
    std::unique_ptr<SimpleUndirectedGraph> graph, double exponent) {
  std::vector<NodeId> cluster_ids;
  std::vector<NodeId> final_cluster_ids(graph->NumNodes());

  std::iota(final_cluster_ids.begin(), final_cluster_ids.end(), 0);

  while (SingleNodeMove(*graph, exponent, &cluster_ids)) {
    ASSIGN_OR_RETURN(graph,
                     CompressGraph(*graph, cluster_ids, std::plus<double>(),
                                   /*ignore_self_loops=*/false));
    final_cluster_ids = FlattenClustering(final_cluster_ids, cluster_ids);
  }
  return final_cluster_ids;
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering> CoconductanceClusterer::Cluster(
    const graph_mining::in_memory::ClustererConfig& config) const {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  RETURN_IF_ERROR(CopyGraph(graph_, graph.get()));
  for (NodeId i = 0; i < graph->NumNodes(); ++i) {
    graph->SetNodeWeight(i, graph->WeightedDegree(i));
  }

  auto coconductance_config = config.coconductance_config();
  if (coconductance_config.has_exponent()) {
    ABSL_CHECK_EQ(coconductance_config.algorithm_case(),
                  CoconductanceConfig::ALGORITHM_NOT_SET);
    coconductance_config.mutable_louvain()->set_exponent(
        coconductance_config.exponent());
  }

  std::vector<NodeId> cluster_ids;
  switch (coconductance_config.algorithm_case()) {
    case CoconductanceConfig::kLouvain:
    case CoconductanceConfig::ALGORITHM_NOT_SET: {
      ASSIGN_OR_RETURN(
          cluster_ids,
          LouvainCoconductance(std::move(graph),
                               coconductance_config.louvain().exponent()));
      break;
    }
    case CoconductanceConfig::kConstantApproximate:
      cluster_ids = ConstantApproximateCoconductance(
          *graph,
          coconductance_config.constant_approximate().num_repetitions());
      break;
    default:
      return absl::InvalidArgumentError("Unknown Coconductance algorithm.");
  }

  return ClusterIdSequenceToClustering(cluster_ids);
}

}  // namespace in_memory
}  // namespace graph_mining
