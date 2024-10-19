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

#include "in_memory/clustering/hac/terahac.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/bridge.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/hac/terahac_internal.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parallel_clustered_graph.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/parallel/scheduler.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"

using ::graph_mining::in_memory::ClustererConfig;

namespace graph_mining::in_memory {

double TeraHacClusterer::GetEpsilon(const ClustererConfig& config) const {
  static constexpr double kDefaultEpsilon = 0.1;
  return (config.terahac_clusterer_config().has_epsilon())
             ? config.terahac_clusterer_config().epsilon()
             : kDefaultEpsilon;
}

template <typename ClusteredGraph>
void TeraHacClusterer::TeraHacImplementation(ClusteredGraph& clustered_graph,
                                             double epsilon,
                                             double linkage_threshold) const {
  size_t num_inner_rounds = 0;
  size_t num_outer_rounds = 0;
  size_t max_inner_rounds = 0;
  size_t n = clustered_graph.NumNodes();

  parlay::sequence<gbbs::uintE> node_to_partition(
      n, std::numeric_limits<gbbs::uintE>::max());
  float pruning_threshold = linkage_threshold / (1 + epsilon);
  auto min_merge_similarities =
      parlay::sequence<float>(n, std::numeric_limits<float>::infinity());

  auto all_vtxs = parlay::delayed_tabulate(n, [&](gbbs::uintE i) { return i; });
  auto active = parlay::filter(all_vtxs, [&](gbbs::uintE i) {
    bool active = clustered_graph.MutableNode(i)->IsActive();
    if (active) {
      auto [id, similarity] = clustered_graph.MutableNode(i)->BestEdge();
      return similarity > pruning_threshold;
    }
    return false;
  });

  double total_lower_threshold_time = 0;
  while (active.size() > 1) {
    size_t size_constraint = std::max(n / 100, size_t{1000000});
    auto cluster_and_node = SizeConstrainedAffinity(
        clustered_graph, active, size_constraint, num_outer_rounds);

    // === Subgraph HAC Stage ===
    // First sort to get clusters.
    parlay::sort_inplace(cluster_and_node);
    // Find the cluster starts.
    auto cluster_starts = parlay::pack_index(
        parlay::delayed_tabulate(cluster_and_node.size(), [&](size_t i) {
          return (i == 0) ||
                 (cluster_and_node[i].first != cluster_and_node[i - 1].first);
        }));
    size_t num_clusters = cluster_starts.size();

    parlay::parallel_for(0, cluster_and_node.size(), [&](size_t i) {
      auto [cluster_id, node_id] = cluster_and_node[i];
      node_to_partition[node_id] = cluster_id;
    });

    using Merges =
        parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, float>>;
    parlay::sequence<Merges> cluster_merges(num_clusters);

    // Now run SubgraphHAC on each cluster. Parallelism is over the
    // individual clusters.
    parlay::parallel_for(
        0, num_clusters,
        [&](size_t i) {
          size_t begin = cluster_starts[i];
          size_t end = (i == num_clusters - 1) ? cluster_and_node.size()
                                               : cluster_starts[i + 1];
          auto cluster_size = end - begin;
          if (cluster_size > 1) {
            auto cluster_vtx_ids = parlay::delayed_tabulate(
                end - begin,
                [&](size_t j) { return cluster_and_node[begin + j].second; });

            auto cluster_id = cluster_and_node[begin].first;

            auto merges = ApproximateSubgraphHacWrapper(
                clustered_graph, cluster_vtx_ids, node_to_partition, cluster_id,
                min_merge_similarities, epsilon, linkage_threshold,
                num_outer_rounds);
            ABSL_CHECK_OK(merges.status());
            cluster_merges[i] = merges.value();
          }
        },
        1);

    clustered_graph.SubgraphMerges(std::move(cluster_merges));

    // Reset node_to_partition
    parlay::parallel_for(0, cluster_and_node.size(), [&](size_t i) {
      auto [cluster_id, node_id] = cluster_and_node[i];
      node_to_partition[node_id] = std::numeric_limits<gbbs::uintE>::max();
    });

    auto next_active =
        parlay::filter(parlay::make_slice(active), [&](gbbs::uintE u) {
          if (clustered_graph.MutableNode(u)->IsActive()) {
            auto [_, wgh] = clustered_graph.MutableNode(u)->BestEdge();
            return wgh > pruning_threshold;
          }
          return false;
        });

    active = std::move(next_active);

    ++num_outer_rounds;
  }
  ABSL_LOG(INFO) << "Total lower_threshold time " << total_lower_threshold_time;
  ABSL_LOG(INFO) << "Num Outer Rounds = " << num_outer_rounds;
  ABSL_LOG(INFO) << "Num Inner Rounds = " << num_inner_rounds;
  ABSL_LOG(INFO) << "Max Inner Rounds = " << max_inner_rounds;
}

absl::StatusOr<TeraHacClusterer::Clustering> TeraHacClusterer::Cluster(
    const ClustererConfig& config) const {
  
  using Graph = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;
  Graph* input_graph = graph_.Graph();
  ClusteredGraph<AverageLinkageWeight, Graph> clustered_graph(input_graph);

  if (!config.terahac_clusterer_config().has_weight_threshold()) {
    return absl::UnimplementedError("Not implemented");
  }
  double linkage_threshold =
      config.terahac_clusterer_config().weight_threshold();
  double epsilon = GetEpsilon(config);

  TeraHacImplementation(clustered_graph, epsilon, linkage_threshold);
  return graph_mining::in_memory::ClusterIdsToClustering(
      clustered_graph.GetDendrogram()->GetSubtreeClustering(linkage_threshold));
}

absl::StatusOr<Dendrogram> TeraHacClusterer::HierarchicalCluster(
    const ClustererConfig& config) const {
  
  using Graph = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;
  Graph* input_graph = graph_.Graph();
  size_t num_nodes = input_graph->num_vertices();
  ClusteredGraph<AverageLinkageWeight, Graph> clustered_graph(input_graph);

  ABSL_CHECK(config.terahac_clusterer_config().has_weight_threshold());
  double linkage_threshold =
      config.terahac_clusterer_config().weight_threshold();
  double epsilon = GetEpsilon(config);

  TeraHacImplementation(clustered_graph, epsilon, linkage_threshold);

  // Convert from a parallel_dendrogram to a dendrogram. This is a stop-gap
  // measure until we unify parallel-dendrogram in //r/g/ and dendrogram in
  // //third_party/graph_mining.
  auto parallel_dendrogram = clustered_graph.GetDendrogram();

  auto parent_ids =
      parlay::delayed_seq<int64_t>(2 * num_nodes - 1, [&](size_t i) {
        auto [parent_id, _] = parallel_dendrogram->GetParent(i);
        return parallel_dendrogram->HasValidParent(i) ? parent_id : -1;
      });
  auto max_parent = parlay::reduce_max(parent_ids) + 1;

  std::vector<DendrogramNode> dendrogram_nodes(max_parent);
  parlay::parallel_for(0, max_parent, [&](size_t i) {
    auto [parent_id, merge_similarity] = parallel_dendrogram->GetParent(i);
    dendrogram_nodes[i] = {parent_id, merge_similarity};
  });
  Dendrogram result(0);
  auto status = result.Init(std::move(dendrogram_nodes), num_nodes);
  if (!status.ok()) {
    return status;
  } else {
    return result;
  }
}

}  // namespace graph_mining::in_memory
