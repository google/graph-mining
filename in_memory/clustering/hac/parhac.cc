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

#include "in_memory/clustering/hac/parhac.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/parhac_internal.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parallel_clustered_graph.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/scheduler.h"

using ::graph_mining::in_memory::ClustererConfig;

namespace graph_mining::in_memory {

template <typename ClusteredGraph>
void ParHacClusterer::ParHacImplementation(ClusteredGraph& clustered_graph,
                                           double epsilon,
                                           double linkage_threshold) const {
  size_t num_active = clustered_graph.NumNodes();

  size_t num_inner_rounds = 0;
  size_t num_outer_rounds = 0;
  size_t max_inner_rounds = 0;

  double total_lower_threshold_time = 0;
  while (num_active > 1) {
    ++num_outer_rounds;
    double max_weight = 0;
    // Start of a new round. First compute the max edge-weight currently in the
    // graph.
    // TODO: the cost of this step is O(m) currently since we don't use
    // augmentation (I don't see how to do it with hash tables unless we build
    // trees on top of the underlying arrays). But maybe lightweight caching
    // would work (and probably help) here since we can cache the best-weight
    // whenever we compute it, and only recompute whenever edges incident to
    // this node are modified.
    auto lower_threshold_start = absl::Now();
    parlay::parallel_for(0, clustered_graph.NumNodes(), [&](size_t i) {
      if (clustered_graph.MutableNode(i)->IsActive()) {
        auto [id, similarity] = clustered_graph.MutableNode(i)->BestEdge();
        if (id != UINT_E_MAX && similarity > max_weight) {
          gbbs::write_max(&max_weight, similarity);
        }
      }
    });
    double lower_threshold_time =
        absl::ToDoubleSeconds(absl::Now() - lower_threshold_start);
    ABSL_LOG(INFO) << "Compute lower_threshold time " << lower_threshold_time;
    total_lower_threshold_time += lower_threshold_time;
    if (max_weight == 0 || max_weight < linkage_threshold) break;

    auto [num_merged, inner_rounds] = ProcessHacBucketRandomized(
        clustered_graph, max_weight / (1 + epsilon), epsilon);
    num_inner_rounds += inner_rounds;
    max_inner_rounds = std::max(max_inner_rounds, inner_rounds);

    num_active -= num_merged;
    ABSL_LOG(INFO) << "Merged: " << num_merged
              << " nodes. Num. active nodes remaining: " << num_active
              << " Max Weight was: " << max_weight
              << " Linkage_threshold: " << linkage_threshold;
  }
  ABSL_LOG(INFO) << "Total lower_threshold time " << total_lower_threshold_time;
  ABSL_LOG(INFO) << "Num Outer Rounds = " << num_outer_rounds;
  ABSL_LOG(INFO) << "Num Inner Rounds = " << num_inner_rounds;
  ABSL_LOG(INFO) << "Max Inner Rounds = " << max_inner_rounds;
}

absl::StatusOr<ParHacClusterer::Clustering> ParHacClusterer::Cluster(
    const ClustererConfig& config) const {
  
  using Graph = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;
  Graph* input_graph = graph_.Graph();

  double weight_threshold = config.parhac_clusterer_config().weight_threshold();
  double epsilon = 0.1;
  ClusteredGraph<AverageLinkageWeight, Graph> clustered_graph(input_graph);
  if (config.parhac_clusterer_config().has_epsilon()) {
    epsilon = config.parhac_clusterer_config().epsilon();
  }

  ParHacImplementation(clustered_graph, epsilon, weight_threshold);
  return graph_mining::in_memory::ClusterIdsToClustering(
      clustered_graph.GetDendrogram()->GetSubtreeClustering(weight_threshold));
}

absl::StatusOr<Dendrogram> ParHacClusterer::HierarchicalCluster(
    const ClustererConfig& config) const {
  
  using Graph = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>;
  Graph* input_graph = graph_.Graph();
  size_t num_nodes = input_graph->num_vertices();

  double weight_threshold = config.parhac_clusterer_config().weight_threshold();
  double epsilon = 0.1;
  ClusteredGraph<AverageLinkageWeight, Graph> clustered_graph(input_graph);
  if (config.parhac_clusterer_config().has_epsilon()) {
    epsilon = config.parhac_clusterer_config().epsilon();
  }

  ParHacImplementation(clustered_graph, epsilon, weight_threshold);

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
