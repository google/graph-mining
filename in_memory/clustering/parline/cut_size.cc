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

#include "in_memory/clustering/parline/cut_size.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/scheduler.h"
#include "parlay/monoid.h"
#include "parlay/parallel.h"

namespace graph_mining::in_memory {
namespace {

using uintE = gbbs::uintE;

// Converts a clustering in a vector of vectors format to a compact vector
// format where entry i is the cluster-id for node i. Caller must make sure that
// the total number of node ids in input clustering is equal to num_nodes and
// that all those ids are less than num_nodes.
std::vector<uintE> ConvertToCompactClustering(
    const InMemoryClusterer::Clustering& clustering, uintE num_nodes) {
  std::vector<uintE> compact_clustering(num_nodes);
  parlay::parallel_for(
      0, clustering.size(),
      [&](std::size_t cluster_id) {
        const auto& cluster = clustering[cluster_id];
        parlay::parallel_for(0, cluster.size(), [&](std::size_t i) {
          ABSL_CHECK_GE(cluster[i], 0)
              << "Node ids in clustering must be between 0 and num_nodes-1";
          ABSL_CHECK_LT(cluster[i], num_nodes)
              << "Node ids in clustering must be between 0 and num_nodes-1";
          compact_clustering[cluster[i]] = cluster_id;
        });
      },
      /*granularity=*/1);
  return compact_clustering;
}

}  // namespace

absl::StatusOr<double> ComputeCutRatio(const std::vector<uintE>& clustering,
                                       const GbbsGraph& graph) {
  
  // Cut-weight and weight sums as a pair.
  using WeightPair = std::pair<double, double>;
  auto map_fn = [&](uintE id, uintE neighbor_id, float weight) -> WeightPair {
    float cut_weight = clustering[id] != clustering[neighbor_id] ? weight : 0;
    return WeightPair(cut_weight, weight);
  };
  auto reduce_fn = parlay::make_monoid(
      [](const WeightPair& a, const WeightPair& b) {
        return WeightPair(a.first + b.first, a.second + b.second);
      },
      WeightPair(0, 0));
  // The first value in the returned pair is the total cut-size and the second
  // entry is the total edge weight of the input graph.
  auto weight_sum_pair = graph.Graph()->reduceEdges(map_fn, reduce_fn);
  if (weight_sum_pair.second == 0) {
    return absl::InvalidArgumentError("Total edge weight in input graph is 0");
  }
  return weight_sum_pair.first / weight_sum_pair.second;
}

absl::StatusOr<double> ComputeCutRatio(
    const InMemoryClusterer::Clustering& clustering, const GbbsGraph& graph) {
  
  return ComputeCutRatio(
      ConvertToCompactClustering(clustering, graph.Graph()->n), graph);
}

}  // namespace graph_mining::in_memory
