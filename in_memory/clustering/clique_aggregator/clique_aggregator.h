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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_CLIQUE_AGGREGATOR_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_CLIQUE_AGGREGATOR_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining {
namespace in_memory {

// Computes a collection of potentially overlapping clusters, that satisfies the
// following guarantees:
//  * Each clique of size >=2 in the graph is fully contained in at least one
//    cluster.
//  * The density of each cluster is at least `min_density` (which is a
//    parameter of the algorithm).
//  * No cluster is a subset of another cluster.
// Here, the density of a cluster is defined as the number of edges in the
// cluster divided by the number of possible edges in the cluster, i.e. (#nodes
// choose 2). As long as the density parameter is <= 0.9, in typical cases the
// number of clusters returned is similar to the number of nodes in the graph.
// For the algorithm description and analysis, see go/dense-subgraphs-paper.
class CliqueAggregatorClusterer : public InMemoryClusterer {
 public:
  // Result of the `CliqueAggregatorClusterer`.
  struct ClusteringWithStatistics {
    ClusteringWithStatistics() = delete;

    // If `save_clusters_density` is true, the density of each cluster is saved
    // in the result.
    explicit ClusteringWithStatistics(bool save_cluster_density) {
      if (save_cluster_density) {
        cluster_density.emplace();
      }
    }

    void AddCluster(std::vector<NodeId>&& cluster, double density) {
      clusters.push_back(std::move(cluster));
      if (cluster_density.has_value()) {
        cluster_density->push_back(density);
      }
    }

    // Number of recursive calls to the `CliqueAggregator` function.
    int64_t num_recursive_calls = 0;
    // Number of recursive calls that did not return before computing the
    // degeneracy ordering.
    int64_t num_recursive_calls_not_immediately_pruned = 0;

    Clustering clusters;
    // Either `nullopt` or has the same size as `clusters`. In the latter case,
    // `cluster_density[i]` is the density of `clusters[i]`.
    std::optional<std::vector<double>> cluster_density;
  };

  Graph* MutableGraph() override { return &graph_; }

  // Each cluster is a *sorted* vector of node IDs.
  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override;

  // Same as `Cluster`, but returns additional statistics about the execution.
  // WARNING: This is for testing purposes and is not thoroughly tested.
  absl::StatusOr<ClusteringWithStatistics> ClusterWithStatistics(
      const graph_mining::in_memory::ClustererConfig& config) const;

 private:
  UnweightedGbbsGraph graph_;
};

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_AGGREGATOR_CLIQUE_AGGREGATOR_H_
