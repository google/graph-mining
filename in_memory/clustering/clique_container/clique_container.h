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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_CLIQUE_CONTAINER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_CLIQUE_CONTAINER_H_

#include "absl/status/statusor.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining {
namespace in_memory {

// Computes a collection of overlapping clusters, that satisfies the following
// two guarantees:
//  * Each clique in the graph is fully contained in at least one cluster.
//  * The density of each cluster is at least `min_density` (which is a
//  parameter of the algorithm).
// Here, the density of a cluster is defined as the number of edges in the
// cluster divided by the number of possible edges in the cluster, i.e. (#nodes
// choose 2). As long as the density parameter is <= 0.9, in typical cases the
// number of clusters returned is similar to the number of nodes in the graph.
// For the algorithm description and analysis, see go/dense-subgraphs-draft.
class CliqueContainerClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override;

 private:
  SimpleUndirectedGraph graph_;
};

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_CLIQUE_CONTAINER_H_
