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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_H_

#include <cstddef>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/correlation/parallel_correlation_util.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
class ParallelCorrelationClusterer : public InMemoryClusterer {
 public:
  using ClusterId = gbbs::uintE;

  ~ParallelCorrelationClusterer() override {}

  absl::Nonnull<Graph*> MutableGraph() ABSL_ATTRIBUTE_LIFETIME_BOUND override {
    return &graph_;
  }

  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override;

  absl::StatusOr<std::vector<NodeId>> ClusterAndReturnClusterIds(
      const graph_mining::in_memory::ClustererConfig& config) const override;

  // initial_clustering must include every node in the range
  // [0, number of nodes in MutableGraph()) exactly once.
  absl::Status RefineClusters(
      const graph_mining::in_memory::ClustererConfig& clusterer_config,
      Clustering* initial_clustering) const override;

 protected:
  absl::StatusOr<std::vector<ClusterId>> RefineClusters(
      const InMemoryClusterer::Clustering& initial_clustering,
      ClusteringHelper& initial_helper) const;

  // Returns an all-singletons clustering with the given number of nodes.
  static InMemoryClusterer::Clustering AllSingletonsClustering(
      size_t num_nodes);

  graph_mining::in_memory::GbbsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_CORRELATION_H_
