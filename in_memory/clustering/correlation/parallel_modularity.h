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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_MODULARITY_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_MODULARITY_INTERNAL_H_

#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "in_memory/clustering/correlation/parallel_correlation.h"

namespace graph_mining::in_memory {

// Parallel Modularity Clusterer. Uses ParallelCorrelationClusterer to optimize
// the modularity partition score.
class ParallelModularityClusterer : public ParallelCorrelationClusterer {
 public:
  using InMemoryClusterer::Clustering;

  ~ParallelModularityClusterer() override {}

  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const override;

  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once. If it doesn't this function
  // will either return an error or run normally starting from an unspecified
  // clustering. (Currently it always returns an error but this may change in
  // the future.)
  absl::Status RefineClusters(
      const graph_mining::in_memory::ClustererConfig& clusterer_config,
      Clustering* initial_clustering) const override;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_PARALLEL_MODULARITY_INTERNAL_H_
