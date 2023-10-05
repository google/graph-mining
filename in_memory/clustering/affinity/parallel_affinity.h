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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_H_

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/affinity/parallel_affinity_internal.h"
#include "in_memory/clustering/affinity/weight_threshold.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

// An in-memory counterpart of go/affinity-clustering (see the link for
// algorithm description). Also, implements the functionality of
// graph_mining::AffinityPartitioner. The parameters to this clusterer are
// specified using AffinityClustererConfig proto. In the returned clustering,
// each node is assigned to exactly one cluster.
//
// The important addition to distributed affinity clustering (borrowed from
// graph_mining::AffinityPartitioner) is that after each step only nodes
// belonging to "unfinished clusters" are retained for the following step. (an
// "unfinished cluster" may be, e.g., a cluster of sufficient density, or
// conductance - this is controlled by AffinityClustererConfig). Nodes belonging
// to other clusters are removed from the graph and immediately added to the
// output (together with their cluster assignment).

class ParallelAffinityClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ::graph_mining::in_memory::ClustererConfig& config)
      const override;

  absl::StatusOr<std::vector<Clustering>> HierarchicalFlatCluster(
      const ::graph_mining::in_memory::ClustererConfig& config)
      const override;

 private:
  GbbsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_H_
