// Copyright 2010-2023 Google LLC
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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_H_

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

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

namespace graph_mining::in_memory {

class ParHacClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ::research_graph::in_memory::ClustererConfig& config)
      const override;

  absl::StatusOr<Dendrogram> HierarchicalCluster(
      const ::research_graph::in_memory::ClustererConfig& config)
      const override;

 private:
  GbbsGraph graph_;

  // Runs the ParHac algorithm described in go/parhac-paper. The algorithm
  // computes a (1+epsilon)^2 approximate clustering, which means that when a
  // node v is merged to a node w, the similarity of this (v, w) edge is within
  // a factor of (1+epsilon)^2 of the similarity of the best edge currently in
  // the graph.
  //
  // The algorithm buckets the edges into buckets using
  // log_{1+epsilon}(similarity). Since the similarities of edges change over
  // the course of the algorithm, the algorithm doesn't explicitly maintain this
  // bucketing (although we could try this in a future version), and instead
  // when it needs the next bucket, it computes the current maximum similarity,
  // W_{max}, and takes the bucket to be edges with similarity in
  // [W_{max}/(1+epsilon), W_{max}].
  //
  // Within each bucket, the algorithm runs a randomized sub-routine,
  // ProcessHacBucketRandomized, which is described in detail in
  // parhac-internal.h.
  template <typename ClusteredGraph>
  void ParHacImplementation(ClusteredGraph& clustered_graph, double epsilon,
                            double linkage_threshold) const;
};

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_H_
