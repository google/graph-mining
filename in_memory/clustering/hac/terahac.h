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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_TERAHAC_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_TERAHAC_H_

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
#include "in_memory/clustering/hac/terahac_internal.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parallel_clustered_graph.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/scheduler.h"

namespace graph_mining::in_memory {

class TeraHacClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ::graph_mining::in_memory::ClustererConfig& config)
      const override;

  absl::StatusOr<Dendrogram> HierarchicalCluster(
      const ::graph_mining::in_memory::ClustererConfig& config)
      const override;

 private:
  GbbsGraph graph_;
  
  double GetEpsilon(
      const ::graph_mining::in_memory::ClustererConfig& config) const;

  // Runs a shared-memory version of the TeraHac algorithm described in
  // https://arxiv.org/abs/2308.03578.
  // TODO: implement edge-pruning.
  template <typename ClusteredGraph>
  void TeraHacImplementation(ClusteredGraph& clustered_graph, double epsilon,
                             double linkage_threshold) const;
};

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_TERAHAC_H_
