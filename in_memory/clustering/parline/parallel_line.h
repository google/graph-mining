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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PARALLEL_LINE_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PARALLEL_LINE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "parlay/primitives.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parline/affinity_hierarchy_embedder.h"
#include "in_memory/clustering/parline/linear_embedder.h"
#include "in_memory/clustering/parline/parline.pb.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

// An in-memory counterpart of go/line balanced partitioner that is based
// on the GBBS framework.
class ParallelLinePartitioner : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ::graph_mining::in_memory::ClustererConfig& config)
      const override;

 private:
  GbbsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PARALLEL_LINE_H_
