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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_H_

#include <algorithm>
#include <utility>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/coconductance/coconductance.pb.h"
#include "in_memory/clustering/coconductance/coconductance_internal.h"
#include "in_memory/clustering/compress_graph.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"

namespace graph_mining {
namespace in_memory {

class CoconductanceClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const override;

 private:
  SimpleUndirectedGraph graph_;
};

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_H_
