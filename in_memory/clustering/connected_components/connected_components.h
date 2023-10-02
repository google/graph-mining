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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_

#include "in_memory/clustering/connected_components/connected_components_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/parallel/scheduler.h"

namespace graph_mining::in_memory {

class ParallelConnectedComponentsClusterer : public InMemoryClusterer {
 public:
  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const research_graph::in_memory::ClustererConfig& config) const override {
    
    const auto& parents = graph_.ParentArray();

    auto get_clusters = [&](NodeId i) -> NodeId { return i; };

    return graph_mining::in_memory::OutputIndicesById<gbbs::uintE, NodeId>(
        absl::MakeSpan(parents), get_clusters, parents.size());
  }

 private:
  ConnectedComponentsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_H_
