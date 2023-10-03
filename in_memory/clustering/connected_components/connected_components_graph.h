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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_

#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/connected_components/asynchronous_union_find.h"

namespace graph_mining::in_memory {

class ConnectedComponentsGraph : public InMemoryClusterer::Graph {
 public:
  absl::Status PrepareImport(int64_t num_nodes) override {
    if (union_find_parent_.NumberOfNodes() != 0)
      return absl::FailedPreconditionError("PrepareImport called twice");
    union_find_parent_ = AsynchronousUnionFind<gbbs::uintE>(num_nodes);
    return absl::OkStatus();
  }

  absl::Status Import(AdjacencyList adjacency_list) override {
    if (union_find_parent_.NumberOfNodes() == 0) {
      return absl::FailedPreconditionError(
          "Using ConnectedComponents requires calling PrepareImport.");
    }
    for (const auto& [neighbor_id, weight] : adjacency_list.outgoing_edges) {
      union_find_parent_.Unite(static_cast<unsigned int>(adjacency_list.id),
                               static_cast<unsigned int>(neighbor_id));
    }
    return absl::OkStatus();
  }

  absl::Status FinishImport() override {
    finished_parents_ = std::move(union_find_parent_).ComponentSequence();
    return absl::OkStatus();
  }

  absl::Span<const gbbs::uintE> ParentArray() const {
    return absl::MakeSpan(finished_parents_);
  }

 private:
  AsynchronousUnionFind<gbbs::uintE> union_find_parent_;
  parlay::sequence<gbbs::uintE> finished_parents_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_
