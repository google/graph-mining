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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_LINEAR_EMBEDDER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_LINEAR_EMBEDDER_H_

#include "absl/status/statusor.h"
#include "in_memory/clustering/gbbs_graph.h"

namespace graph_mining::in_memory {

// An interface for embedding the nodes of a graph into a line.
class LinearEmbedder {
 public:
  LinearEmbedder() = default;
  LinearEmbedder(const LinearEmbedder&) = delete;
  LinearEmbedder(LinearEmbedder&&) = delete;
  virtual ~LinearEmbedder() = default;

  // Embeds the nodes of an input graph into a line. The entries of the returned
  // vector are node ids ordered based on the underlying embedding
  // implementation.
  virtual absl::StatusOr<std::vector<gbbs::uintE>> EmbedGraph(
      const GbbsGraph& graph) = 0;

  // Embeds the nodes of an input graph into a line using the node weights.
  // The entries of the returned vector are (node_id, location) pairs where the
  // location of an entry is the sum of all the node weights before it.
  virtual absl::StatusOr<std::vector<std::pair<gbbs::uintE, double>>>
  EmbedGraphWeighted(const GbbsGraph& graph) = 0;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_LINEAR_EMBEDDER_H_
