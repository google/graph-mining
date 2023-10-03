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

#include "in_memory/generation/barabasi_albert.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> UnweightedBarabasiAlbert(
    size_t num_nodes, size_t num_initial_nodes, size_t edges_per_node) {
  ABSL_CHECK_GE(num_nodes, 1);
  ABSL_CHECK_GE(edges_per_node, 1);
  ABSL_CHECK_LE(edges_per_node, num_initial_nodes);
  ABSL_CHECK_LE(num_initial_nodes, num_nodes);

  // A node with degree d has its node_id appear d times in the node_ids array.
  // This array is used to sample a new neighbor with probability proportional
  // to its degree.
  std::vector<int32_t> node_ids;
  node_ids.reserve(num_nodes * edges_per_node);

  auto result = std::make_unique<SimpleUndirectedGraph>();
  result->SetNumNodes(num_nodes);
  // Create a line graph on the initial nodes.
  for (size_t i = 0; i < num_initial_nodes - 1; ++i) {
    RETURN_IF_ERROR(result->AddEdge(i, i + 1, 1.0));
    // Add these ids to the node_ids vector (which we use for sampling).
    node_ids.push_back(i);
    node_ids.push_back(i + 1);
  }
  // Add the last cycle edge.
  if (num_initial_nodes > 1) {
    RETURN_IF_ERROR(result->AddEdge(0, num_initial_nodes - 1, 1.0));
    node_ids.push_back(0);
    node_ids.push_back(num_initial_nodes - 1);
  }

  absl::BitGen gen;
  auto sample_edge = [&]() {
    int32_t index = absl::Uniform<int32_t>(gen, 0, node_ids.size());
    return node_ids[index];
  };

  for (size_t i = num_initial_nodes; i < num_nodes; ++i) {
    size_t edges_added = 0;
    while (edges_added < edges_per_node) {
      int32_t neighbor_id = sample_edge();
      ABSL_CHECK_LE(neighbor_id, i);

      if (!result->EdgeWeight(i, neighbor_id).has_value()) {
        RETURN_IF_ERROR(result->AddEdge(i, neighbor_id, 1.0));
        node_ids.push_back(neighbor_id);
        ++edges_added;
      }
    }
    for (size_t k = 0; k < edges_per_node; ++k) {
      node_ids.push_back(i);
    }
  }

  return result;
}

}  // namespace graph_mining::in_memory
