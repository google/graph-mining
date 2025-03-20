// Copyright 2024 Google LLC
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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_

#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Runs Dijkstra's algorithm on `graph` to compute single-source shortest-path
// distances from the source node `source_node_id`. Returns a vector whose
// length is equal to the number of nodes of the input graph, and whose `i`-th
// element is equal to the distance from node `source_node_id` to node `i` if
// node `i` is reachable from node `source_node_id`, or equal to infinity
// otherwise. Returns an error if `source_node_id` is not a valid node ID of the
// graph, or if an edge that can be reached from `source_node_id` via a directed
// path has a negative weight.
absl::StatusOr<std::vector<float>> Dijkstra(
    const graph_mining::in_memory::DirectedGbbsGraph& graph,
    graph_mining::in_memory::NodeId source_node_id);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_
