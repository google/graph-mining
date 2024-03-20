/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_ADJACENCY_LISTS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_ADJACENCY_LISTS_H_

#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Returns adjacency lists corresponding to edges in `graph`. Nodes have the
// same weight as the node weight in `graph`. Self edges are ignored. Returns
// error status if any edge weight is negative.
absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
ConvertToAdjList(const SimpleUndirectedGraph& graph);

// Converts `edges` to adjacency lists. All nodes and edges have weight 1.
absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
ConvertToAdjList(absl::Span<const std::pair<graph_mining::in_memory::NodeId,
                                            graph_mining::in_memory::NodeId>>
                     edges);

// Converts `edges` to adjacency lists. Nodes weights are added according to
// `node_weights`. Nodes not in `node_weights` have weight 1.
absl::StatusOr<std::vector<
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>>
ConvertToAdjList(
    absl::Span<const std::tuple<graph_mining::in_memory::NodeId,
                                graph_mining::in_memory::NodeId, double>>
        edges,
    const absl::flat_hash_map<graph_mining::in_memory::NodeId,
                              graph_mining::in_memory::NodeId>& node_weights);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_ADJACENCY_LISTS_H_
