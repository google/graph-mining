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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_BARABASI_ALBERT_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_BARABASI_ALBERT_H_

#include "in_memory/clustering/graph.h"

namespace graph_mining::in_memory {

// Construct a Barabasi-Albert (BA) preferential attachment graph. See
// https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model for details
// about this model. The parameters are num_nodes, the number of nodes in the
// resulting graph, num_initial_nodes, the number of initially connected nodes
// to create, and edges_per_node, the number of edges to sample per node.
// Callers should ensure that:
//   [1 <= num_initial_nodes <= num_nodes] and
//   [1 <= edges_per_node <= num_initial_nodes]
// The resulting graphs are unweighted.
//
// The graph is constructed by building an initially connected graph (a cycle)
// on num_initial_nodes nodes so the initial degree of each initial node is 2.
// Then, for each remaining node to add, the algorithm samples edges_per_node
// distinct edges per node by sampling an edge to node i with probability d(i) /
// total_degree, where d(i) is the current degree of node i and total_degree is
// the current total degree.
absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> UnweightedBarabasiAlbert(
    size_t num_nodes, size_t num_initial_nodes, size_t edges_per_node);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_BARABASI_ALBERT_H_
