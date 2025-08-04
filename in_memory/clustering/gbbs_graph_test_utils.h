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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_TEST_UTILS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_TEST_UTILS_H_

#include <tuple>
#include <vector>

#include "gbbs/graph.h"
#include "gbbs/macros.h"

namespace graph_mining::in_memory {

// Tests that the given graph has exactly num_vertices vertices, and consists
// exactly of the edges given in neighbors, where neighbors is a vector of the
// adjacency lists of the graph. Note that the adjacency lists do not have
// to be sorted.
void CheckGbbsGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph,
    std::size_t num_vertices,
    const std::vector<std::vector<std::tuple<gbbs::uintE, float>>>& neighbors);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GBBS_GRAPH_TEST_UTILS_H_
