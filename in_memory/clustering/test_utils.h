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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_TEST_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_TEST_UTILS_H_

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

using NodeId = InMemoryClusterer::NodeId;

// Adds a clique of the given size to an undirected graph. The edges all have
// weight 1.0.
void AddUndirectedCliqueToGraph(int32_t size, int32_t initial_n,
                                SimpleUndirectedGraph* graph);

// Adds undirected clique-barbell graph with arbitrary clique sizes. All
// clique edges have weight 1.0. Clique 1 is indexed [0, ..., size1 - 1] and
// clique 2 is indexed [size1, ..., size2 - 1]. An (undirected) edge between
// 0 and size1 with weight 1.0 is created.
void AddUndirectedCliqueBarbellGraph(int32_t size1, int32_t size2,
                                     SimpleUndirectedGraph* graph);

// As above, but also makes and returns the graph.
std::unique_ptr<SimpleUndirectedGraph> MakeUndirectedCliqueBarbellGraph(
    int32_t size1, int32_t size2);

// This graph has a 2-clique and a 4-clique. Nodes with indices 0 and 2 are
// connected (the first node in each clique). Then nodes 1 and 3 are connected
// with weight 2, and the (0, 1) edge weight is changed to 3.
std::unique_ptr<SimpleUndirectedGraph> MakeSmallTestGraph();

// This graph has two 5-node cliques A and B and one 10-node clique C. A final
// "overlap" node is added which connects fully to A and B but is also disjoint
// from C. Illustration ('u' is the overlap node):
//        B
//      /
//    u           C
//      \
//        A
// This is a canonical example of cluster overlap because u is preferential to A
// and B equally (it ignores C).
void AddClusteringWithOverlapNode(SimpleUndirectedGraph* graph);

// Brute-force O(n^2) computation of cluster modularity. O(n^2) is met because
// the function computes the total graph edge weight internally. With C as the
// cluster, Wij as edge {i, j}'s weight, wi & wj as node i & j's
// weighted degree, and gw as the total graph edge weight, the score is defined:
//
//     (sum_{ij in C} Wij - resolution * wi * wj / gw) / |C|^scale_power
//
double ComputeClusterModularity(const absl::flat_hash_set<NodeId>& cluster,
                                const SimpleUndirectedGraph& graph,
                                double resolution = 1.0,
                                double scale_power = 0.0);

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_TEST_UTILS_H_
