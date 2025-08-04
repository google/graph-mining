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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_UTILS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_UTILS_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// Computes the weighted degree of a node in O(degree) steps.
double WeightedDegree(InMemoryClusterer::NodeId node,
                      const SimpleUndirectedGraph& graph);

// Returns vector with jth entry = the weighted degree of node j in the graph,
// in O(E) steps.
std::vector<double> WeightedDegrees(const SimpleUndirectedGraph& graph);

// Intersects two sets of edges and applies a user-defined function f on each
// neighbor k in the intersection. f also takes the weight of the (i, k) edge
// and the weight of the (j, k) edge.
void IntersectEdgeSets(
    const absl::flat_hash_map<InMemoryClusterer::NodeId, double>& neighbors_i,
    const absl::flat_hash_map<InMemoryClusterer::NodeId, double>& neighbors_j,
    const std::function<void(SimpleUndirectedGraph::NodeId, double, double)>&
        f);

// Directs the input graph according to degree (vertices are ranked from lowest
// degree to highest degree, and edges are directed from lower ranks to higher
// ranks). Outputs the edges into directed_input.
std::unique_ptr<SimpleDirectedGraph> DirectGraphByDegree(
    const SimpleUndirectedGraph& input);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_GRAPH_UTILS_H_
