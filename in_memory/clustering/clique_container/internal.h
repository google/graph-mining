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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_INTERNAL_H_

#include <memory>
#include <vector>

#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Degeneracy ordering of a given graph as a vector. The length of the returned
// vector is graph.NumNodes(). The ith element of the vector is the ith node in
// the degeneracy ordering ordering.
// The degeneracy ordering is obtained by repeatedly removing the minimum-degree
// node from the graph until the graph is empty. In case of ties, the node with
// the smallest id is removed.
// Works for graphs of less than 2^30 nodes.
std::vector<NodeId> DegeneracyOrdering(
    std::unique_ptr<SimpleUndirectedGraph> graph);

// Given a graph and an ordering of its nodes, returns a directed graph with the
// the edges directed from the node that comes earlier in the ordering to the
// node that comes later. The returned graph has exactly one edge for each edge
// in the input graph, except that self-loops are removed. `ordering` must be a
// permutation of [0, graph.NumNodes()). Otherwise, nullptr is returned.
std::unique_ptr<SimpleDirectedGraph> DirectGraph(
    const SimpleUndirectedGraph& graph, const std::vector<NodeId>& ordering);

// Given a collection of clusters, filters out the ones that are *strictly*
// contained in other clusters. Here, strictly means that a cluster is only
// filtered out if it is a *proper* subset of another cluster. Thus, if cluster
// appears multiple times, either all copies of the cluster are returned or all
// are filtered out. All empty clusters are always filtered out.
//
// Returns the clusters that remain in arbitrary order. The elements of each
// cluster in the result are sorted in ascending order. Returns an empty vector
// if any node id is negative.
//
// Uses additional space proportional to the maximum node id in any cluster.
// The asymptotic running time is upper bounded by the time it takes to sort all
// clusters and call std::includes on all pairs of clusters that have a
// nonempty intersection.
std::vector<ClusterContents> SortClustersAndRemoveContained(
    std::vector<ClusterContents> clusters);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_INTERNAL_H_
