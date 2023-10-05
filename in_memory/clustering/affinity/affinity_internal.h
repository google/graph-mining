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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_AFFINITY_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_AFFINITY_INTERNAL_H_

#include <memory>
#include <vector>

#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// Flattens the clusterings, i.e., assuming the graph was compressed using
// cluster_ids and then clustered obtaining compressed_cluster_ids, computes the
// induced clustering of the initial graph, that is result[i] =
// compressed_cluster_ids[cluster_ids[i]].
// Moreover, if cluster_id[i] = -1, result[i] = -1.
//
// Requires that both vectors have the same size n and
// -1 <= cluster_ids[i] < n (CHECK-fails otherwise).
std::vector<InMemoryClusterer::NodeId> FlattenClustering(
    const std::vector<InMemoryClusterer::NodeId>& cluster_ids,
    const std::vector<InMemoryClusterer::NodeId>& compressed_cluster_ids);

// Compresses the graph using the given clustering and aggregation function,
// which specifies how to combine multiple edge weights into one.
// The resulting graph has the same number of nodes as the input graph. It does
// not have self-loops. A value of -1 in cluster_ids[i] means that a node and
// all its incident edges should be ignored.
//
// Requires that cluster_ids and graph have the same size n and -1 <=
// cluster_ids[i] < n (CHECK-fails otherwise).
absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> CompressGraph(
    const SimpleUndirectedGraph& graph,
    const std::vector<InMemoryClusterer::NodeId>& cluster_ids,
    graph_mining::in_memory::AffinityClustererConfig clusterer_config);

// Performs a single round of nearest-neighbor clustering. First, each node
// marks the highest weight incident edge. Then, we compute connected components
// given by the selected edges. For a graph of size n, returns a vector of size
// n, where 0 <= result[i] < n gives the cluster id of node i. Edges of weight
// smaller than the threshold are ignored. Ties in edge weights are broken using
// graph_mining::BestNeighborFinder. Because of that, one needs to provide a
// function that returns a unique string id of each node.
std::vector<InMemoryClusterer::NodeId> NearestNeighborLinkage(
    const SimpleUndirectedGraph& graph, double weight_threshold,
    std::function<std::string(InMemoryClusterer::NodeId)> get_node_id);

// Converts a node_id -> cluster_id clustering representation to a list of
// clusters. Let n be the size of cluster_ids[i]. The value of cluster_ids[i]
// must satisfy -1 <= cluster_ids[i] < n, where -1 denotes a nonexistent node,
// and other values give the cluster id of node i.
// The resulting clusters are sorted by the cluster_ids and have sorted
// elements.
InMemoryClusterer::Clustering ComputeClusters(
    const std::vector<InMemoryClusterer::NodeId>& cluster_ids);

struct ClusterQualityIndicators {
  // Density is the total weight of all (undirected) edges divided by the number
  // of (unordered) pairs of distinct nodes (or 0.0, if the cluster has less
  // than 2 nodes).
  double density;

  // For the definition of conductance, see
  // https://en.wikipedia.org/wiki/Conductance_(graph)
  // For the cases where conductance is undefined (due to division by something
  // smaller than 1e-6), we assume it to be 1.0.
  double conductance;
};

// Given a vector of NodeIds describing a cluster and a graph, computes the
// corresponding ClusterQualityIndicators.
//
// graph_volume parameter is passed for performance reasons only and should
// contain *twice* the total weight of all edges in the graph, except for
// self-loops, which are counted once.
ClusterQualityIndicators ComputeClusterQualityIndicators(
    const std::vector<InMemoryClusterer::NodeId>& cluster,
    const SimpleUndirectedGraph& graph, double graph_volume);

// Returns true if a given cluster, given as a list of distinct NodeIds is an
// active cluster, wrt the given graph and
// graph_mining::in_memory::AffinityClustererConfig. graph_volume parameter is
// used for performance only. It should be equal to twice the total weight of
// all undirected edges.
bool IsActiveCluster(
    const std::vector<InMemoryClusterer::NodeId>& cluster,
    const SimpleUndirectedGraph& graph,
    const graph_mining::in_memory::AffinityClustererConfig& config,
    double graph_volume);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_AFFINITY_INTERNAL_H_
