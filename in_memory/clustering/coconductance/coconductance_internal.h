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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_INTERNAL_H_

#include <cmath>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining {
namespace in_memory {

// Describes the current clustering state.
struct ClusteringState {
  // Maps each node id into a cluster id.
  std::vector<InMemoryClusterer::NodeId> cluster_ids;
  // Maps each cluster id into the total weight of its nodes.
  // Entries not corresponding to cluster ids are undefined.
  std::vector<double> cluster_weight;
  // Maps each cluster id into the total weight of undirected edges inside the
  // cluster (including self-loops). Entries not corresponding to cluster ids
  // are undefined.
  std::vector<double> cluster_edges;
};

// Returns a state with each node in its own cluster.
ClusteringState InitialState(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph);

// Given the total weight of undirected edges and total node weight, returns
// the cluster objective.
inline double ClusterObjective(double edge_weight, double node_weight,
                               double exponent) {
  if (node_weight < 1e-6) return 0.0;
  return std::pow(2.0 * edge_weight / node_weight, exponent);
}

// Returns the change in objective after moving node to new_cluster.
// edges_to_current_cluster/edges_to_new_cluster should contain the total weight
// of edges between the node and its current/new cluster.
double ObjectiveChangeAfterMove(
    InMemoryClusterer::NodeId node, InMemoryClusterer::NodeId new_cluster,
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    const ClusteringState& state, double edges_to_current_cluster,
    double edges_to_new_cluster, double exponent);

// Updates the state after moving the node to new_cluster.
void MoveNodeAndUpdateState(
    ClusteringState& state, InMemoryClusterer::NodeId node,
    InMemoryClusterer::NodeId new_cluster,
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    double edges_to_current_cluster, double edges_to_new_cluster);

// Constant-approximate algorithm for optimizing co-conductance from
// go/coconductance-paper. Repeats the algorithm num_repetitions times and then
// returns the best result seen so far.
// Requires that each node has weight equal to its weighted degree.
std::vector<InMemoryClusterer::NodeId> ConstantApproximateCoconductance(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    int num_repetitions);

// Functions below are exposed in the header for testing only.
// Computes the objective for p = 1. Requires that each node has weight equal to
// its weighted degree.
double CoconductanceObjective(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    const std::vector<InMemoryClusterer::NodeId>& cluster_ids);

// A single repetition of the constant-approximate algorithm. Requires that each
// node has weight equal to its weighted degree.
std::vector<InMemoryClusterer::NodeId> ConstantApproximateCoconductance(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    absl::BitGenRef rng);

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COCONDUCTANCE_COCONDUCTANCE_INTERNAL_H_
