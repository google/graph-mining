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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_QUICK_CLUSTER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_QUICK_CLUSTER_H_
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/random/random.h"
#include "absl/random/bit_gen_ref.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Generate a random permutation of {0, 1, ..., num_nodes - 1}.
std::vector<NodeId> RandomVisitOrder(int num_nodes, absl::BitGenRef rand);

// Runs the Ailon Charikar Newman approximation algorithm for unweighted
// correlation clustering. For a description of the algorithm see the paper
// "Aggregating Inconsistent Information: Ranking and Clustering" in JACM or
// http://dimacs.rutgers.edu/~alantha/papers2/aggregating_journal.pdf.
// Parameter visit_order should be a permutation of the nodes.
// The first overload (with explicit `visit_order`) is mostly for testing
// purposes.
std::vector<std::vector<NodeId>> QuickCluster(
    const SimpleUndirectedGraph& graph,
    const CorrelationClustererConfig& config,
    const std::vector<NodeId>& visit_order);
inline std::vector<std::vector<NodeId>> QuickCluster(
    const SimpleUndirectedGraph& graph,
    const CorrelationClustererConfig& config,
    absl::BitGenRef rand) {
  return QuickCluster(graph, config, RandomVisitOrder(graph.NumNodes(), rand));
}

// Generalized QuickCluster algorithm that accepts a callback argument,
// `cluster_together`. The callback can be used to implement other rounding
// algorithms, e.g. the 2.06 approximation
// (https://arxiv.org/pdf/1412.0681.pdf). The callback returns true if `other`
// should be included in the same cluster as `center`. The callback is called
// with a center and all nodes that are neighbors of the center in the graph
// that have yet to be included in a cluster.
// The first overload (with explicit `visit_order`) is mostly for testing
// purposes.
std::vector<std::vector<NodeId>> GeneralizedQuickCluster(
    const SimpleUndirectedGraph& graph,
    absl::FunctionRef<bool(NodeId center, NodeId other)> cluster_together,
    const std::vector<NodeId>& visit_order);
inline std::vector<std::vector<NodeId>> GeneralizedQuickCluster(
    const SimpleUndirectedGraph& graph,
    absl::FunctionRef<bool(NodeId center, NodeId other)> cluster_together,
    absl::BitGenRef rand) {
  return GeneralizedQuickCluster(graph, cluster_together,
                                 RandomVisitOrder(graph.NumNodes(), rand));
}
}  // namespace graph_mining::in_memory
#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CORRELATION_QUICK_CLUSTER_H_
