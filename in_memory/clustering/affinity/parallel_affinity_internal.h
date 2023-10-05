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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_INTERNAL_H_

#include <array>
#include <tuple>

#include "absl/status/statusor.h"
#include "parlay/sequence.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/parallel/parallel_graph_utils.h"

namespace graph_mining::in_memory {

// Compute finished clusters given cluster ids and a predicate function
// is_finished. Returns a cluster id v if and only if is_finished(v) == true.
InMemoryClusterer::Clustering ComputeClusters(
    const std::vector<gbbs::uintE>& cluster_ids,
    std::function<bool(gbbs::uintE)> is_finished);

namespace internal {

// Parameter type for size constraint computation.
struct SizeConstraintConfig {
  const graph_mining::in_memory::AffinityClustererConfig::SizeConstraint&
      size_constraint;
  const std::vector<double>& node_weights;
};

// Edge struct for size constraint computation.
struct Edge {
  gbbs::uintE neighbor_id;
  float weight;
};

// Given a graph with exactly one outgoing edge per node, splits the connected
// components of the graph into clusters of at most the specified maximum size.
// If `size_constraint_config.prefer_min_cluster_size` is set, then the function
// does not merge two clusters if both are above the min cluster size threshold.
//
// Input
// - size_constraint_config: Size constraint configuration and original node
// weight vector.
// - cluster_ids: the id of the connected component of each node in the graph.
// - best_neighbors: collection of best neighbors for each node that leads to
// the connected components in `cluster_ids`. In each connected component in the
// input graph, edges are processed according to the edge weights (in descending
// order).
//
// Return
// - Cluster ids conforming to the size constraint configuration.
parlay::sequence<gbbs::uintE> EnforceMaxClusterSize(
    const SizeConstraintConfig& size_constraint_config,
    absl::Span<const gbbs::uintE> cluster_ids,
    parlay::sequence<Edge>&& best_neighbors);

}  // namespace internal

// Performs a single round of nearest-neighbor clustering. First, each node
// marks the highest weight incident edge. Then, we compute connected components
// given by the selected edges. For a graph of size n, returns a sequence of
// size n, where 0 <= result[i] < n gives the cluster id of node i. Edges of
// weight smaller than the threshold are ignored. Ties in edge weights are
// broken using edge endpoint ids.
//
// If `size_constraint_config` is provided, then size constraints are taken into
// consideration (1) when selecting best neighbor edges and (2) when computing
// connected components. See the configuration proto
// AffinityClustererConfig::SizeConstraint (http://shortn/_Y0QaDS1dRB) for
// details. If SizeConstraintConfig.node_weights is empty then default node
// weight of 1 will be used for each node.
absl::StatusOr<std::vector<gbbs::uintE>> NearestNeighborLinkage(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    float weight_threshold,
    std::optional<internal::SizeConstraintConfig> size_constraint_config =
        std::nullopt);

// Compute a compressed graph where vertices are given by cluster ids, and edges
// are aggregated according to affinity_config. A cluster id of UINT_E_MAX
// means that the corresponding vertex has already been clustered into
// a final cluster, by virtue of end conditions given by affinity_config.
// TODO: Switch original_graph to a const reference (which requires
// GBBS to support const get_vertex() calls)
absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<double>& original_node_weights,
    const std::vector<gbbs::uintE>& cluster_ids,
    const graph_mining::in_memory::AffinityClustererConfig& affinity_config);

// Determine which clusters, as given by cluster_ids, are "finished", where
// "finished" is defined by AffinityClustererConfig (e.g., a cluster of
// sufficient density or conductance). These clusters are aggregated and
// returned, and the cluster ids and compressed cluster ids for the
// corresponding vertices are updated to be invalid.
InMemoryClusterer::Clustering FindFinishedClusters(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    const graph_mining::in_memory::AffinityClustererConfig& affinity_config,
    std::vector<gbbs::uintE>& cluster_ids,
    std::vector<gbbs::uintE>& compressed_cluster_ids);

namespace internal {

struct ClusterStats {
  // Constructor for initialization within parallel_for_bc macro
  ClusterStats(float _density, float _conductance)
      : density(_density), conductance(_conductance) {}
  float density;
  float conductance;
};

std::vector<ClusterStats> ComputeFinishedClusterStats(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    const std::vector<gbbs::uintE>& cluster_ids,
    gbbs::uintE num_compressed_vertices);

}  // namespace internal

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_PARALLEL_AFFINITY_INTERNAL_H_
