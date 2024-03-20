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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_INTERNAL_H_

#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Represents the min merge similarities of nodes in local node id space.
class SubgraphMinMergeSimilarity {
  using NodeId = graph_mining::in_memory::NodeId;

 public:
  SubgraphMinMergeSimilarity(
      const std::vector<NodeId>& node_map,
      const absl::flat_hash_map<NodeId, double>& min_merge_similarities)
      : node_map_(node_map), min_merge_similarities_(min_merge_similarities) {}

  // Returns `min_merge_similarities_[node_map_[i]]`. Returns error if
  // `node_map_[i]` is not in `min_merge_similarities_`. `i` is in local node id
  // space.
  absl::StatusOr<double> operator()(NodeId i) const {
    const auto node_id = node_map_[i];
    auto sim = min_merge_similarities_.find(node_id);
    if (sim == min_merge_similarities_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Node not in min_merge_similarities, id = ", node_id));
    } else {
      return sim->second;
    }
  }

  const std::vector<NodeId>& NodeMap() const { return node_map_; }

 private:
  // Maps from local node id to global node id.
  const std::vector<NodeId>& node_map_;
  // Min merge similarities in global id space.
  const absl::flat_hash_map<NodeId, double>& min_merge_similarities_;
};

// Update the `partition_map` of `new_nodes`, the neighbors of `new_nodes`, and
// `neighbors_deleted` in `graph`. `neighbors_deleted` represents the neighbors
// of deleted nodes. Red nodes are in their own partitions, and a blue node's
// partition is its red neighbor with highest edge similarity. If a blue node
// does not have a neighbor, its partition is itself. The color of nodes are
// determined by `color`. Return the tuple (node id, partition id before,
// partition id after) for `new_nodes`, the neighbors of `new_nodes`, and
// `neighbors_deleted`. For `new_nodes`, partition_before is -1. Returns error
// status if any node is not in `graph`.
absl::StatusOr<std::vector<
    std::tuple<graph_mining::in_memory::NodeId, graph_mining::in_memory::NodeId,
               graph_mining::in_memory::NodeId>>>
UpdatePartitions(
    const std::vector<
        graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList>&
        new_nodes,
    const absl::flat_hash_set<graph_mining::in_memory::NodeId>&
        neighbors_deleted,
    const DynamicClusteredGraph& graph, const DynamicHacNodeColorBase& color,
    absl::flat_hash_map<graph_mining::in_memory::NodeId,
                        graph_mining::in_memory::NodeId>& partition_map);

// Return the dirty partitions based on `nodes_changed`. For [u, p_before,
// p_after] in `nodes_changed`, dirty partition set contains all p_after, and
// red p_before that is in `graph`. If a partition is a singleton blue partition
// before and after the update, it is also returned. We return singleton blue
// partitions so that it's easier to add nodes back when they turn from having
// no heavy edges to having heavy edges.
absl::StatusOr<absl::flat_hash_set<graph_mining::in_memory::NodeId>>
DirtyPartitions(absl::Span<const std::tuple<graph_mining::in_memory::NodeId,
                                            graph_mining::in_memory::NodeId,
                                            graph_mining::in_memory::NodeId>>
                    nodes_changed,
                const DynamicClusteredGraph& graph,
                const DynamicHacNodeColorBase& color);

// Run HAC with `epsilon` on `partition_graph`. `partition_graph` is a subgraph
// of some original graph G. `min_merge_similarities_partition_map` maps
// each node id to its min merge similarity (let's denote it by M(i) for short),
// which is the minimum merge similarity of node i. If node i is a single node,
// this is infinity. If Node i is obtained by merge(u,v), its
// min_merge_similarity is min(M(u), M(v), w(u,v)). Returns the merges done in
// the HAC in terms of the ids of `partition_graph` and the dendrogram returned
// by HAC. Returns an error status if `min_merge_similarities_partition_map(i)`
// returns an error.
absl::StatusOr<SubgraphHacResults> RunSubgraphHac(
    std::unique_ptr<graph_mining::in_memory::SimpleUndirectedGraph>&
        partition_graph,
    const SubgraphMinMergeSimilarity& min_merge_similarities_partition_map,
    double epsilon);

// Return min_merge_similarities (in local cluster id) of nodes in `merges`. The
// returned vector has size (2*partition_num_nodes-1). If there are less than
// partition_num_nodes-1 number of merges, the rest of the local cluster ids
// have inf as min_merge_similarities. `merges` is in local cluster id.
// `min_merge_similarities_partition_map` maps each local node id to its min
// merge similarity (let's denote it by M(i) for short), which is the minimum
// merge similarity of node i. If node i is a single node, this is infinity.
// Only nodes with the same partition in `partition_map` can be merged. If Node
// i is obtained by merge(u,v), its min_merge_similarity is min(M(u), M(v),
// w(u,v)). Returns error status if any node i in 0...`partition_num_nodes-1` is
// not valid for `min_merge_similarity_partition_map(i)`.
absl::StatusOr<std::vector<double>> LocalMinMergeSimilarities(
    absl::Span<const std::tuple<graph_mining::in_memory::NodeId,
                                graph_mining::in_memory::NodeId, double>>
        merges,
    const SubgraphMinMergeSimilarity& min_merge_similarities_partition_map,
    std::size_t partition_num_nodes);

// Return a map from leaves nodes to their root nodes in `dendrogram`.
std::vector<graph_mining::in_memory::NodeId> LeafToRootId(
    const graph_mining::in_memory::Dendrogram& dendrogram);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_HAC_INTERNAL_H_
