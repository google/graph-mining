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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLUSTERING_UTILS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLUSTERING_UTILS_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Transforms a clustering to a canonical form: sorts the NodeIds in
// non-decreasing order within each cluster and then sorts the clusters
// lexicographically.
Clustering CanonicalizeClustering(Clustering clustering);

// Same as above, but uses the provided vector for comparing the NodeIds,
// i.e., instead of comparing ids x and y, we compare sort_keys[x] and
// sort_keys[y]. sort_keys must have a corresponding entry for each NodeId
// used in the clustering.
Clustering CanonicalizeClustering(Clustering clustering,
                                  const std::vector<std::string>& sort_keys);

// Given an expected number of nodes and a clustering, produce a mapping from
// NodeId to ClusterId. The mapping is structured as a vector since it is
// assumed that the NodeIds are contiguous non-negative integers.
std::vector<NodeId> CreateNodeIdToClusterIdMap(int num_nodes,
                                               const Clustering& clustering);

// Given a clustering represented as a node_id -> cluster_id mapping, returns
// the corresponding Clustering. The input is any iterable
// container of pairs, where each element is a (node_id, cluster_id) pair.
// The keys in the container must be of type NodeId.
// The values in the container can have any type which can be used as a
// flat_hash_map key.
// The order of elements of each cluster and the order of first elements of all
// clusters are both consistent with the order of iteration on the container.
template <typename Container>
Clustering ClusterIdsToClustering(const Container& cluster_ids) {
  static_assert(
      std::is_same<decltype(cluster_ids.begin()->first), NodeId>::value,
      "The keys must be of type NodeId");
  absl::flat_hash_map<decltype(cluster_ids.begin()->second), int>
      cluster_id_to_cluster_index;
  Clustering result;
  for (const auto& [node_id, cluster_id] : cluster_ids) {
    // Looks up the value associated with cluster_id or inserts a new one.
    auto [_, cluster_index] = *(
        cluster_id_to_cluster_index.insert({cluster_id, result.size()}).first);
    if (cluster_index == result.size())
      result.emplace_back(std::vector<NodeId>{node_id});
    else
      result[cluster_index].emplace_back(node_id);
  }
  return result;
}

// Given a clustering represented by a vector, where cluster_ids[i] gives the
// cluster id of node i, return the corresponding Clustering.
// The id of each cluster has to be between 0 and cluster_ids.size()-1.
// The result contains nonempty clusters only, in the order of their ids.
// Elements in each cluster are given in increasing order of ids.
absl::StatusOr<Clustering> ClusterIdSequenceToClustering(
    absl::Span<const NodeId> cluster_ids);

// Map the node ids to a clustering by using the provided integer -> string map.
absl::StatusOr<std::vector<absl::flat_hash_set<std::string>>>
MapClusteringToStringIds(const Clustering& clustering,
                         const std::vector<std::string>& node_id_map);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLUSTERING_UTILS_H_
