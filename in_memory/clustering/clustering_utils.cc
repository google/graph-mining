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

#include "in_memory/clustering/clustering_utils.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/slice.h"

namespace graph_mining::in_memory {

Clustering CanonicalizeClustering(Clustering clustering) {
  for (auto& cluster : clustering) {
    std::sort(cluster.begin(), cluster.end());
  }
  std::sort(clustering.begin(), clustering.end());
  return clustering;
}

Clustering CanonicalizeClustering(Clustering clustering,
                                  const std::vector<std::string>& sort_keys) {
  auto compare_by_sort_keys = [&sort_keys](NodeId a, NodeId b) {
    ABSL_CHECK_GE(a, 0);
    ABSL_CHECK_GE(b, 0);
    ABSL_CHECK_LT(a, sort_keys.size());
    ABSL_CHECK_LT(b, sort_keys.size());
    return sort_keys[a] < sort_keys[b];
  };

  for (auto& cluster : clustering) {
    std::sort(cluster.begin(), cluster.end(), compare_by_sort_keys);
  }

  std::sort(clustering.begin(), clustering.end(),
            [compare_by_sort_keys](const std::vector<NodeId>& a,
                                   const std::vector<NodeId>& b) {
              return std::lexicographical_compare(
                  a.begin(), a.end(), b.begin(), b.end(), compare_by_sort_keys);
            });
  return clustering;
}

std::vector<NodeId> CreateNodeIdToClusterIdMap(NodeId num_nodes,
                                               const Clustering& clustering) {
  std::vector<NodeId> node_id_to_cluster_id(num_nodes);
  for (NodeId i = 0; i < clustering.size(); i++) {
    for (NodeId node_id : clustering[i]) {
      node_id_to_cluster_id[node_id] = i;
    }
  }

  return node_id_to_cluster_id;
}

std::vector<NodeId> CreateNodeIdToClusterIdMapParallel(
    const Clustering& clustering) {
  auto cluster_sizes = parlay::delayed_seq<std::size_t>(
      clustering.size(), [&](size_t i) { return clustering[i].size(); });
  std::size_t num_nodes = parlay::reduce(parlay::make_slice(cluster_sizes));
  std::vector<NodeId> node_id_to_cluster_id(num_nodes);
  parlay::parallel_for(0, clustering.size(), [&](std::size_t i) {
    parlay::parallel_for(0, clustering[i].size(), [&](std::size_t j) {
      NodeId node_id = clustering[i][j];
      ABSL_CHECK_GE(node_id, 0)
          << "Node IDs must be non-negative; found " << node_id;
      ABSL_CHECK_LT(node_id, num_nodes)
          << "Node IDs must be smaller than the number of nodes (" << num_nodes
          << "); found " << node_id;
      node_id_to_cluster_id[node_id] = i;
    });
  });
  return node_id_to_cluster_id;
}

absl::StatusOr<std::vector<absl::flat_hash_set<std::string>>>
MapClusteringToStringIds(const Clustering& clustering,
                         const std::vector<std::string>& node_id_map) {
  auto integer_clustering = CanonicalizeClustering(clustering);
  std::vector<absl::flat_hash_set<std::string>> result;
  for (const std::vector<NodeId>& integer_cluster : integer_clustering) {
    absl::flat_hash_set<std::string>& cluster = result.emplace_back();
    for (NodeId id : integer_cluster) {
      cluster.insert(node_id_map[id]);
    }
  }
  return result;
}

absl::StatusOr<Clustering> ClusterIdSequenceToClustering(
    absl::Span<const NodeId> cluster_ids) {
  NodeId n = cluster_ids.size();
  std::vector<std::vector<NodeId>> clustering(n);

  for (NodeId i = 0; i < n; ++i) {
    if (cluster_ids[i] < 0 || cluster_ids[i] >= n)
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid cluster id=", cluster_ids[i]));
    clustering[cluster_ids[i]].push_back(i);
  }
  clustering.erase(std::remove_if(clustering.begin(), clustering.end(),
                                  [](const std::vector<NodeId>& cluster) {
                                    return cluster.empty();
                                  }),
                   clustering.end());
  return clustering;
}

}  // namespace graph_mining::in_memory
