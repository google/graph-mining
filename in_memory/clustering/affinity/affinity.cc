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

#include "in_memory/clustering/affinity/affinity.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/affinity/affinity_internal.h"
#include "in_memory/clustering/affinity/weight_threshold.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using research_graph::in_memory::AffinityClustererConfig;
using research_graph::in_memory::ClustererConfig;

namespace {

using NodeId = AffinityClusterer::NodeId;

void DeactivateNodesInFinishedClusters(
    const AffinityClusterer::Clustering& clusters,
    const SimpleUndirectedGraph& graph, const AffinityClustererConfig& config,
    std::vector<bool>* active_nodes) {
  AffinityClusterer::Clustering clustering;

  double graph_volume = 0;
  for (NodeId i = 0; i < graph.NumNodes(); ++i)
    for (const auto& neighbor : graph.Neighbors(i))
      graph_volume += neighbor.second;

  for (const auto& cluster : clusters) {
    if (!IsActiveCluster(cluster, graph, config, graph_volume)) {
      for (auto node : cluster) (*active_nodes)[node] = false;
    }
  }
}

}  // namespace

absl::StatusOr<std::vector<AffinityClusterer::Clustering>>
AffinityClusterer::HierarchicalFlatCluster(
    const ClustererConfig& config) const {
  const AffinityClustererConfig& affinity_config =
      config.affinity_clusterer_config();

  const int n = graph_.NumNodes();
  std::vector<NodeId> cluster_ids(n);
  for (NodeId i = 0; i < n; ++i) cluster_ids[i] = i;

  std::vector<bool> active_nodes(n, true);
  std::vector<AffinityClusterer::Clustering> result;

  for (int i = 0; i < affinity_config.num_iterations(); ++i) {
    auto weight_threshold = AffinityWeightThreshold(affinity_config, i);
    if (!weight_threshold.ok()) return weight_threshold.status();

    auto cluster_ids_for_compression = cluster_ids;
    for (size_t node = 0; node < cluster_ids.size(); ++node) {
      if (!active_nodes[node]) cluster_ids_for_compression[node] = -1;
    }

    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSIGN_OR_RETURN(compressed_graph,
                     (i == 0)
                         ? std::unique_ptr<SimpleUndirectedGraph>(nullptr)
                         : CompressGraph(graph_, cluster_ids_for_compression,
                                         affinity_config));
    // Clear cluster_ids_for_compression (no longer in use).
    std::vector<NodeId>().swap(cluster_ids_for_compression);

    std::vector<NodeId> compressed_cluster_ids = NearestNeighborLinkage(
        !compressed_graph ? graph_ : *compressed_graph, *weight_threshold,
        [this](NodeId node_id) { return StringId(node_id); });
    cluster_ids = FlattenClustering(cluster_ids, compressed_cluster_ids);

    auto current_clusters = ComputeClusters(cluster_ids);
    DeactivateNodesInFinishedClusters(current_clusters, graph_, affinity_config,
                                      &active_nodes);

    result.emplace_back(std::move(current_clusters));
  }

  return result;
}

absl::StatusOr<AffinityClusterer::Clustering> AffinityClusterer::Cluster(
    const ClustererConfig& config) const {
  std::vector<AffinityClusterer::Clustering> clustering_hierarchy;
  ASSIGN_OR_RETURN(clustering_hierarchy, HierarchicalFlatCluster(config));
  if (clustering_hierarchy.empty()) {
    AffinityClusterer::Clustering trivial_clustering;
    for (int i = 0; i < graph_.NumNodes(); ++i) {
      trivial_clustering.push_back(std::vector<NodeId>{i});
    }
    return trivial_clustering;
  } else {
    return clustering_hierarchy.back();
  }
}

}  // namespace graph_mining::in_memory
