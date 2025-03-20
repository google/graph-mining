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

#include "in_memory/clustering/clique_container/internal.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/types.h"
#include "utils/container/fixed_size_priority_queue.h"

namespace graph_mining {
namespace in_memory {

std::vector<NodeId> DegeneracyOrdering(
    std::unique_ptr<SimpleUndirectedGraph> graph) {
  std::vector<NodeId> degeneracy_ordering;
  degeneracy_ordering.reserve(graph->NumNodes());

  // For each node, we insert it into the queue with priority equal to the
  // negative of its out-degree. This ensures that the Top() method of the
  // queue returns the node with the smallest out-degree.
  // We use a power-of-two sized queue to ensure that ties are broken by
  // returning the node with the smallest id (see comment on
  // FixedSizePriorityQueue::Top).
  FixedSizePriorityQueue<NodeId, NodeId, std::less<NodeId>> queue(
      std::bit_ceil(static_cast<uint32_t>(graph->NumNodes())));
  for (NodeId node_id = 0; node_id < graph->NumNodes(); ++node_id) {
    queue.InsertOrUpdate(node_id, graph->Neighbors(node_id).size());
  }
  for (int i = 0; i < graph->NumNodes(); ++i) {
    auto node_id = queue.Top();
    queue.Remove(node_id);
    degeneracy_ordering.push_back(node_id);
    auto neighbors = graph->Neighbors(node_id);
    for (const auto& [neighbor_id, _] : neighbors) {
      queue.InsertOrUpdate(neighbor_id, queue.Priority(neighbor_id) - 1);
      ABSL_CHECK_OK(graph->RemoveEdge(node_id, neighbor_id));
    }
  }
  return degeneracy_ordering;
}

std::unique_ptr<SimpleDirectedGraph> DirectGraph(
    const SimpleUndirectedGraph& graph, const std::vector<NodeId>& ordering) {
  if (ordering.size() != graph.NumNodes()) return nullptr;
  auto directed_graph = std::make_unique<SimpleDirectedGraph>();
  directed_graph->SetNumNodes(graph.NumNodes());
  std::vector<NodeId> position(graph.NumNodes(), -1);
  for (int32_t i = 0; i < ordering.size(); ++i) {
    if (ordering[i] >= graph.NumNodes() || ordering[i] < 0 ||
        position[ordering[i]] != -1) {
      return nullptr;
    }
    position[ordering[i]] = i;
  }

  for (NodeId node_id = 0; node_id < graph.NumNodes(); ++node_id) {
    for (const auto& [neighbor_id, weight] : graph.Neighbors(node_id)) {
      if (position[node_id] < position[neighbor_id]) {
        ABSL_CHECK_OK(directed_graph->AddEdge(node_id, neighbor_id, weight));
      }
    }
  }
  return directed_graph;
}

namespace {

// returns true if cluster represented by `small` is strictly contained in the
// cluster represented by `big`. Both clusters must be sorted in ascending
// order.
bool IsStrictlyContained(const ClusterContents& small,
                         const ClusterContents& big) {
  return small.size() < big.size() &&
         std::includes(big.begin(), big.end(), small.begin(), small.end());
}

}  // namespace

std::vector<ClusterContents> SortClustersAndRemoveContained(
    std::vector<ClusterContents> clusters) {
  NodeId min_node_id = 0;
  NodeId max_node_id = -1;
  for (const auto& cluster : clusters) {
    if (cluster.empty()) continue;

    auto [cluster_min, cluster_max] =
        std::minmax_element(cluster.begin(), cluster.end());
    min_node_id = std::min(min_node_id, *cluster_min);
    max_node_id = std::max(max_node_id, *cluster_max);
  }
  ABSL_CHECK_GE(min_node_id, 0);

  // For each node v, stores a vector of pointers to *some* clusters containing
  // v. Each (non-empty) cluster is stored in exactly one of these vectors.
  std::vector<std::vector<const ClusterContents*>> cluster_ptrs(max_node_id +
                                                                1);
  for (auto& cluster : clusters) {
    if (cluster.empty()) continue;
    cluster_ptrs[cluster[0]].push_back(&cluster);
    std::sort(cluster.begin(), cluster.end());
  }

  for (const auto& cluster : clusters) {
    for (auto element : cluster) {
      std::erase_if(cluster_ptrs[element],
                    [&cluster](const ClusterContents* cluster_ptr) {
                      return IsStrictlyContained(*cluster_ptr, cluster);
                    });
    }
  }
  std::vector<ClusterContents> result;
  for (const auto& ptrs_per_index : cluster_ptrs) {
    for (auto cluster_ptr : ptrs_per_index) {
      result.push_back(std::move(*cluster_ptr));
    }
  }
  return result;
}

}  // namespace in_memory
}  // namespace graph_mining
