// Copyright 2024 Google LLC
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

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_graph.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_node.h"
#include "in_memory/clustering/types.h"
#include "utils/container/fixed_size_priority_queue.h"
#include "utils/math.h"

namespace graph_mining::in_memory {

namespace {
using NodeId = ApproximateSubgraphHacGraph::NodeId;
}  // namespace

ApproximateSubgraphHacGraph::ApproximateSubgraphHacGraph(
    const SimpleUndirectedGraph& graph, NodeId num_nodes, double epsilon,
    double alpha, std::vector<bool> is_active,
    const std::vector<double>& min_merge_similarities)
    : is_active_(std::move(is_active)),
      min_merge_similarities_(min_merge_similarities),
      node_pq_(
          FixedSizePriorityQueue</*PriorityType=*/double, /*IndexType=*/NodeId>(
              num_nodes)),
      one_plus_alpha_(1 + alpha),
      one_plus_eps_(1 + epsilon) {
  nodes_.reserve(graph.NumNodes());
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    nodes_.push_back(
        ApproximateSubgraphHacNode(graph.NodeWeight(i), one_plus_alpha_));
  }

  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    if (is_active_[i]) {
      for (auto [neighbor, weight] : graph.Neighbors(i)) {
        // Ensure no self-loops.
        ABSL_CHECK_NE(i, neighbor);
        nodes_[i].InsertEdge(neighbor, graph.NodeWeight(neighbor), weight);
      }
    }
  }

  // Initialize goodness.
  for (NodeId node_v = 0; node_v < graph.NumNodes(); ++node_v) {
    if (is_active_[node_v]) {
      auto best_v = nodes_[node_v].ApproximateBestWeightAndId().first;
      for (auto [node_w, _] : graph.Neighbors(node_v)) {
        // Only process (active, active) edges, and only in one direction.
        if (!is_active_[node_w] || node_v > node_w) {
          continue;
        }

        double goodness_vw = Goodness(node_v, node_w);
        auto best_w = nodes_[node_w].ApproximateBestWeightAndId().first;

        // Assign the edge to whichever of node_v, node_w have the higher best
        // incident weight value.
        if (best_v < best_w) {
          nodes_[node_w].AssignEdge(node_v, goodness_vw);
        } else {
          nodes_[node_v].AssignEdge(node_w, goodness_vw);
        }
      }
    }
  }

  // Finally, initialize node_pq_.
  for (NodeId i = 0; i < graph.NumNodes(); ++i) {
    if (is_active_[i]) {
      auto get_goodness = [&](NodeId node_u, NodeId node_v) {
        return Goodness(node_u, node_v);
      };
      auto [goodness, _] =
          nodes_[i].GetGoodEdge(i, one_plus_eps_, get_goodness);
      if (goodness != kDefaultGoodness) {
        node_pq_.InsertOrUpdate(i, -1 * goodness);
      }
    }
  }
}

size_t ApproximateSubgraphHacGraph::NumNodes() const { return nodes_.size(); }

absl::StatusOr<NodeId> ApproximateSubgraphHacGraph::Merge(
    Dendrogram* dendrogram, std::vector<NodeId>* to_cluster_id,
    std::vector<double>* min_merge_similarities, NodeId node_a, NodeId node_b) {
  NodeId node_from = node_a;
  NodeId node_to = node_b;

  // Update cluster_sizes
  if (!is_active_[node_from]) {
    return absl::FailedPreconditionError(
        absl::StrFormat("NodeFrom (%d) was not active", node_from));
  }
  if (!is_active_[node_to]) {
    return absl::FailedPreconditionError(
        absl::StrFormat("NodeTo (%d) was not active", node_to));
  }
  NodeId node_a_size = nodes_[node_from].Neighbors().size();
  NodeId node_b_size = nodes_[node_to].Neighbors().size();
  // Merge from smaller size to larger size. If nodes have the same size,
  // merge from smaller id to larger.
  if (node_a_size > node_b_size ||
      (node_a_size == node_b_size && node_from > node_to)) {
    std::swap(node_from, node_to);
  }

  // Merge the neighborhood data.
  auto [edges_to_reassign, nodes_to_update_in_pq] = nodes_[node_from].Merge(
      node_from, node_to, absl::MakeSpan(nodes_), is_active_);

  // Remove node_from from the PQ.
  ABSL_DCHECK(!is_active_[node_from]);
  node_pq_.Remove(node_from);

  // node_from is no longer active. node_to now has all of node_from's edges.
  ABSL_DCHECK(is_active_[node_to]);
  // Fix node_to's weight (goodness) in the node PQ.
  nodes_to_update_in_pq.insert(node_to);
  // No edges left that are assigned to node_from.
  ABSL_DCHECK_EQ(nodes_[node_from].NumAssignedEdges(), 0);

  // Finished merge of neighbor info; broadcast from the merged vertex if
  // its cluster size or best values changed sufficiently.
  nodes_[node_to].MaybeBroadcastClusterSize(node_to, absl::MakeSpan(nodes_),
                                            is_active_);

  // Broadcast and reassign edges if the best weight value changed enough.
  auto get_goodness = [&](NodeId node_u, NodeId node_v) {
    return Goodness(node_u, node_v);
  };
  nodes_[node_to].MaybeReassignEdges(node_to, absl::MakeSpan(nodes_),
                                     is_active_, &nodes_to_update_in_pq,
                                     get_goodness);

  // Reassign edges.
  ReassignChangedEdges(std::move(edges_to_reassign), nodes_to_update_in_pq);

  // Reassign nodes / update PQs.
  for (auto node_id : nodes_to_update_in_pq) {
    UpdateNodePQ(node_id);
  }
  nodes_to_update_in_pq.clear();

  return node_to;
}

std::tuple<NodeId, NodeId, double> ApproximateSubgraphHacGraph::GetGoodEdge() {
  // Scan node_pq_, and get the best edge incident to the top-goodness node.
  while (!node_pq_.Empty()) {
    NodeId node_a = node_pq_.Top();
    // Get the current best edge for node_a.
    auto get_goodness = [&](NodeId node_u, NodeId node_v) {
      return Goodness(node_u, node_v);
    };
    auto [goodness_ab, node_b] =
        nodes_[node_a].GetGoodEdge(node_a, one_plus_eps_, get_goodness);

    if (goodness_ab > one_plus_eps_ &&
        !AlmostEquals(goodness_ab, one_plus_eps_)) {
      node_pq_.Remove(node_a);
    } else {
      // We found a (1+epsilon)-good edge. Return a lex-ordered tuple.
      return {std::min(node_a, node_b), std::max(node_a, node_b), goodness_ab};
    }
  }
  return {std::numeric_limits<NodeId>::max(),
          std::numeric_limits<NodeId>::max(), kDefaultGoodness};
}

void ApproximateSubgraphHacGraph::ReassignChangedEdges(
    std::vector<std::pair<NodeId, NodeId>> edges_to_reassign,
    absl::flat_hash_set<NodeId>& nodes_to_update_in_pq) {
  for (auto [node_u, node_v] : edges_to_reassign) {
    ABSL_DCHECK(is_active_[node_u]);
    ABSL_DCHECK(is_active_[node_v]);

    auto best_u = nodes_[node_u].ApproximateBestWeightAndId().first;
    auto best_v = nodes_[node_v].ApproximateBestWeightAndId().first;

    auto goodness_uv = Goodness(node_u, node_v);
    if (best_u < best_v) {
      // Reassign the edge to v.
      nodes_[node_v].AssignEdge(node_u, goodness_uv);
      ABSL_DCHECK(nodes_[node_u].IsNeighbor(node_v));
      ABSL_DCHECK_EQ(nodes_[node_u].GetNeighborInfo(node_v).goodness,
                     kDefaultGoodness);
      nodes_to_update_in_pq.insert(node_v);
    } else {
      // Otherwise, keep the edge with u.
      nodes_[node_u].AssignEdge(node_v, goodness_uv);
      ABSL_DCHECK(nodes_[node_v].IsNeighbor(node_u));
      ABSL_DCHECK_EQ(nodes_[node_v].GetNeighborInfo(node_u).goodness,
                     kDefaultGoodness);
      nodes_to_update_in_pq.insert(node_u);
    }
  }
}

// Updates the node_pq_ value of this node. If the node has no more incident
// active edges, the node is removed from node_pq_.
void ApproximateSubgraphHacGraph::UpdateNodePQ(NodeId node_id) {
  // Previous goodness and neighbor stored in the PQ.
  ABSL_DCHECK(is_active_[node_id]);

  // Get the current goodness/ngh, and update PQ if different.
  auto get_goodness = [&](NodeId node_u, NodeId node_v) {
    return Goodness(node_u, node_v);
  };
  auto [new_goodness, _] =
      nodes_[node_id].GetGoodEdge(node_id, one_plus_eps_, get_goodness);
  if (new_goodness == std::numeric_limits<double>::max()) {
    node_pq_.Remove(node_id);
  } else {
    node_pq_.InsertOrUpdate(node_id, -1 * new_goodness);
  }
}

// Uses (1+alpha) approximations for best_u and best_v.
// Uses the true edge weight of (u,v).
double ApproximateSubgraphHacGraph::Goodness(NodeId node_u, NodeId node_v) {
  double best_u = nodes_[node_u].ApproximateBestWeightAndId().first;
  double best_v = nodes_[node_v].ApproximateBestWeightAndId().first;
  double mm_u = min_merge_similarities_[node_u];
  double mm_v = min_merge_similarities_[node_v];
  double w_uv = EdgeWeight(node_u, node_v);  // Using CurrentClusterSize.
  return std::max(best_u, best_v) / std::min({w_uv, mm_u, mm_v});
}

double ApproximateSubgraphHacGraph::EdgeWeight(NodeId node_u,
                                               NodeId node_v) const {
  return nodes_[node_u].EdgeWeight(node_v, CurrentClusterSize(node_v));
}

double ApproximateSubgraphHacGraph::EdgeWeightUnnormalized(
    NodeId node_u, NodeId node_v) const {
  auto [partial_weight, cluster_size_estimate, _] =
      nodes_[node_u].Neighbors().at(node_v);
  return partial_weight * cluster_size_estimate;
}

bool ApproximateSubgraphHacGraph::IsActive(NodeId node_u) {
  return is_active_[node_u];
}

size_t ApproximateSubgraphHacGraph::CurrentClusterSize(NodeId node_u) const {
  return nodes_[node_u].CurrentClusterSize();
}

std::vector<NodeId> ApproximateSubgraphHacGraph::Neighbors(
    NodeId node_id) const {
  const auto& neighbors_map = nodes_[node_id].Neighbors();
  std::vector<NodeId> neighbors;
  neighbors.reserve(neighbors_map.size());
  for (const auto& [node_v, _] : neighbors_map) {
    neighbors.push_back(node_v);
  }
  return neighbors;
}

}  // namespace graph_mining::in_memory
