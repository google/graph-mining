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

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_node.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"
#include "utils/math.h"

namespace graph_mining::in_memory {

namespace {
using NodeId = ApproximateSubgraphHacNode::NodeId;
}

ApproximateSubgraphHacNode::ApproximateSubgraphHacNode(NodeId cluster_size,
                                                       double one_plus_alpha)
    : prev_best_weight_(std::numeric_limits<double>::min()),
      current_cluster_size_(cluster_size),
      last_updated_cluster_size_(cluster_size),
      one_plus_alpha_(one_plus_alpha) {}

size_t ApproximateSubgraphHacNode::NumAssignedEdges() const {
  return goodness_.size();
}

bool ApproximateSubgraphHacNode::IsNeighbor(NodeId neighbor_id) const {
  return neighbor_info_.contains(neighbor_id);
}

const absl::flat_hash_map<NodeId, ApproximateSubgraphHacNode::NeighborInfo>&
ApproximateSubgraphHacNode::Neighbors() const {
  return neighbor_info_;
}

ApproximateSubgraphHacNode::NeighborInfo
ApproximateSubgraphHacNode::GetNeighborInfo(NodeId neighbor_id) const {
  return neighbor_info_.at(neighbor_id);
}

size_t ApproximateSubgraphHacNode::CurrentClusterSize() const {
  return current_cluster_size_;
}

void ApproximateSubgraphHacNode::UpdateClusterSize(size_t new_cluster_size) {
  current_cluster_size_ = new_cluster_size;
}

void ApproximateSubgraphHacNode::UpdateLastUpdatedClusterSize() {
  last_updated_cluster_size_ = current_cluster_size_;
}

void ApproximateSubgraphHacNode::InsertEdge(NodeId neighbor,
                                            size_t neighbor_cluster_size,
                                            double weight) {
  double partial_weight = weight * CurrentClusterSize();
  ABSL_CHECK(neighbor_info_
                 .insert({neighbor,
                          {.partial_weight = partial_weight,
                           .cluster_size = neighbor_cluster_size,
                           .goodness = kDefaultGoodness}})
                 .second);

  ABSL_CHECK(all_neighbors_.insert({partial_weight, neighbor}).second);
  prev_best_weight_ = std::max(prev_best_weight_, weight);
}

double ApproximateSubgraphHacNode::EdgeWeight(NodeId neighbor,
                                              size_t neighbor_size) const {
  auto [partial_weight, cluster_size_estimate, _] = neighbor_info_.at(neighbor);
  return (partial_weight * cluster_size_estimate) /
         (CurrentClusterSize() * neighbor_size);
}

void ApproximateSubgraphHacNode::AssignEdge(NodeId neighbor,
                                            double goodness_to_neighbor) {
  ABSL_DCHECK(neighbor_info_.contains(neighbor));
  ABSL_CHECK(goodness_.insert({goodness_to_neighbor, neighbor}).second);
  neighbor_info_[neighbor].goodness = goodness_to_neighbor;
}

bool ApproximateSubgraphHacNode::BestWeightChangedEnough() const {
  double cur_best = ApproximateBestWeightAndId().first;
  // Check if best dropped sufficiently to broadcast.
  return cur_best < prev_best_weight_ / one_plus_alpha_;
}

void ApproximateSubgraphHacNode::UpdateEdge(NodeId neighbor,
                                            double partial_weight,
                                            size_t cluster_size) {
  // Update partial weights. Leave goodness info alone.
  auto old_goodness = neighbor_info_[neighbor].goodness;
  auto old_partial_weight = neighbor_info_[neighbor].partial_weight;
  all_neighbors_.erase({old_partial_weight, neighbor});

  neighbor_info_[neighbor] = {.partial_weight = partial_weight,
                              .cluster_size = cluster_size,
                              .goodness = old_goodness};
  all_neighbors_.insert({partial_weight, neighbor});
}

std::pair<double, NodeId>
ApproximateSubgraphHacNode::ApproximateBestWeightAndId() const {
  if (all_neighbors_.empty()) {
    return {0, std::numeric_limits<NodeId>::max()};
  }
  auto [best_partial_weight, best_ngh] = *(all_neighbors_.begin());
  double best_weight = best_partial_weight / CurrentClusterSize();
  return {best_weight, best_ngh};
}

bool ApproximateSubgraphHacNode::ClusterSizeChangedEnough() const {
  return CurrentClusterSize() >= one_plus_alpha_ * last_updated_cluster_size_;
}

std::pair<double, NodeId> ApproximateSubgraphHacNode::GetGoodEdge(
    NodeId node_id, double threshold,
    std::function<double(NodeId, NodeId)> get_goodness) {
  auto best_goodness = kDefaultGoodness;
  auto best_neighbor = std::numeric_limits<NodeId>::max();
  std::vector<NodeId> neighbors_to_fix;

  // Scan the assigned edges for this endpoint.
  for (auto it = goodness_.begin(); it != goodness_.end();) {
    auto [old_goodness, node_v] = *it;
    ABSL_DCHECK(IsNeighbor(node_v));
    ABSL_DCHECK_EQ(GetNeighborInfo(node_v).goodness, old_goodness);

    // Get the current goodness of the edge.
    double current_goodness = get_goodness(node_id, node_v);

    // If the edge's goodness is below the threshold, we're done.
    if (current_goodness <= threshold ||
        AlmostEquals(current_goodness, threshold)) {
      std::tie(best_goodness, best_neighbor) = {current_goodness, node_v};
      break;
    }
    // Otherwise, recompute the goodness of this edge at the end of the loop.
    it = goodness_.erase(it);
    neighbors_to_fix.push_back(node_v);

    // Every remaining (unexplored) edge has old_goodness > threshold.
    // For these edges, their true goodness can be <= threshold / (1+alpha),
    // i.e., for threshold = (1+eps), this would be <= (1+eps)/(1+alpha).
    if (old_goodness > threshold) {
      break;
    }
  }

  // Recompute goodness values for edges to fix and reassign them to ourselves.
  for (NodeId neighbor : neighbors_to_fix) {
    double goodness = get_goodness(node_id, neighbor);
    AssignEdge(neighbor, goodness);
  }

  return {best_goodness, best_neighbor};
}

bool ApproximateSubgraphHacNode::MaybeBroadcastClusterSize(
    NodeId node_id, absl::Span<ApproximateSubgraphHacNode> nodes,
    const std::vector<bool>& is_active) {
  if (!ClusterSizeChangedEnough()) {
    return false;
  }

  auto new_cluster_size = CurrentClusterSize();
  // Update the last updated size.
  UpdateLastUpdatedClusterSize();

  // Map over all neighbors of v, and update in their storage / heap.
  for (auto [node_w, partial_vw] : neighbor_info_) {
    auto [partial_weight_vw, size_estimate_vw, goodness_vw] = partial_vw;

    // Only active neighbors need updates.
    if (is_active[node_w]) {
      double cut_weight_vw = partial_weight_vw * size_estimate_vw;
      double new_partial_weight = cut_weight_vw / new_cluster_size;
      // Update w. Note that goodness_wv is left unchanged. If this edge is
      // assigned to node_w, this means we can still find it using the stored
      // goodness value in goodness_.
      nodes[node_w].UpdateEdge(node_id, new_partial_weight, new_cluster_size);
    }
  }
  return true;
}

bool ApproximateSubgraphHacNode::MaybeReassignEdges(
    NodeId node_id, absl::Span<ApproximateSubgraphHacNode> nodes,
    const std::vector<bool>& is_active,
    absl::flat_hash_set<NodeId>* nodes_to_update_in_pq_in_pq,
    std::function<double(NodeId, NodeId)> get_goodness) {
  if (!BestWeightChangedEnough()) {
    return false;
  }
  double best = ApproximateBestWeightAndId().first;
  prev_best_weight_ = best;

  // Go over node_v's assigned edges, update goodness values, and potentially
  // reassign.
  absl::btree_set<GoodnessAndId, std::less<GoodnessAndId>> new_node_v;
  for (auto [goodness_vw, node_w] : goodness_) {
    ABSL_DCHECK(is_active[node_w]);

    auto current_goodness_vw = get_goodness(node_id, node_w);
    auto best_w = nodes[node_w].ApproximateBestWeightAndId().first;
    neighbor_info_[node_w].goodness = kDefaultGoodness;

    ABSL_DCHECK_EQ(nodes[node_w].GetNeighborInfo(node_id).goodness,
              kDefaultGoodness);

    if (best >= best_w) {
      // Leave the edge at this endpoint and just update the goodness value.
      new_node_v.insert({current_goodness_vw, node_w});
      neighbor_info_[node_w].goodness = current_goodness_vw;
      ABSL_DCHECK_EQ(nodes[node_w].GetNeighborInfo(node_id).goodness,
                kDefaultGoodness);
    } else {
      nodes[node_w].AssignEdge(node_id, current_goodness_vw);
      nodes_to_update_in_pq_in_pq->insert(node_w);
    }
  }
  goodness_ = std::move(new_node_v);
  return true;
}

std::pair<std::vector<std::pair<NodeId, NodeId>>, absl::flat_hash_set<NodeId>>
ApproximateSubgraphHacNode::Merge(NodeId node_from, NodeId node_to,
                                  absl::Span<ApproximateSubgraphHacNode> nodes,
                                  std::vector<bool>& is_active) {
  auto& node_from_neighbor_info = nodes[node_from].neighbor_info_;
  auto& node_to_neighbor_info = nodes[node_to].neighbor_info_;

  auto& node_to_goodness = nodes[node_to].goodness_;
  auto& node_to_all_neighbors = nodes[node_to].all_neighbors_;
  auto new_cluster_size_v = nodes[node_to].CurrentClusterSize() +
                            nodes[node_from].CurrentClusterSize();

  // Update node_v's cluster size.
  nodes[node_to].UpdateClusterSize(new_cluster_size_v);

  // Remove references to node_from from node_v.
  auto [partial_weight_vu, size_estimate_vu, goodness_vu] =
      node_to_neighbor_info[node_from];

  if (goodness_vu != kDefaultGoodness) {
    ABSL_DCHECK(nodes[node_to].goodness_.contains({goodness_vu, node_from}));
  }
  node_to_goodness.erase({goodness_vu, node_from});
  node_to_all_neighbors.erase({partial_weight_vu, node_from});
  node_to_neighbor_info.erase(node_from);

  // Will never need to use our assigned edges again.
  goodness_.clear();

  std::vector<std::pair<NodeId, NodeId>> edges_to_reassign;
  absl::flat_hash_set<NodeId> nodes_to_update_in_pq;

  // Want to do work proportional to node_from only.
  for (const auto& [node_w, partial_uw] : node_from_neighbor_info) {
    // partial_weight_uw = cut_weight(u,w)/|w|
    auto [partial_weight_uw, size_estimate_uw, goodness_uw] = partial_uw;
    double cut_weight_uw = partial_weight_uw * size_estimate_uw;

    ABSL_CHECK_NE(node_from, node_w);  // No self-loops.
    if (node_w == node_to) {
      continue;
    }

    // For updating w.
    auto& node_w_neighbor_info = nodes[node_w].neighbor_info_;
    auto& node_w_all_neighbors = nodes[node_w].all_neighbors_;
    auto& node_w_goodness = nodes[node_w].goodness_;
    auto cluster_size_w = nodes[node_w].CurrentClusterSize();

    // If w is active, start by removing reference to u.
    if (is_active[node_w]) {
      auto [partial_weight_wu, _, goodness_wu] =
          node_w_neighbor_info[node_from];
      node_w_neighbor_info.erase(node_from);
      node_w_all_neighbors.erase({partial_weight_wu, node_from});
      if (goodness_wu != kDefaultGoodness) {
        ABSL_DCHECK(node_w_goodness.contains({goodness_wu, node_from}));
        node_w_goodness.erase({goodness_wu, node_from});
      }
    }

    // Case 1: w in N(node_from) \cap N(node_to)
    if (node_to_neighbor_info.contains(node_w)) {
      // Sum the partial weights
      auto [partial_weight_vw, size_estimate_vw, goodness_vw] =
          node_to_neighbor_info.find(node_w)->second;
      double cut_weight_vw = partial_weight_vw * size_estimate_vw;
      auto new_partial_weight_vw =
          (cut_weight_uw + cut_weight_vw) / cluster_size_w;

      // Update weight info in v's neighbor info. Use a default goodness
      // since this edge will get reassigned.
      node_to_neighbor_info[node_w] = {.partial_weight = new_partial_weight_vw,
                                       .cluster_size = cluster_size_w,
                                       .goodness = kDefaultGoodness};

      // Remove previous_weight_key from v's set and update with new weights.
      PartialWeightAndId previous_weight_key = {partial_weight_vw, node_w};
      node_to_all_neighbors.erase(previous_weight_key);
      node_to_all_neighbors.insert({new_partial_weight_vw, node_w});

      if (goodness_vw != kDefaultGoodness) {
        // Sanity check about goodness state.
        ABSL_DCHECK(node_to_goodness.contains({goodness_vw, node_w}));
        ABSL_DCHECK(is_active[node_w]);
        // Remove from node_to's goodness.
        node_to_goodness.erase({goodness_vw, node_w});
      }

      if (is_active[node_w]) {
        // Remove reference to v's old state in w's storage.
        auto [partial_weight_wv, _, goodness_wv] =
            node_w_neighbor_info[node_to];
        node_w_all_neighbors.erase({partial_weight_wv, node_to});

        if (goodness_wv != kDefaultGoodness) {
          // Sanity check.
          ABSL_DCHECK(node_w_goodness.contains({goodness_wv, node_to}));
          // Erase from goodness; the edge will be reassigned.
          node_w_goodness.erase({goodness_wv, node_to});
        }

        // Update v's info in w's neighborhood storage.
        auto new_partial_weight_wv =
            (cut_weight_uw + cut_weight_vw) / new_cluster_size_v;
        // Use kDefaultGoodness since this edge will be reassigned.
        node_w_neighbor_info[node_to] = {
            .partial_weight = new_partial_weight_wv,
            .cluster_size = new_cluster_size_v,
            .goodness = kDefaultGoodness};
        node_w_all_neighbors.insert({new_partial_weight_wv, node_to});

        // Reassign the (v,w) edge later.
        edges_to_reassign.push_back(
            {std::min(node_to, node_w), std::max(node_to, node_w)});
      } else {
        // should never store values in inactive nodes' goodness or neighbor
        // sets.
        ABSL_DCHECK(nodes[node_w].goodness_.empty());
        ABSL_DCHECK(nodes[node_w].neighbor_info_.empty()) << node_w;
        ABSL_DCHECK(nodes[node_w].all_neighbors_.empty());
      }
    } else {  // Case 2: otherwise, node_w in N(node_from) only.
      // First update entries for node_to.
      double new_partial_weight_vw = cut_weight_uw / cluster_size_w;
      node_to_neighbor_info[node_w] = {.partial_weight = new_partial_weight_vw,
                                       .cluster_size = cluster_size_w,
                                       .goodness = kDefaultGoodness};
      node_to_all_neighbors.insert({new_partial_weight_vw, node_w});

      // Next, if node_w is active, set correct references to node_to
      // (references to node_from already deleted).
      if (is_active[node_w]) {
        double new_partial_weight_wv = cut_weight_uw / new_cluster_size_v;
        ABSL_DCHECK(!node_w_neighbor_info.contains(node_to));
        ABSL_DCHECK(!node_w_neighbor_info.contains(node_from));

        // Add new entries for node_to in w's neighbors.
        node_w_neighbor_info[node_to] = {
            .partial_weight = new_partial_weight_wv,
            .cluster_size = new_cluster_size_v,
            .goodness = kDefaultGoodness};
        node_w_all_neighbors.insert({new_partial_weight_wv, node_to});

        // Reassign the (v,w) edge later.
        edges_to_reassign.push_back(
            {std::min(node_to, node_w), std::max(node_to, node_w)});
      } else {
        // Should never store values in inactive nodes' goodness.
        ABSL_DCHECK(nodes[node_w].goodness_.empty());
        ABSL_DCHECK(nodes[node_w].neighbor_info_.empty());
      }
    }

    // Finished updating w. If its best priority changed, update it.
    if (is_active[node_w]) {
      nodes_to_update_in_pq.insert(node_w);
    }
  }
  neighbor_info_.clear();
  all_neighbors_.clear();

  // Set node_from to be inactive.
  is_active[node_from] = false;

  return std::make_pair(std::move(edges_to_reassign),
                        std::move(nodes_to_update_in_pq));
}

}  // namespace graph_mining::in_memory
