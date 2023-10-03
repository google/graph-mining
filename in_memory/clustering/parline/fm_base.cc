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

#include "in_memory/clustering/parline/fm_base.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {
namespace {

// The structure stored on the priority queue. It represents a node in the
// graph, and its gain is the current improvement to the cut value should the
// node be moved from its cluster to the other cluster in consideration.
struct PqNode {
  gbbs::uintE node_id;
  double gain = 0.0;
};

// A strict weak ordering such that larger gains will be popped earlier in a
// max-priority queue.
class PqNodeCompare {
 public:
  bool operator()(const PqNode* node1, const PqNode* node2) const {
    return node1->gain == node2->gain ? node1->node_id < node2->node_id
                                      : node1->gain > node2->gain;
  }
};

// A simple priority queue implementation that also supports searching and
// adjusting arbitrary node values, and keeps track of total weight.
// Does not take ownership of any elements.
class WeightedPriorityQueue {
 public:
  void Push(PqNode* node) { set_.insert(node); }

  PqNode* Pop() { return set_.extract(set_.begin()).value(); }

  PqNode* Top() { return *set_.begin(); }

  bool Contains(PqNode* node) { return set_.find(node) != set_.end(); }

  bool Empty() { return set_.empty(); }

  // Adjusts the gain value of a node.
  void Adjust(PqNode* node, double gain) {
    auto nh = set_.extract(node);
    nh.value()->gain = gain;
    set_.insert(std::move(nh));
  }

  double weight() const { return weight_; }
  void set_weight(double weight) { weight_ = weight; }

 private:
  absl::btree_set<PqNode*, PqNodeCompare> set_;
  // weight is the sum of the weights of all PqNodes.
  double weight_ = 0;
};

// The interface structure to the ChooseMove function; it captures only the
// results of a particular move.
// Gain is the improvement to the cut from the move.
// Slack is the distance (in node weight) between the larger cluster's size and
// the maximum allowed cluster size.
struct MoveStats {
  double gain;
  double pre_move_slack;
  double post_move_slack;
};

enum class MOVE_CHOICE { LEFT, RIGHT };

// The function to change to overwrite the FM heuristic behavior. When called,
// both moves have been verified as feasible given the balance conditions.
// Right now the following heuristic has been performing the best. One potential
// alternative is to increase the relative importance of imbalance based on
// the current distance to the maximum weight threshold.
MOVE_CHOICE ChooseMove(const MoveStats& left, const MoveStats& right) {
  return left.gain > right.gain ? MOVE_CHOICE::LEFT : MOVE_CHOICE::RIGHT;
}

double GetNodeWeight(const GbbsGraph& gbbs_graph, gbbs::uintE node_id) {
  auto graph = gbbs_graph.Graph();
  return graph->vertex_weights == nullptr ? 1.0
                                          : graph->vertex_weights[node_id];
}

// Fills out the MoveStats structure for a given candidate move.
MoveStats AnalyzeMove(const GbbsGraph& gbbs_graph, const PqNode& moved_node,
                      const WeightedPriorityQueue& from_pq,
                      const WeightedPriorityQueue& to_pq, double max_weight) {
  MoveStats stats;
  const double node_weight = GetNodeWeight(gbbs_graph, moved_node.node_id);
  stats.gain = moved_node.gain;
  stats.pre_move_slack =
      max_weight - std::max(to_pq.weight(), from_pq.weight());
  stats.post_move_slack = max_weight - std::max(to_pq.weight() + node_weight,
                                                from_pq.weight() - node_weight);
  return stats;
}

// Given the id of a node which has been moved, update the gain values of all
// its neighbor nodes in both clusters.
void UpdateGains(
    const GbbsGraph& gbbs_graph, const PqNode& moved_node,
    const absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>>& pq_node_index,
    WeightedPriorityQueue& from_pq, WeightedPriorityQueue& to_pq) {
  // Iterate over the edges of node
  auto graph = gbbs_graph.Graph();
  auto neighbors = graph->get_vertex(moved_node.node_id).out_neighbors();
  for (int i = 0; i < neighbors.get_degree(); ++i) {
    auto neighbor_id = neighbors.get_neighbor(i);
    if (neighbor_id == moved_node.node_id) {
      continue;  // this is a self edge; ignore it
    }
    auto it = pq_node_index.find(neighbor_id);
    // If we can't look up the neighbor's pq, it is in a cluster we aren't
    // currently considering.
    if (it == pq_node_index.end()) continue;

    PqNode* neighbor_pq_node = it->second.get();
    auto edge_weight = neighbors.get_weight(i);
    if (from_pq.Contains(neighbor_pq_node)) {
      double new_gain = neighbor_pq_node->gain + 2 * edge_weight;
      from_pq.Adjust(neighbor_pq_node, new_gain);
    } else if (to_pq.Contains(neighbor_pq_node)) {
      double new_gain = neighbor_pq_node->gain - 2 * edge_weight;
      to_pq.Adjust(neighbor_pq_node, new_gain);
    }
  }
}

// Update the priority queue structures and weights given the id of a node that
// should be moved between clusters.
void MakeMove(
    const GbbsGraph& gbbs_graph, const PqNode& moved_node,
    const absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>>& pq_node_index,
    WeightedPriorityQueue& from_pq, WeightedPriorityQueue& to_pq,
    std::vector<PqNode*>& move_history) {
  move_history.push_back(from_pq.Top());
  from_pq.Pop();
  auto node_weight = GetNodeWeight(gbbs_graph, moved_node.node_id);
  from_pq.set_weight(from_pq.weight() - node_weight);
  to_pq.set_weight(to_pq.weight() + node_weight);
  UpdateGains(gbbs_graph, moved_node, pq_node_index, from_pq, to_pq);
}

// For each node in the cluster, compute its gain and insert it into the
// pq. Also collect the total node weight in the pq.
void AddClusterToHeap(
    const GbbsGraph& gbbs_graph, const absl::flat_hash_set<NodeId>& cluster,
    const absl::flat_hash_set<NodeId>& external_nodes,
    WeightedPriorityQueue& pq,
    absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>>& pq_node_index) {
  auto graph = gbbs_graph.Graph();
  for (const auto& node_id : cluster) {
    std::unique_ptr<PqNode> pq_node = std::make_unique<PqNode>();
    pq_node->node_id = node_id;
    pq_node->gain = 0.0;

    // Iterate over the edges of node.
    auto neighbors = graph->get_vertex(node_id).out_neighbors();
    for (int i = 0; i < neighbors.get_degree(); ++i) {
      auto neighbor_id = neighbors.get_neighbor(i);
      auto edge_weight = neighbors.get_weight(i);
      if (cluster.find(neighbor_id) != cluster.end()) {
        pq_node->gain -= edge_weight;
      } else if (external_nodes.find(neighbor_id) != external_nodes.end()) {
        pq_node->gain += edge_weight;
      }
    }

    pq.Push(pq_node.get());
    pq.set_weight(pq.weight() + GetNodeWeight(gbbs_graph, node_id));
    pq_node_index[pq_node->node_id] = std::move(pq_node);
  }
}

// Compute the optimal point in the sequence of moves. Selects -1 if the
// best improvement was negative.
void FindBestMoveSubsequence(const std::vector<PqNode*>& move_history,
                             int& best_cut_point, double& improvement) {
  improvement = 0.0;
  best_cut_point = -1;
  double current_cut_value = 0.0;
  for (int i = 0; i < move_history.size(); ++i) {
    current_cut_value += move_history[i]->gain;
    if (current_cut_value >= improvement) {
      best_cut_point = i;
      improvement = current_cut_value;
    }
  }
}

// Locks a node from further consideration during the round.
void LockNode(WeightedPriorityQueue& pq, PqNode& node) { pq.Pop(); }

// Repeatedly pop off the top node of remaining_pq, and check if it is
// possible to move that node without violating the imbalance condition.
// If it is, log the move and pop it off the pq. Otherwise, lock the node.
// Assumes that empty_pq has no nodes left for move consideration.
void MoveRemainingPqNodes(
    const GbbsGraph& gbbs_graph,
    const absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>>& pq_node_index,
    double max_cluster_weight, WeightedPriorityQueue& remaining_pq,
    WeightedPriorityQueue& empty_pq, std::vector<PqNode*>& move_history) {
  while (!remaining_pq.Empty()) {
    auto top_node = remaining_pq.Top();
    MoveStats move_stats = AnalyzeMove(gbbs_graph, *top_node, remaining_pq,
                                       empty_pq, max_cluster_weight);
    if (move_stats.post_move_slack >= 0.0) {
      MakeMove(gbbs_graph, *top_node, pq_node_index, remaining_pq, empty_pq,
               move_history);
    } else {
      LockNode(remaining_pq, *top_node);
    }
  }
}

// While there are unlocked nodes left in both clusters, peek at the top
// nodes from each pq. If both moves are possible, defer to ChooseMove.
// If only one is possible, make that move.
// If neither move is possible, lock the larger node.
// Keep track of the sequence of moves, and return that sequence in
// move_history.
void CreateMoveSequenceFromHeaps(
    const GbbsGraph& gbbs_graph,
    const absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>>& pq_node_index,
    double max_cluster_weight, WeightedPriorityQueue& left_pq,
    WeightedPriorityQueue& right_pq, std::vector<PqNode*>& move_history) {
  while (!left_pq.Empty() && !right_pq.Empty()) {
    auto left_node = left_pq.Top();
    auto right_node = right_pq.Top();

    MoveStats left_stats = AnalyzeMove(gbbs_graph, *left_node, left_pq,
                                       right_pq, max_cluster_weight);
    MoveStats right_stats = AnalyzeMove(gbbs_graph, *right_node, right_pq,
                                        left_pq, max_cluster_weight);

    bool left_move_allowed = left_stats.post_move_slack >= 0.0;
    bool right_move_allowed = right_stats.post_move_slack >= 0.0;

    MOVE_CHOICE chosen_move;
    if (left_move_allowed && right_move_allowed) {
      chosen_move = ChooseMove(left_stats, right_stats);
    } else if (!left_move_allowed && right_move_allowed) {
      chosen_move = MOVE_CHOICE::RIGHT;
    } else if (left_move_allowed) {
      chosen_move = MOVE_CHOICE::LEFT;
    } else {
      // Both moves would violate the balance constraint; lock the larger node
      // and move on.
      if (GetNodeWeight(gbbs_graph, left_node->node_id) >
          GetNodeWeight(gbbs_graph, right_node->node_id)) {
        LockNode(left_pq, *left_node);
      } else {
        LockNode(right_pq, *right_node);
      }
      continue;
    }
    // Make the chosen move.
    if (chosen_move == MOVE_CHOICE::LEFT) {
      MakeMove(gbbs_graph, *left_node, pq_node_index, left_pq, right_pq,
               move_history);
    } else {
      MakeMove(gbbs_graph, *right_node, pq_node_index, right_pq, left_pq,
               move_history);
    }
  }

  // For the priority queue that has leftover nodes, move or lock them all in
  // descending order by gain.
  WeightedPriorityQueue& remaining_pq = left_pq;
  WeightedPriorityQueue& empty_pq = right_pq;
  if (!right_pq.Empty()) {
    remaining_pq = right_pq;
    empty_pq = left_pq;
  }

  MoveRemainingPqNodes(gbbs_graph, pq_node_index, max_cluster_weight,
                       remaining_pq, empty_pq, move_history);
}

}  // namespace

double FMBase::Improve(const GbbsGraph& graph,
                       const absl::flat_hash_set<NodeId>& cluster1,
                       const absl::flat_hash_set<NodeId>& cluster2,
                       double max_cluster_weight,
                       absl::flat_hash_set<NodeId>& cluster1_to_cluster2,
                       absl::flat_hash_set<NodeId>& cluster2_to_cluster1) {
  WeightedPriorityQueue left_pq;
  WeightedPriorityQueue right_pq;
  // The following map allows lookup of priority queue nodes in order to adjust
  // values.
  absl::flat_hash_map<NodeId, std::unique_ptr<PqNode>> pq_node_index;

  AddClusterToHeap(graph, cluster1, cluster2, left_pq, pq_node_index);
  AddClusterToHeap(graph, cluster2, cluster1, right_pq, pq_node_index);

  // Use this vector to record each swap and corresponding cut weight change.
  std::vector<PqNode*> move_history;

  // Repeatedly remove nodes from the priority queues and make the best valid
  // move, generating move_history:
  CreateMoveSequenceFromHeaps(graph, pq_node_index, max_cluster_weight, left_pq,
                              right_pq, move_history);

  // Compute the optimal point in the sequence of moves and add those moves to
  // the output.
  int best_cut_point;
  double improvement;
  FindBestMoveSubsequence(move_history, best_cut_point, improvement);
  for (int i = 0; i <= best_cut_point; ++i) {
    const NodeId moved_id = move_history[i]->node_id;
    if (cluster1.find(moved_id) != cluster1.end()) {
      cluster1_to_cluster2.insert(moved_id);
    } else {
      cluster2_to_cluster1.insert(moved_id);
    }
  }

  return improvement;
}

}  // namespace graph_mining::in_memory
