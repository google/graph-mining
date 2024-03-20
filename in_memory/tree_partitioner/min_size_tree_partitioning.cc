#include "in_memory/tree_partitioner/min_size_tree_partitioning.h"

#include <algorithm>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

namespace internal {

absl::StatusOr<SubtreeInformation> ProcessChildrenAndSubtreeWeights(
    const std::vector<NodeId>& parent_ids,
    const std::vector<double>& node_weights) {
  const int n = parent_ids.size();
  if (node_weights.size() != n) {
    return absl::InvalidArgumentError(
        "The sizes of parent_ids and node_weights are inconsistent.");
  }
  std::vector<std::vector<NodeId>> children(n);
  std::vector<double> subtree_weights(node_weights);
  // The number of children that have not been processed.
  std::vector<int> in_degree(n);
  for (int i = 0; i < n; ++i) {
    if (parent_ids[i] < -1 || parent_ids[i] >= n) {
      return absl::InvalidArgumentError("The parent id is out of range.");
    }
    if (node_weights[i] < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The node weight of node ", i, " is negative: ", node_weights[i]));
    }
    if (parent_ids[i] >= 0) {
      ++in_degree[parent_ids[i]];
    }
  }
  std::queue<NodeId> nodes_to_process;
  for (int i = 0; i < n; ++i) {
    if (in_degree[i] == 0) {
      nodes_to_process.push(i);
    } else {
      children[i].reserve(in_degree[i]);
    }
  }
  // Use topological sort to update subtree weights and children of each node.
  int num_nodes_processed = 0;
  while (!nodes_to_process.empty()) {
    const int current_node = nodes_to_process.front();
    nodes_to_process.pop();
    ++num_nodes_processed;
    if (parent_ids[current_node] >= 0) {
      --in_degree[parent_ids[current_node]];
      subtree_weights[parent_ids[current_node]] +=
          subtree_weights[current_node];
      children[parent_ids[current_node]].push_back(current_node);
      if (in_degree[parent_ids[current_node]] == 0) {
        nodes_to_process.push(parent_ids[current_node]);
      }
    }
  }
  if (num_nodes_processed != n) {
    return absl::InvalidArgumentError("Invalid parent ids: cycle detected.");
  }
  return SubtreeInformation{std::move(children), std::move(subtree_weights)};
}

std::vector<std::pair<NodeId, NodeId>> PartitionClusters(
    const std::vector<std::pair<NodeId, double>>& nodes_with_unassigned_weights,
    const double min_weight_threshold, const NodeId& root_id) {
  std::vector<std::pair<NodeId, double>> sorted_nodes_with_unassigned_weights(
      nodes_with_unassigned_weights);
  // Sort nodes by unassigned weights in their subtrees.
  absl::c_sort(
      sorted_nodes_with_unassigned_weights,
      [](std::pair<NodeId, double>& x, const std::pair<NodeId, double>& y) {
        return (x.second < y.second) ||
               (x.second == y.second && x.first < y.first);
      });
  std::vector<std::pair<NodeId, NodeId>> cluster_id_map;
  cluster_id_map.reserve(sorted_nodes_with_unassigned_weights.size());
  double last_cluster_weight = 0, current_cluster_weight = 0;
  NodeId last_cluster_id = -1, current_cluster_id = -1;
  for (const auto& [node_id, weight] : sorted_nodes_with_unassigned_weights) {
    current_cluster_weight += weight;
    if (current_cluster_id == -1) {
      current_cluster_id = node_id;
    }
    cluster_id_map.push_back({node_id, current_cluster_id});
    if (current_cluster_weight >= min_weight_threshold) {
      last_cluster_weight = current_cluster_weight;
      last_cluster_id = current_cluster_id;
      current_cluster_weight = 0;
      current_cluster_id = -1;
    }
  }
  // If the last cluster is not large enough, merge it with the previous
  // cluster.
  if (current_cluster_id != -1 &&
      current_cluster_weight < min_weight_threshold) {
    ABSL_CHECK(last_cluster_weight >= min_weight_threshold &&
          last_cluster_weight <= 2 * min_weight_threshold &&
          last_cluster_id != -1);
    for (auto& [node_id, cluster_id] : cluster_id_map) {
      if (cluster_id == current_cluster_id) {
        cluster_id = last_cluster_id;
      }
    }
  }
  // For cluster containing the node with root_id, set the cluster id as
  // root_id.
  NodeId old_cluster_id;
  for (const auto& [node_id, cluster_id] : cluster_id_map) {
    if (node_id == root_id) {
      old_cluster_id = cluster_id;
      break;
    }
  }
  for (auto& [node_id, cluster_id] : cluster_id_map) {
    if (cluster_id == old_cluster_id) {
      cluster_id = root_id;
    }
  }
  return cluster_id_map;
}

// The recursion state of the subtree partitioning process.
struct PartitionSubtreeRecursionState {
  int current_root;
  double unassigned_weights_outside_current_tree;
  int num_children_processed;
  double current_unassigned_weights;
  std::vector<std::pair<NodeId, double>> nodes_with_unassigned_weights;
};

// result_parent_ids is properly set to partition the subtree. This is a
// non-recursive implementation of the recursive algorithm of partitioning the
// subtree, see go/min-size-tree-partitioner-design for more details.
void PartitionSubtree(const int root, const double min_weight_threshold,
                      const SubtreeInformation& subtree_information,
                      absl::Span<const double> node_weights,
                      std::vector<NodeId>* result_parent_ids) {
  std::stack<PartitionSubtreeRecursionState> state_stack;
  state_stack.push({root, 0, 0, subtree_information.subtree_weights[root], {}});
  state_stack.top().nodes_with_unassigned_weights.reserve(
      subtree_information.children[root].size() + 1);
  while (!state_stack.empty()) {
    PartitionSubtreeRecursionState& state = state_stack.top();
    ABSL_CHECK(state.current_unassigned_weights >= min_weight_threshold);
    // Recurse on children.
    if (state.num_children_processed <
        subtree_information.children[state.current_root].size()) {
      const int child =
          subtree_information
              .children[state.current_root][state.num_children_processed];
      state_stack.push({child,
                        state.current_unassigned_weights -
                            subtree_information.subtree_weights[child],
                        0,
                        state.current_unassigned_weights,
                        {}});
      state_stack.top().nodes_with_unassigned_weights.reserve(
          subtree_information.children[child].size() + 1);
      continue;
    }
    // Add current root node and its weight to the list of unassigned nodes in
    // the current tree.
    state.nodes_with_unassigned_weights.push_back(
        {state.current_root, node_weights[state.current_root]});
    double unassigned_weight_in_current_tree = 0;
    for (const auto& node_with_unassigned_weight :
         state.nodes_with_unassigned_weights) {
      unassigned_weight_in_current_tree += node_with_unassigned_weight.second;
    }
    // If unassigned total weights in the current tree is below
    // min_weight_threshold, add the root of the current tree into
    // nodes_with_unassigned_weihghts of its parent.
    if (unassigned_weight_in_current_tree < min_weight_threshold) {
      const auto unassigned_weight_of_parent =
          state.unassigned_weights_outside_current_tree +
          unassigned_weight_in_current_tree;
      const auto node_with_unassigned_weights_for_parent =
          std::make_pair(state.current_root, unassigned_weight_in_current_tree);
      state_stack.pop();
      auto& parent_state = state_stack.top();
      parent_state.current_unassigned_weights = unassigned_weight_of_parent;
      parent_state.nodes_with_unassigned_weights.push_back(
          node_with_unassigned_weights_for_parent);
      ++parent_state.num_children_processed;
      continue;
    }
    // Unassigned total weight in the current tree is above
    // min_weight_threshold. In this case, we can partition children and the
    // root such that each cluster has weight above min_weight_threshold. In
    // addition, each cluster either has weight at most 3 *
    // min_weight_threshold or at most min_weight_threshold after removal of
    // the node with the largest weight.
    std::vector<std::pair<NodeId, NodeId>> cluster_id_map =
        PartitionClusters(state.nodes_with_unassigned_weights,
                          min_weight_threshold, state.current_root);
    NodeId old_parent_of_root = (*result_parent_ids)[state.current_root];
    for (const auto& [node_id, cluster_id] : cluster_id_map) {
      if (node_id == cluster_id) {
        (*result_parent_ids)[node_id] = -1;
      } else {
        (*result_parent_ids)[node_id] = cluster_id;
      }
    }
    // If total unassigned weights outside the tree is below
    // min_weight_threshold and the current_root is not the root of the entire
    // tree, we need to keep a large cluster unassigned to "save" the
    // unassigned nodes outside the tree.
    if (old_parent_of_root != -1 &&
        state.unassigned_weights_outside_current_tree < min_weight_threshold) {
      // Note: it is not necessary to recover
      // (*result_parent_ids)[current_root] to be old_parent_of_root here
      // since it will be handeled by the previous level of the recursion.
      ABSL_CHECK_EQ((*result_parent_ids)[state.current_root], -1);
      double weight_of_cluster_containing_root = 0;
      double max_unassigned_weight = 0;
      for (const auto& node_with_unassigned_weight :
           state.nodes_with_unassigned_weights) {
        if (node_with_unassigned_weight.second > max_unassigned_weight) {
          max_unassigned_weight = node_with_unassigned_weight.second;
        }
        if ((*result_parent_ids)[node_with_unassigned_weight.first] ==
                state.current_root ||
            node_with_unassigned_weight.first == state.current_root) {
          weight_of_cluster_containing_root +=
              node_with_unassigned_weight.second;
        }
      }
      ABSL_CHECK(weight_of_cluster_containing_root >= min_weight_threshold &&
            weight_of_cluster_containing_root <
                std::max(2 * min_weight_threshold, max_unassigned_weight) +
                    min_weight_threshold);
      const auto unassigned_weight_of_parent =
          state.unassigned_weights_outside_current_tree +
          weight_of_cluster_containing_root;
      const auto node_with_unassigned_weights_for_parent =
          std::make_pair(state.current_root, weight_of_cluster_containing_root);
      state_stack.pop();
      auto& parent_state = state_stack.top();
      parent_state.current_unassigned_weights = unassigned_weight_of_parent;
      parent_state.nodes_with_unassigned_weights.push_back(
          node_with_unassigned_weights_for_parent);
      ++parent_state.num_children_processed;
    } else {
      const auto unassigned_weight_of_parent =
          state.unassigned_weights_outside_current_tree;
      state_stack.pop();
      if (!state_stack.empty()) {
        auto& parent_state = state_stack.top();
        parent_state.current_unassigned_weights = unassigned_weight_of_parent;
        ++parent_state.num_children_processed;
      }
    }
  }
}

}  // namespace internal

absl::StatusOr<std::vector<NodeId>> MinWeightedSizeTreePartitioning(
    const std::vector<NodeId>& parent_ids,
    const std::vector<double>& node_weights,
    const double min_weight_threshold) {
  if (min_weight_threshold < 0) {
    return absl::InvalidArgumentError("Negative min_weight_threshold.");
  }

  ASSIGN_OR_RETURN(
      auto subtree_information,
      internal::ProcessChildrenAndSubtreeWeights(parent_ids, node_weights));

  // Initialize final parent ids as input parent_ids.
  std::vector<NodeId> result_parent_ids(parent_ids);
  for (int i = 0; i < parent_ids.size(); ++i) {
    if (parent_ids[i] == -1) {
      if (subtree_information.subtree_weights[i] > min_weight_threshold) {
        // Partition the tree.
        PartitionSubtree(i, min_weight_threshold, subtree_information,
                         node_weights, &result_parent_ids);
      }
    }
  }

  return result_parent_ids;
}

}  // namespace graph_mining::in_memory
