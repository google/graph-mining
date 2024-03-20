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

#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/parallel_dendrogram.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using NodeId = graph_mining::in_memory::NodeId;
using graph_mining::in_memory::Dendrogram;
using graph_mining::in_memory::ParallelDendrogram;

absl::StatusOr<absl::flat_hash_set<NodeId>> DynamicDendrogram::RemoveAncestors(
    absl::Span<const NodeId> node_ids) {
  absl::flat_hash_set<NodeId> removed_ancestors;
  // For each node, recursively remove the ancestors of its parent (if a parent
  // exists).
  for (const NodeId current_node : node_ids) {
    ASSIGN_OR_RETURN(auto has_parent, HasValidParent(current_node));
    if (has_parent) {
      // Get the sibling of current_node. `current_node` must have a sibling
      // since it has a parent.
      const auto& sib_itr = siblings_.find(current_node);
      if (sib_itr == siblings_.end()) {
        return absl::InternalError(
            absl::StrCat("node has no sibling, node: ", current_node));
      }
      const auto sib = sib_itr->second;
      if (sib == kInvalidClusterId) {
        return absl::InternalError(
            absl::StrCat("node has invalid sibling, node: ", current_node));
      }
      // Get the parent of current_node.
      ASSIGN_OR_RETURN(const auto parent_node, Parent(current_node));
      const auto parent = parent_node.parent_id;

      // Recursively remove ancestors of parent.
      ASSIGN_OR_RETURN(const auto local_ancestors, RemoveAncestors({parent}));
      removed_ancestors.insert(local_ancestors.begin(), local_ancestors.end());

      // Remove the current root `parent`.
      RemoveNode(parent, current_node, sib);
      removed_ancestors.insert(parent);
    }
  }
  return removed_ancestors;
}

std::pair<std::vector<DynamicDendrogram::Merge>, std::vector<NodeId>>
DynamicDendrogram::MergeSequence() const {
  absl::flat_hash_map<NodeId, absl::flat_hash_set<NodeId>> unfinished_children;

  for (const auto& [v, parent_edge] : parent_pointers_) {
    unfinished_children[parent_edge.parent_id].insert(v);
  }
  std::vector<Merge> merges;
  std::queue<std::pair<NodeId, ParentEdge>> ready_nodes;
  std::vector<NodeId> leaves;
  // leaf nodes are ready.
  for (const auto& [v, parent] : parent_pointers_) {
    if (!unfinished_children.contains(v)) {
      ready_nodes.push({v, parent});
      leaves.push_back(v);
    }
  }

  // A node is ready to be merged when both of its children are merged.
  while (!ready_nodes.empty()) {
    const auto [v, parent] = ready_nodes.front();
    ready_nodes.pop();
    // Skip root node.
    if (parent.parent_id == kInvalidClusterId) continue;
    // v is ready, so we remove it from unfinished_children.
    auto itr = unfinished_children.find(parent.parent_id);
    itr->second.erase(v);
    if (itr->second.empty()) {
      // parent of v is ready.
      const auto grand_parent = parent_pointers_.find(parent.parent_id)->second;
      ready_nodes.push({parent.parent_id, grand_parent});
      const auto sib = siblings_.find(v)->second;
      // node_a < node_b. This is to make testing easier.
      const auto node_a = std::min(v, sib);
      const auto node_b = std::max(v, sib);
      merges.push_back(
          Merge(parent.merge_similarity, node_a, node_b, parent.parent_id));
    }
  }

  return {std::move(merges), std::move(leaves)};
}

absl::StatusOr<std::pair<Dendrogram, std::vector<NodeId>>>
DynamicDendrogram::ConvertToDendrogram() const {
  const auto num_nodes = NumNodes();
  // map from dynamic dendro id to consecutive node id space.
  absl::flat_hash_map<NodeId, NodeId> node_map;
  std::vector<NodeId> node_map_reverse(num_nodes);
  Dendrogram dendrogram(num_nodes);

  auto [merges, leaves] = MergeSequence();
  ABSL_CHECK_EQ(leaves.size(), num_nodes);
  absl::c_sort(leaves);

  for (int i = 0; i < num_nodes; ++i) {
    node_map[leaves[i]] = i;
    node_map_reverse[i] = leaves[i];
  }

  for (const auto& m : merges) {
    const auto node_a = node_map[m.node_a];
    const auto node_b = node_map[m.node_b];
    ASSIGN_OR_RETURN(
        const NodeId parent_id,
        dendrogram.BinaryMerge(node_a, node_b, m.merge_similarity));
    node_map[m.parent_id] = parent_id;
  }

  return std::make_pair(std::move(dendrogram), node_map_reverse);
}

absl::StatusOr<std::pair<ParallelDendrogram, std::vector<NodeId>>>
DynamicDendrogram::ConvertToParallelDendrogram() const {
  ASSIGN_OR_RETURN(const auto& result, ConvertToDendrogram());
  const auto& [dendrogram, node_ids] = result;
  ParallelDendrogram parallel_dendrogram(dendrogram.NumClusteredNodes());
  for (std::size_t i = 0; i < dendrogram.Nodes().size(); ++i) {
    const auto& node = dendrogram.Nodes()[i];
    parallel_dendrogram.MergeToParent(i, node.parent_id, node.merge_similarity);
  }
  return std::make_pair(std::move(parallel_dendrogram), node_ids);
}

}  // namespace graph_mining::in_memory
