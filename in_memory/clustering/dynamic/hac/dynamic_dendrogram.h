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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_DENDROGRAM_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_DENDROGRAM_H_

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/parallel_dendrogram.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

// This class implements a dendrogram, or a node-weighted tree representing
// a set of clusterings.
// It is used by the DynamicHacClusterer object to maintain the dendrogram
// that is updated by a dynamic hierarchical clustering algorithm.
// Invariants:
// 1) if a node x has a parent y, then we have node y in the dendrogram
// 2) each internal node has 2 children who have the same merge similarities
// 3) each node has a unique node id, which may not be consecutive
class DynamicDendrogram {
 public:
  using NodeId = graph_mining::in_memory::NodeId;
  // An edge from a child cluster to a parent cluster with a float similarity.
  using ParentEdge = graph_mining::in_memory::ParallelDendrogram::ParentEdge;
  using ParallelDendrogram = graph_mining::in_memory::ParallelDendrogram;
  using Dendrogram = graph_mining::in_memory::Dendrogram;
  // Structure representing a binary merge.
  struct Merge {
   public:
    Merge();
    Merge(const double merge_similarity, const NodeId node_a,
          const NodeId node_b, const NodeId parent_id)
        : merge_similarity(merge_similarity),
          node_a(node_a),
          node_b(node_b),
          parent_id(parent_id) {
      CHECK_LT(node_a, node_b);
    }

    // The merge similarity of this merge.
    const double merge_similarity;
    // Node id of the first endpoint. Can be an internal node id or leaf node
    // id.
    const NodeId node_a;
    // Node id of the second endpoint. Can be an internal node id or leaf node
    // id.
    const NodeId node_b;
    // The id of the node's parent in the dendrogram.
    const NodeId parent_id;
  };

  DynamicDendrogram() : num_leaves_(0) {}

  bool HasNode(NodeId id) const { return parent_pointers_.contains(id); }

  void Clear() {
    num_leaves_ = 0;
    parent_pointers_.clear();
    siblings_.clear();
  }

  absl::StatusOr<ParentEdge> Parent(NodeId id) const {
    auto current_parent = parent_pointers_.find(id);
    if (current_parent == parent_pointers_.end())
      return absl::InvalidArgumentError(
          absl::StrCat("node not in dendrogram, node: ", id));
    return current_parent->second;
  }

  // Returns std::nullopt if any of they following cases:
  //  * `id` is a root and  does not have a sibling.
  //  * `id` is not in the dendrogram.
  std::optional<NodeId> Sibling(NodeId id) const {
    auto current_sibling = siblings_.find(id);
    if (current_sibling == siblings_.end()) {
      return std::nullopt;
    }
    if (current_sibling->second == kInvalidClusterId) {
      return std::nullopt;
    }
    return current_sibling->second;
  }

  absl::StatusOr<bool> HasValidParent(NodeId id) const {
    if (id >= kInvalidClusterId)
      return absl::InvalidArgumentError(
          absl::StrCat("node id >= Invalid Cluster Id, node: ", id,
                       " kInvalidClusterId, ", kInvalidClusterId));
    ASSIGN_OR_RETURN(auto parent, Parent(id));
    return parent.parent_id != kInvalidClusterId;
  }

  // Add a leaf node to the dendrogram
  absl::Status AddLeafNode(NodeId id) {
    if (id >= kInvalidClusterId)
      return absl::InvalidArgumentError(
          absl::StrCat("node id >= Invalid Cluster Id, node: ", id,
                       " kInvalidClusterId, ", kInvalidClusterId));
    ++num_leaves_;
    parent_pointers_[id] = {kInvalidClusterId, 0};
    siblings_[id] = kInvalidClusterId;
    return absl::OkStatus();
  }

  // Remove a singleton node that does not have a parent and does not have any
  // child. Returns error status when `id` has a parent.
  // WARNING: the tree structure will become invalid (the first invariant does
  // not hold) if `id` has children. This function does NOT check if `id` has
  // any child.
  absl::Status RemoveSingletonLeafNode(NodeId id) {
    ASSIGN_OR_RETURN(const bool has_parent, HasValidParent(id));
    if (has_parent) {
      return absl::InvalidArgumentError(
          absl::StrCat("node id has parent, node: ", id));
    }
    --num_leaves_;
    CHECK_EQ(parent_pointers_.erase(id), 1);
    CHECK_EQ(siblings_.erase(id), 1);
    return absl::OkStatus();
  }

  // Add an internal node with children `child1` and `child2`.
  // `child1` and `child2` should already exist in the dendrogram.
  // When this returns non-OK status, the DynamicDendrogram data structure
  // becomes not usable and no longer keeps the invariants.
  absl::Status AddInternalNode(NodeId id, NodeId child1, NodeId child2,
                               float similarity) {
    if (id >= kInvalidClusterId)
      return absl::InvalidArgumentError(
          absl::StrCat("node id >= Invalid Cluster Id ", id));

    auto id_entry = parent_pointers_.find(id);
    if (id_entry != parent_pointers_.end())
      return absl::AlreadyExistsError(
          absl::StrCat("parent node already in dendrogram, parent: ", id));

    for (auto child : {child1, child2}) {
      auto child_entry = parent_pointers_.find(child);
      if (child_entry == parent_pointers_.end())
        return absl::FailedPreconditionError(
            absl::StrCat("child node not in dendrogram, child: ", child));
      if (child_entry->second.parent_id != kInvalidClusterId)
        return absl::FailedPreconditionError(
            absl::StrCat("child node already has parent, child: ", child));
      child_entry->second = {id, similarity};
    }

    parent_pointers_[id] = {kInvalidClusterId, 0};
    siblings_[id] = kInvalidClusterId;
    siblings_[child1] = child2;
    siblings_[child2] = child1;
    return absl::OkStatus();
  }

  // Remove all ancestors of nodes in `node_ids` and return the removed
  // ancestors. The nodes themselves are not deleted. The children of removed
  // nodes (except those children which are among the deleted ancestors) become
  // root nodes. Prereq: Nodes in `node_ids` should not be the ancestor of other
  // nodes in `node_ids`.
  absl::StatusOr<absl::flat_hash_set<NodeId>> RemoveAncestors(
      absl::Span<const NodeId> node_ids);

  NodeId NumNodes() const { return num_leaves_; }

  // Returns a sequence of merges that respects the topology of the dendrogram,
  // and the leaf nodes. Ids of merge is the dendrogram node id. The returned
  // merges can be applied in order and reconstruct the dendrogram.
  std::pair<std::vector<Merge>, std::vector<NodeId>> MergeSequence() const;

  // Returns a dendrogram with consecutive leaf ids [0 ... num_leaves_],
  // and a mapping from new dendrogram leaf ids to current dendrogram's leaf
  // ids. Current leaf nodes are ordered from small to large in the mapping. So
  // if the original dendrogram has consecutive leaf ids [0 ... num_leaves_],
  // the returned dendrogram has the same corresponding leaf ids.
  absl::StatusOr<std::pair<Dendrogram, std::vector<NodeId>>>
  ConvertToDendrogram() const;

  // The same as ConvertToDendrogram, except returns a ParallelDendrogram.
  absl::StatusOr<std::pair<ParallelDendrogram, std::vector<NodeId>>>
  ConvertToParallelDendrogram() const;

 private:
  // Remove `id`. Can make the dendrogram invalid if not used to delete root
  // nodes. When a node `id` is deleted, we also delete the parents of its two
  // children, `child1` and `child2`.
  void RemoveNode(NodeId id, NodeId child1, NodeId child2) {
    parent_pointers_.erase(id);
    siblings_.erase(id);
    RemoveParent(child1);
    RemoveParent(child2);
  }

  // Remove the parent of `id`.
  void RemoveParent(NodeId id) {
    parent_pointers_[id] = {kInvalidClusterId, 0};
    siblings_[id] = kInvalidClusterId;
  }

  static constexpr NodeId kInvalidClusterId =
      std::numeric_limits<NodeId>::max();

  absl::flat_hash_map<NodeId, ParentEdge> parent_pointers_;
  absl::flat_hash_map<NodeId, NodeId> siblings_;
  NodeId num_leaves_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_DYNAMIC_DENDROGRAM_H_
