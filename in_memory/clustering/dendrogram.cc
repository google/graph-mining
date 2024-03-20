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

#include "in_memory/clustering/dendrogram.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/types.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/status_macros.h"
#include "utils/math.h"

namespace graph_mining::in_memory {

using absl::InvalidArgumentError;

/* ======================== DendrogramNode functions ======================== */
DendrogramNode::DendrogramNode()
    : parent_id(kNoParentId),
      merge_similarity(std::numeric_limits<double>::infinity()) {}

DendrogramNode::DendrogramNode(ParentId parent_id, double merge_similarity)
    : parent_id(parent_id), merge_similarity(merge_similarity) {}

/* ======================= DendrogramMerge functions ======================== */
DendrogramMerge::DendrogramMerge()
    : merge_similarity(std::numeric_limits<double>::infinity()),
      node_a(DendrogramNode::kNoParentId),
      node_b(DendrogramNode::kNoParentId),
      parent_id(DendrogramNode::kNoParentId),
      min_merge_similarity(std::numeric_limits<double>::infinity()) {}

DendrogramMerge::DendrogramMerge(double merge_similarity, ParentId node_a,
                                 ParentId node_b, ParentId parent_id,
                                 double min_merge_similarity)
    : merge_similarity(merge_similarity),
      node_a(node_a),
      node_b(node_b),
      parent_id(parent_id),
      min_merge_similarity(min_merge_similarity) {}

/* ========================== Dendrogram functions ========================== */
Dendrogram::Dendrogram(size_t num_clustered_nodes)
    : nodes_(std::vector<DendrogramNode>(num_clustered_nodes)),
      num_clustered_nodes_(num_clustered_nodes) {}

absl::Status Dendrogram::Init(std::vector<DendrogramNode> nodes,
                              size_t num_clustered_nodes) {
  if (num_clustered_nodes > 0 && nodes.size() >= 2 * num_clustered_nodes) {
    return absl::InvalidArgumentError("Too many dendrogram nodes");
  }
  if (num_clustered_nodes == 0 && !nodes.empty()) {
    return absl::InvalidArgumentError("Too many dendrogram nodes");
  }

  std::vector<std::vector<double>> children_vecs(nodes.size());

  // Check conditions (5) and (6), and add children similarities to their
  // parent's vectors.
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto [parent_id, weight] = nodes[i];
    if (parent_id != kNoParentId) {  // This is an actual merge.
      if (parent_id >= nodes.size()) {
        return InvalidArgumentError("Parent id too large");
      }
      if (i >= parent_id) {
        return InvalidArgumentError("Parent id is not larger than node id.");
      }
      if (weight <= 0) {
        return InvalidArgumentError("Non-positive merge similarity");
      }
      children_vecs[parent_id].push_back(weight);
    }
  }

  // Check conditions (1) -- (4).
  for (size_t i = 0; i < children_vecs.size(); ++i) {
    auto& children = children_vecs[i];
    if (i < num_clustered_nodes && !children.empty()) {
      return InvalidArgumentError("Leaf node cannot have children nodes");
    }
    if (i >= num_clustered_nodes) {
      // This is an inner node.
      if (children.size() < 2) {
        return InvalidArgumentError(
            "Non-leaf node must have at least 2 children");
      }
      std::nth_element(children.begin(), children.begin() + 1, children.end(),
                       std::greater<>());
      if (children[0] != children[1])
        return InvalidArgumentError(
            "The two largest merge similarities do not match");
    }
  }
  num_clustered_nodes_ = num_clustered_nodes;
  nodes_ = std::move(nodes);
  return absl::OkStatus();
}

absl::StatusOr<Dendrogram::ParentId> Dendrogram::BinaryMerge(
    Dendrogram::ParentId a, Dendrogram::ParentId b, double merge_similarity) {
  ParentId new_id = nodes_.size();
  if (a >= nodes_.size() || b >= nodes_.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Trying to merge %d and %d, but one of these ids is "
                        "not a valid cluster id in the dendrogram (valid "
                        "cluster ids are currently between [0, and %d)",
                        a, b, new_id));
  }
  if (new_id >= 2 * num_clustered_nodes_ - 1) {
    return absl::InternalError(
        absl::StrFormat("Trying to merge %d and %d, but new parent id (%d) is "
                        "more than the number of allowed merges (%d).",
                        a, b, new_id, 2 * num_clustered_nodes_ - 1));
  }
  nodes_.push_back(DendrogramNode());
  if (nodes_[a].parent_id != kNoParentId ||
      nodes_[b].parent_id != kNoParentId) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Either node %d or node %d already has a non-empty "
                        "parent (nodes are %d and %d respectively).",
                        a, b, nodes_[a].parent_id, nodes_[b].parent_id));
  }
  nodes_[a] = {new_id, merge_similarity};
  nodes_[b] = {new_id, merge_similarity};
  return new_id;
}

// TODO Understand the internal errors below. These checks seem to
// be getting violated when supplying a dendrogram constructed by hac_clusterer.
absl::StatusOr<bool> Dendrogram::IsMonotone() const {
  for (ParentId node_id = 0; node_id < num_clustered_nodes_; ++node_id) {
    auto [parent_id, parent_merge_similarity] = nodes_[node_id];
    if (parent_id != kNoParentId) {
      auto [grandparent_id, grandparent_merge_similarity] = nodes_[parent_id];
      if (grandparent_id != kNoParentId &&
          !(AlmostEquals(parent_merge_similarity,
                         grandparent_merge_similarity)) &&
          parent_merge_similarity < grandparent_merge_similarity) {
        return false;
      }
      if (parent_id >= grandparent_id) {
        return absl::InternalError(
            absl::StrFormat("Expected parent_id (%d) < grandparent_id (%d)",
                            parent_id, grandparent_id));
      }
    }
    if (node_id >= parent_id) {
      return absl::InternalError(absl::StrFormat(
          "Expected node_id (%d) < parent_id (%d)", node_id, parent_id));
    }
  }
  return true;
}

std::vector<size_t> Dendrogram::GetClusterSizes() const {
  std::vector<size_t> sizes(nodes_.size());
  for (size_t i = 0; i < num_clustered_nodes_; ++i) {
    sizes[i] = 1;
  }
  for (size_t i = 0; i < nodes_.size(); ++i) {
    auto [parent_id, merge_similarity] = nodes_[i];
    if (parent_id != kNoParentId) {
      sizes[parent_id] += sizes[i];
    }
  }
  return sizes;
}

absl::StatusOr<Clustering> Dendrogram::FlattenClustering(
    double linkage_similarity) const {
  std::vector<Clustering> clusterings;
  ASSIGN_OR_RETURN(clusterings,
                   HierarchicalFlattenClustering({linkage_similarity}));
  if (clusterings.size() != 1) {
    return absl::InternalError("Should have exactly 1 clustering");
  }
  return clusterings[0];
}

absl::StatusOr<Clustering> Dendrogram::FullyFlattenClustering() const {
  std::vector<NodeId> nodes(nodes_.size());

  SequentialUnionFind<NodeId> cc(nodes_.size());
  for (ParentId i = 0; i < nodes_.size(); ++i) {
    auto [parent_id, merge_similarity] = nodes_[i];
    if (parent_id != kNoParentId) {
      cc.Unite(i, parent_id);
    }
  }
  auto components_int = cc.ComponentIds();
  std::vector<NodeId> components{components_int.begin(), components_int.end()};
  return ClusterIdSequenceToClustering(
      absl::MakeSpan(components).subspan(0, num_clustered_nodes_));
}

absl::StatusOr<Clustering> Dendrogram::FlattenSubtreeClustering(
    double linkage_similarity) const {
  return PrunedDendrogramWithThreshold(linkage_similarity)
      .FullyFlattenClustering();
}

Dendrogram Dendrogram::PrunedDendrogramWithThreshold(
    double linkage_similarity) const {
  std::vector<double> max_merge(nodes_.size());

  // Store the max similarity merge that forms each node.
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i < num_clustered_nodes_) {
      max_merge[i] = std::numeric_limits<double>::infinity();
    }
    if (nodes_[i].parent_id != kNoParentId) {
      max_merge[nodes_[i].parent_id] =
          std::max(nodes_[i].merge_similarity, max_merge[nodes_[i].parent_id]);
    }
  }
  // Go through the dendrogram in reverse order; max_merge values now store
  // the maximum merge similarity of any ancestor (including the node itself).
  for (int64_t i = nodes_.size() - 1; i >= 0; --i) {
    auto parent_id = nodes_[i].parent_id;
    if (parent_id != kNoParentId) {
      max_merge[i] = std::max(max_merge[i], max_merge[parent_id]);
    }
  }

  // Keep any node that has an ancestor (inclusive) that has merge similarity at
  // least linkage_similarity.
  auto keep = [&](size_t id) { return max_merge[id] >= linkage_similarity; };

  // Build maps from old_id -> new_id and vice versa.
  std::vector<DendrogramNode::ParentId> old_to_new(nodes_.size(),
                                                   Dendrogram::kNoParentId);
  std::vector<DendrogramNode::ParentId> new_to_old(nodes_.size(),
                                                   Dendrogram::kNoParentId);
  size_t new_id = 0;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (keep(i)) {
      new_to_old[new_id] = i;
      old_to_new[i] = new_id;
      ++new_id;
    }
  }

  // Build the pruned dendrogram using the maps above to translate old ids to
  // the new (compacted) ones.
  std::vector<DendrogramNode> pruned_nodes(new_id);
  for (size_t i = 0; i < new_id; ++i) {
    auto& old_node = nodes_[new_to_old[i]];
    if (old_node.parent_id != Dendrogram::kNoParentId) {
      pruned_nodes[i].parent_id = old_to_new[old_node.parent_id];
      pruned_nodes[i].merge_similarity = old_node.merge_similarity;
    }
  }

  Dendrogram result(0);
  ABSL_CHECK_OK(result.Init(std::move(pruned_nodes), num_clustered_nodes_));
  return result;
}

absl::StatusOr<std::vector<Clustering>>
Dendrogram::HierarchicalFlattenClustering(
    const std::vector<double>& similarity_thresholds) const {
  // TODO, understand how the IsMonotone check can be failed when
  // using a dendrogram emitted by hac_clusterer.
  bool is_monotone;
  ASSIGN_OR_RETURN(is_monotone, IsMonotone());
  if (!is_monotone) {
    return absl::FailedPreconditionError(
        "Trying to flatten a non-monotone dendrogram.");
  }

  if (similarity_thresholds.empty()) {
    return absl::InvalidArgumentError(
        "similarity_thresholds must be non-empty.");
  }

  // Enforce that the thresholds are in non-ascending order.
  if (!absl::c_is_sorted(similarity_thresholds, std::greater<double>())) {
    return absl::InvalidArgumentError(
        "similarity_thresholds must be in descending order.");
  }

  SequentialUnionFind<NodeId> cc(nodes_.size());
  std::vector<Clustering> clusterings;
  for (double similarity_threshold : similarity_thresholds) {
    // TODO: Do this more efficiently for the case of many thresholds.
    for (ParentId i = 0; i < nodes_.size(); ++i) {
      auto [parent_id, merge_similarity] = nodes_[i];
      if (parent_id != kNoParentId &&
          merge_similarity >= similarity_threshold) {
        cc.Unite(i, parent_id);
      }
    }

    auto components_int = cc.ComponentIds();
    std::vector<NodeId> components{components_int.begin(),
                                   components_int.end()};
    Clustering clustering;
    ASSIGN_OR_RETURN(
        clustering,
        ClusterIdSequenceToClustering(
            absl::MakeSpan(components).subspan(0, num_clustered_nodes_)));
    clusterings.push_back(clustering);
  }

  return clusterings;
}

std::vector<Dendrogram::ParentId> Dendrogram::GetNumChildren() const {
  std::vector<Dendrogram::ParentId> num_children(nodes_.size(), 0);
  for (const auto& [parent_id, _] : nodes_) {
    if (parent_id != Dendrogram::kNoParentId) {
      num_children[parent_id]++;
    }
  }
  return num_children;
}

std::vector<DendrogramMerge> Dendrogram::GetMergeSequence() const {
  // Topologically sort the merges in the dendrogram and add the merges to
  // merges in decreasing order of merge_similarity.
  std::vector<DendrogramMerge> merges;

  // Compute the number of children of each cluster in the dendrogram.
  const std::vector<ParentId> num_children = GetNumChildren();

  // Create two vectors to store the number of (1) pending children and (2)
  // finished children for each node. Initially all children of every internal
  // node are *pending*. When all merges corresponding to a child are performed,
  // the child is *finished*. Once all of a node's children are finished, the
  // node becomes *ready*. All leaves of the dendrogram are initially ready.
  // Ready nodes are processed using the ProcessReadyNode function below, which
  // decrements the number of pending children of its parent. Also see the
  // comment about children_vecs, defined below.
  std::vector<ParentId> pending_children = num_children;
  std::vector<ParentId> finished_children(2 * num_clustered_nodes_ - 1, 0);

  // Store the children of each node. The first component of each pair is the
  // merge similarity of the child to the parent, and the second component is
  // the current cluster id of the child node, i.e., a value in [0,
  // num_clustered_nodes). Initially the vectors are all empty. A child c of a
  // node p is added to children_vecs[p] once all of c's children are finished,
  // i.e., c's children's merges have been processed by the algorithm and c is
  // now ready. By this definition, all of the leaves are ready, and so we
  // initialize the vectors by processing the leaves, and adding leaf nodes to
  // the nodes' vectors.
  std::vector<std::vector<std::pair<double, ParentId>>> children_vecs(
      2 * num_clustered_nodes_ - 1);

  auto cmp = [](DendrogramMerge left, DendrogramMerge right) {
    return left.merge_similarity < right.merge_similarity;
  };
  std::priority_queue<DendrogramMerge, std::vector<DendrogramMerge>,
                      decltype(cmp)>
      merge_queue(cmp);

  // The initial clustering is just the identity clustering, with the cluster
  // ids of internal dendrogram nodes set to kNoParentId.
  auto cluster_ids =
      std::vector<ParentId>(2 * num_clustered_nodes_ - 1, kNoParentId);
  for (ParentId node_id = 0; node_id < num_clustered_nodes_; ++node_id) {
    cluster_ids[node_id] = node_id;
  }

  // Helper function that tries to add the parent of a ready node, node_id, in
  // the dendrogram to the priority queue, if all of node_id's siblings are also
  // ready.
  auto ProcessReadyNode = [&](ParentId node_id,
                              double min_merge_similarity =
                                  std::numeric_limits<double>::infinity()) {
    auto [parent_id, parent_merge] = nodes_[node_id];
    ABSL_CHECK_EQ(pending_children[node_id],
                  0);  // Verify that the node is ready.
    if (parent_id != Dendrogram::kNoParentId) {
      pending_children[parent_id]--;
      auto& children = children_vecs[parent_id];
      ParentId current_cluster_id = cluster_ids[node_id];
      children.push_back({parent_merge, current_cluster_id});
      if (pending_children[parent_id] == 0) {  // All children in children_vecs.
        // Sort the children merges by weight.
        std::sort(children.begin(), children.end(), std::greater<>());
        // There should be at least two children.
        ABSL_CHECK_GE(children.size(), 2);
        // Check that all children are distinct.
        for (size_t i = 0; i < children.size() - 1; ++i) {
          ABSL_CHECK_NE(children[i].second, children[i + 1].second);
        }
        // Go through the merges in descending order. We merge the first two
        // children together first.
        auto [first_sim, first_child] = children[0];
        auto [second_sim, second_child] = children[1];
        ABSL_CHECK_EQ(first_sim, second_sim);
        auto smallest_id = std::min(first_child, second_child);

        DendrogramMerge merge({first_sim, smallest_id,
                               std::max(first_child, second_child), parent_id,
                               std::min(first_sim, min_merge_similarity)});
        merge_queue.push(merge);

        for (size_t i = 2; i < children.size(); ++i) {
          auto [merge_sim, child_id] = children[i];
          merge_queue.push({merge_sim, std::min(smallest_id, child_id),
                            std::max(smallest_id, child_id), parent_id,
                            std::min(merge_sim, min_merge_similarity)});
          smallest_id = std::min(smallest_id, child_id);
        }
      }
    }
  };

  // Process the leafs of the dendrogram, which are all ready.
  for (ParentId node_id = 0; node_id < num_clustered_nodes_; ++node_id) {
    ProcessReadyNode(node_id);
  }

  while (!merge_queue.empty()) {
    // Extract the next merge to process.
    DendrogramMerge merge = merge_queue.top();
    auto [similarity, node_a, node_b, parent_id, min_merge_similarity] = merge;
    merge_queue.pop();

    // Push this merge onto merges.
    merges.push_back(merge);

    // Apply the merge and compress cluster_ids.
    ABSL_CHECK_LT(node_a, node_b);
    cluster_ids[parent_id] = std::min(cluster_ids[parent_id], node_a);

    // We finished processing one more merge for the internal node, parent_id.
    finished_children[parent_id]++;
    // Check if all of parent_id's children are done. If so, parent_id is now
    // ready.
    if (finished_children[parent_id] == num_children[parent_id] - 1) {
      ABSL_CHECK_EQ(pending_children[parent_id], 0);
      ProcessReadyNode(parent_id, min_merge_similarity);
    }
  }

  return merges;
}

namespace {

// Helper function for GetMergeOrderedNodes. Recursively traverses
// `parent_to_children` in depth first order starting at `node`. Leaf entries
// will be written to the `nodes` result.
void TraverseDendrogram(
    const absl::flat_hash_map<NodeId, std::vector<NodeId>>& parent_to_children,
    NodeId node, std::vector<NodeId>& nodes) {
  const auto& children = parent_to_children.find(node);
  if (children == parent_to_children.end()) {
    nodes.push_back(node);
    return;
  }
  for (NodeId child : children->second) {
    TraverseDendrogram(parent_to_children, child, nodes);
  }
}

}  // namespace

std::vector<NodeId> Dendrogram::GetMergeOrderedNodes() const {
  std::vector<NodeId> nodes;
  if (Nodes().empty()) {
    return nodes;
  }
  // Store child IDs per node. The special NodeId kNoParentId denotes the root
  // node.
  absl::flat_hash_map<NodeId, std::vector<NodeId>> parent_to_children;
  for (NodeId node_id = 0; node_id < Nodes().size(); ++node_id) {
    const DendrogramNode& node = Nodes()[node_id];
    parent_to_children[node.parent_id].push_back(node_id);
  }

  nodes.reserve(num_clustered_nodes_);  // One entry per leaf node.
  TraverseDendrogram(parent_to_children, kNoParentId, nodes);
  return nodes;
}

}  // namespace graph_mining::in_memory
