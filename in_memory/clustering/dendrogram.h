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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_H_

#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Represents the merge information for a node (a leaf or an internal node) in a
// dendrogram. See the comment for Dendrogram below.
struct DendrogramNode {
 public:
  // The numerical type of a parent id.
  using ParentId = NodeId;

  // Default NodeId representing an invalid parent.
  static constexpr ParentId kNoParentId = std::numeric_limits<ParentId>::max();

  DendrogramNode();
  DendrogramNode(ParentId parent_id, double merge_similarity);

  // The id of the node's parent, or kNoParentId if the node is a root.
  ParentId parent_id;
  // The similarity of the merge to the node's parent.
  double merge_similarity;
};

// Structure representing a binary merge. Note that although an internal node
// can be a merge of k >= 2 children, a DendrogramMerge represents a merge of
// just two nodes (aka binary merge).
struct DendrogramMerge {
 public:
  // The numerical type of a parent id.
  using ParentId = DendrogramNode::ParentId;

  DendrogramMerge();
  DendrogramMerge(double merge_similarity, ParentId node_a, ParentId node_b,
                  ParentId parent_id, double min_merge_similarity);

  // The merge similarity of this merge.
  double merge_similarity;
  // Node id of the first endpoint, in [0, num_clustered_nodes), i.e., a graph
  // node.
  ParentId node_a;
  // Node id of the second endpoint, in [0, num_clustered_nodes), i.e., a graph
  // node.
  ParentId node_b;
  // The id of the node's parent in the dendrogram (a value >=
  // num_clustered_nodes).
  ParentId parent_id;
  // The minimum merge similarity among all previous merges that contributed to
  // this merge.
  double min_merge_similarity;
};

// Represents a dendrogram in the parent-array format. i.e., a vector of length
// <= 2*num_clustered_nodes-1 containing, for each vertex, the id of its parent
// cluster and a floating point value indicating the similarity of the merge
// (the merge_similarity). Ids in [0, num_clustered_nodes) represent the leaves
// of the dendrogram, and ids in [num_clustered_nodes, 2*num_clustered_nodes-1)
// represent internal nodes. Note that the dendrogram may consist of multiple
// trees. In an extreme case, the dendrogram has num_clustered_nodes nodes, and
// all of them are both leaves and roots.
//
// The dendrogram is guaranteed to be in a valid state and satisfy the following
// assumptions:
// 1. The fanout of each internal node (non-leaf node) in the dendrogram is
//    >= 2. Internal nodes with a fanout of 1 (i.e., renaming a cluster) are not
//    supported.
// 2. For internal nodes with fanout 2, the merge similarities of both children
//    to the parent are identical.
// 3. For internal nodes with fanout > 2, consider the sorted sequence of merge
//    similarities of the children. The two largest values in this sequence are
//    identical.
// 4. There are no isolated internal nodes (an isolated internal node is a node
//    with id >= num_clustered_nodes that has no children).
// 5. All merge similarities are strictly positive.
// 6. For each non-root node i, we have parent_id > i.
class Dendrogram {
 public:
  using ParentId = DendrogramNode::ParentId;

  // Default NodeId representing an invalid parent.
  static constexpr ParentId kNoParentId = DendrogramNode::kNoParentId;

  // Construct a dendrogram consisting of num_clustered_nodes leaves.
  Dendrogram(size_t num_clustered_nodes);

  // Validates the provided nodes array wrt assumptions provided in the comments
  // above this class declaration. If the validation passes, the dendrogram is
  // replaced with the one described by the provided array.
  absl::Status Init(std::vector<DendrogramNode> nodes,
                    size_t num_clustered_nodes);

  // Perform a binary merge of nodes a and b into a new internal node with
  // similarity merge_similarity. The return value is the id of the new internal
  // node (a value in [n, 2*n-1)).
  absl::StatusOr<ParentId> BinaryMerge(ParentId a, ParentId b,
                                       double merge_similarity);

  // Returns true if the dendrogram is *monotone*, i.e., the similarities along
  // all leaf-to-root paths are non-increasing.
  // This function only returns a non-OK status in case of a bug.
  absl::StatusOr<bool> IsMonotone() const;

  // Returns the sequences of merges in decreasing order of merge similarity.
  // Each merge consists of two *original* ids in the graph, i.e., values in [0,
  // num_clustered_nodes), which are two representative nodes from the two
  // clusters being merged by this merge. The function guarantees that these
  // representatives have the smallest id out of all nodes contained in each
  // cluster. Furthermore, the first node (node_a) is guaranteed to have a
  // smaller id than the second (node_b).
  //
  // Note that this function works on non-monotone dendrograms. For non-monotone
  // dendrograms, the merge similarities of merges in the returned sequence is
  // not necessarily decreasing.
  std::vector<DendrogramMerge> GetMergeSequence() const;

  // The dendrogram implies an order on the set of nodes. This may be a useful
  // order if you want to sort a similarity matrix
  // (https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html).
  // The returned vector contains node IDs of the original graph in the order
  // they got merged in the dendrogram, i.e. grouping similar nodes together.
  // This effectively produces a linear embedding of the graph, similar to
  // http://google3/third_party/graph_mining/in_memory/clustering/parline/affinity_hierarchy_embedder.h
  std::vector<NodeId> GetMergeOrderedNodes() const;

  // Returns a vector containing the cluster size of each node. The size of a
  // leaf node is 1, and the size of an internal node is the sum of the sizes of
  // its two children.
  std::vector<size_t> GetClusterSizes() const;

  // Flatten a dendrogram by ignoring all merges with similarity below
  // linkage_similarity, and computing the connected components induced by the
  // remaining merges. Note that this function currently assumes (and checks)
  // that the current dendrogram is monotone.
  absl::StatusOr<Clustering> FlattenClustering(double linkage_similarity) const;

  // Fully flatten the dendrogram by returning the clustering corresponding to
  // the connected components induced by all merges. This is similar to
  // calling FlattenClustering with linkage_similarity -infty, but note that
  // this function also applies to non-monotone dendrograms.
  absl::StatusOr<Clustering> FullyFlattenClustering() const;

  // Same as FlattenClustering, except that it outputs a vector of clusterings,
  // one for each similarity threshold given in |similarity_thresholds|. This is
  // equivalent to calling FlattenClustering repeatedly for each threshold in
  // |similarity_thresholds|, and outputing the vector of results in the same
  // order as given in |similarity_thresholds|. The input
  // |similarity_thresholds| must be monontonically non-increasing, or else an
  // error is returned.
  absl::StatusOr<std::vector<Clustering>> HierarchicalFlattenClustering(
      const std::vector<double>& similarity_thresholds) const;

  // Given a linkage (similarity) threshold, the algorithm produces a new
  // dendrogram where all nodes containing an ancestor (inclusive) with merge
  // similarity at least the linkage threshold are preserved. This method
  // also applies to non-monotone dendrograms.
  Dendrogram PrunedDendrogramWithThreshold(double linkage_similarity) const;

  // This method is relevant if the dendrogram potentially has non-monotone
  // leaf-to-root paths. Here, given a linkage (similarity) threshold, the
  // algorithm produces a flat clustering where each cluster is guaranteed to
  // be a subtree of the dendrogram. The algorithm works by assigning the
  // cluster of each node to be the last node along its leaf-to-root path
  // which has a merge similarity that is at least the linkage threshold.
  absl::StatusOr<Clustering> FlattenSubtreeClustering(
      double linkage_similarity) const;

  // The number of nodes that are clustered in this dendrogram (i.e. the
  // number of dendrogram leaves). This should not be confused with the total
  // number of nodes in the dendrogram.
  size_t NumClusteredNodes() const { return num_clustered_nodes_; }

  const std::vector<DendrogramNode>& Nodes() const { return nodes_; }

  DendrogramNode GetParent(NodeId id) const { return nodes_[id]; }

  bool HasValidParent(NodeId id) const {
    return GetParent(id).parent_id != kNoParentId;
  }

 private:
  std::vector<Dendrogram::ParentId> GetNumChildren() const;

  std::vector<DendrogramNode> nodes_;
  size_t num_clustered_nodes_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_H_
