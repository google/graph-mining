#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TREE_PARTITIONER_MIN_SIZE_TREE_PARTITIONING_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TREE_PARTITIONER_MIN_SIZE_TREE_PARTITIONING_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

// Outputs a min weighted size partitioned forest represented by parent ids of
// each node. The input is a forest of nodes with ids from 0 ... n - 1. The
// parent node id of each node is given by a length-n vector parent_ids. A node
// is a root iff its parent node id is -1. Each node is associated with a
// non-negative weight given by a length-n vector node-weights. The goal is to
// run min weighted size partitioning over the forest with min weight threshold.
// The implementation follows the design of the in-memory algorithm proposed in
// go/min-size-tree-partitioner-design.
// This function returns an error if any of the following are true:
// 1) The sizes of parent_ids and node_weights are inconsistent.
// 2) Invalid parent_ids, i.e., parent_ids does not represent a rooted forest
// with nodes with ids 0 ... n - 1.
// 3) There exists a negative node weight.
// 4) min_weight_threshold is negative.
absl::StatusOr<std::vector<graph_mining::in_memory::NodeId>>
MinWeightedSizeTreePartitioning(
    const std::vector<graph_mining::in_memory::NodeId>& parent_ids,
    const std::vector<double>& node_weights, double min_weight_threshold);

// TODO: add a function doing the same thing as
// MinWeightedSizeTreePartitioning except rounding the weights to a particular
// precision at the beginning.

// The functions in the following namespace should only be used internally. They
// are declared in .h file for test purpose only.
namespace internal {

struct SubtreeInformation {
  std::vector<std::vector<graph_mining::in_memory::NodeId>> children;
  std::vector<double> subtree_weights;
};

// Given a rooted forest represented by parent ids of each node where each node
// has a non-negative weight, computes the list of children and the total
// subtree weight of each node.
absl::StatusOr<SubtreeInformation> ProcessChildrenAndSubtreeWeights(
    const std::vector<graph_mining::in_memory::NodeId>& parent_ids,
    const std::vector<double>& node_weights);

// Given a list of nodes with non-negative weights where the total weight is at
// least min_weight_threshold, outputs a clustering represented by a cluster id
// map such that each cluster has weight at least min_weight_threshold. In
// addition, each cluster either has weight at most 4 * min_weight_threshold or
// has a weight at most 2 * min_weight_threshold after removing the node with
// the largest weight. Note: each cluster id can only be a node id of a node in
// the cluster. In addition, if a cluster contains a node with root_id, the
// cluster id must be root_id.
std::vector<
    std::pair<graph_mining::in_memory::NodeId, graph_mining::in_memory::NodeId>>
PartitionClusters(
    const std::vector<std::pair<graph_mining::in_memory::NodeId, double>>&
        nodes_with_unassigned_weights,
    double min_weight_threshold,
    const graph_mining::in_memory::NodeId& root_id);

}  // namespace internal

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_TREE_PARTITIONER_MIN_SIZE_TREE_PARTITIONING_H_
