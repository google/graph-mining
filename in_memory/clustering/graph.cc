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

#include "in_memory/clustering/graph.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"

// TODO: relax the use of this optimization after we enable
// resizing the graph down to the actual size during/after FinishImport.
ABSL_FLAG(bool, lock_free_import, false,
          "Whether to use lock-free graph building during import for "
          "Simple{Directed,Undirected}Graph. This can be enabled only if the "
          "number of nodes is known prior to the graph import process.");

namespace graph_mining {
namespace in_memory {

using NodeId = InMemoryClusterer::NodeId;

static constexpr double kDefaultNodeWeight = 1.0;
static constexpr int32_t kDefaultNodePartId = 0;

absl::Status MultipleGraphs::PrepareImport(int64_t num_nodes) {
  for (InMemoryClusterer::Graph* graph : graphs_) {
    RETURN_IF_ERROR(graph->PrepareImport(num_nodes));
  }
  return absl::OkStatus();
}

absl::Status MultipleGraphs::Import(AdjacencyList adjacency_list) {
  for (InMemoryClusterer::Graph* graph : graphs_) {
    RETURN_IF_ERROR(graph->Import(adjacency_list));
  }
  return absl::OkStatus();
}

absl::Status MultipleGraphs::FinishImport() {
  for (InMemoryClusterer::Graph* graph : graphs_) {
    RETURN_IF_ERROR(graph->FinishImport());
  }
  return absl::OkStatus();
}

absl::Status SimpleDirectedGraph::PrepareImport(int64_t num_nodes) {
  lock_free_import_ = absl::GetFlag(FLAGS_lock_free_import);
  if (lock_free_import_) {
    adjacency_lists_.resize(num_nodes);
    node_weights_.resize(num_nodes, kDefaultNodeWeight);
    node_parts_.resize(num_nodes, kDefaultNodePartId);
  }
  return absl::OkStatus();
}

absl::Status SimpleDirectedGraph::ImportHelper(AdjacencyList adjacency_list) {
  SetNodeWeight(adjacency_list.id, adjacency_list.weight);
  if (adjacency_list.part.has_value()) {
    SetNodePart(adjacency_list.id, *adjacency_list.part);
  }
  for (auto [neighbor_id, weight] : adjacency_list.outgoing_edges) {
    RETURN_IF_ERROR(AddEdge(adjacency_list.id, neighbor_id, weight));
  }
  return absl::OkStatus();
}

absl::Status SimpleDirectedGraph::Import(AdjacencyList adjacency_list) {
  if (lock_free_import_) {
    return ImportHelper(std::move(adjacency_list));
  } else {
    absl::MutexLock lock(&mutex_);
    return ImportHelper(std::move(adjacency_list));
  }
}

absl::Status SimpleDirectedGraph::FinishImport() {
  if (lock_free_import_) {
    // We are done with importing. Disable the lock-free optimization.
    lock_free_import_ = false;
  }

  MaybeClearNodeWeights();
  MaybeClearNodeParts();

  return absl::OkStatus();
}

NodeId SimpleDirectedGraph::AddNode() {
  adjacency_lists_.emplace_back();
  return adjacency_lists_.size() - 1;
}

void SimpleDirectedGraph::SetNumNodes(NodeId num_nodes) {
  ABSL_CHECK_GE(num_nodes, adjacency_lists_.size());
  adjacency_lists_.resize(num_nodes);
}

absl::Status SimpleDirectedGraph::AddEdge(NodeId from_node, NodeId to_node,
                                          double weight) {
  RETURN_IF_ERROR(CheckNodeId(from_node));
  RETURN_IF_ERROR(CheckNodeId(to_node));

  double* existing_weight = MutableEdgeWeight(from_node, to_node, weight);
  *existing_weight = std::max(*existing_weight, weight);
  return absl::OkStatus();
}

absl::Status SimpleDirectedGraph::SetEdgeWeight(NodeId from_node,
                                                NodeId to_node, double weight) {
  RETURN_IF_ERROR(CheckNodeId(from_node));
  RETURN_IF_ERROR(CheckNodeId(to_node));
  *MutableEdgeWeight(from_node, to_node, 0.0) = weight;
  return absl::OkStatus();
}

std::optional<double> SimpleDirectedGraph::EdgeWeight(NodeId from_node,
                                                      NodeId to_node) const {
  if (!HasNode(from_node) || !HasNode(to_node)) return {};
  auto it = adjacency_lists_[from_node].find(to_node);
  if (it == adjacency_lists_[from_node].end())
    return {};
  else
    return it->second;
}

double SimpleDirectedGraph::NodeWeight(NodeId id) const {
  ABSL_CHECK_OK(CheckNodeId(id));
  return id < node_weights_.size() ? node_weights_[id] : kDefaultNodeWeight;
}

int32_t SimpleDirectedGraph::NodePart(NodeId id) const {
  ABSL_CHECK_OK(CheckNodeId(id));
  return id < node_parts_.size() ? node_parts_[id] : kDefaultNodePartId;
}

bool SimpleDirectedGraph::IsUnipartite() const {
  for (const int32_t node_part : node_parts_) {
    if (node_part != kDefaultNodePartId) return false;
  }
  return true;
}

double SimpleDirectedGraph::WeightedOutDegree(NodeId id) const {
  ABSL_CHECK_OK(CheckNodeId(id));
  double result = 0;
  for (const auto& [_, weight] : Neighbors(id)) {
    result += weight;
  }
  return result;
}

void SimpleDirectedGraph::SetNodeWeight(NodeId id, double weight) {
  ABSL_CHECK_OK(CheckNodeId(id));
  if (lock_free_import_) {
    // In the lock-free import logic, adjacency_lists_ and node_weights_ are
    // preallocated to the same size. Thus checking against the size of
    // adjacency_lists_ is sufficient.
    ABSL_CHECK_LT(id, adjacency_lists_.size());
  } else {
    EnsureSize(id + 1);
    if (id >= node_weights_.size()) {
      node_weights_.resize(id + 1, kDefaultNodeWeight);
    }
  }
  node_weights_[id] = weight;
}

void SimpleDirectedGraph::SetNodePart(NodeId id, int32_t part) {
  ABSL_CHECK_OK(CheckNodeId(id));
  if (lock_free_import_) {
    // In the lock-free import logic, adjacency_lists_ and node_parts_ are
    // preallocated to the same size. Thus checking against the size of
    // adjacency_lists_ is sufficient.
    ABSL_CHECK_LT(id, adjacency_lists_.size());
  } else {
    EnsureSize(id + 1);
    if (id >= node_parts_.size()) {
      node_parts_.resize(id + 1, kDefaultNodePartId);
    }
  }
  node_parts_[id] = part;
}

double* SimpleDirectedGraph::MutableEdgeWeight(NodeId from_node, NodeId to_node,
                                               double default_weight) {
  if (lock_free_import_) {
    ABSL_CHECK_LT(std::max(from_node, to_node), adjacency_lists_.size());
  } else {
    EnsureSize(std::max(from_node, to_node) + 1);
  }
  auto [iterator, _] =
      adjacency_lists_[from_node].insert({to_node, default_weight});
  return &iterator->second;
}

void SimpleDirectedGraph::EnsureSize(NodeId id) {
  if (adjacency_lists_.size() < id) {
    adjacency_lists_.resize(id);
  }
}

absl::Status SimpleDirectedGraph::CheckNodeId(NodeId id) const {
  if (id < 0)
    return absl::InvalidArgumentError(absl::StrCat("id < 0: id = ", id));
  return absl::OkStatus();
}

absl::Status SimpleDirectedGraph::ClearNeighbors(NodeId id) {
  RETURN_IF_ERROR(CheckNodeId(id));
  if (adjacency_lists_.size() <= id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "id: ", id, " >= adjacency_lists_ size: ", adjacency_lists_.size()));
  }
  // Free the capacity of adjacency list.
  absl::flat_hash_map<NodeId, double>().swap(adjacency_lists_[id]);
  return absl::OkStatus();
}

// Clears v if all values in v equal to default_value.
// Returns true if v is cleared.
template <typename T>
bool MaybeClearVector(std::vector<T>& v, T default_value) {
  bool should_clear = true;
  for (const auto i : v) {
    if (i != default_value) {
      should_clear = false;
      break;
    }
  }

  if (should_clear) {
    std::vector<T>().swap(v);
  }

  return should_clear;
}

bool SimpleDirectedGraph::MaybeClearNodeWeights() {
  return MaybeClearVector(node_weights_, kDefaultNodeWeight);
}

bool SimpleDirectedGraph::MaybeClearNodeParts() {
  return MaybeClearVector(node_parts_, kDefaultNodePartId);
}

absl::Status SimpleUndirectedGraph::AddEdge(NodeId from_node, NodeId to_node,
                                            double weight) {
  if (!per_node_lock_.empty() &&
      per_node_lock_.size() <=
          std::max(from_node, to_node) / kLockGranularity) {
    return absl::InvalidArgumentError(
        "node id exceeds preallocated lock range.");
  }

  absl::Status status = absl::OkStatus();
  if (per_node_lock_.empty()) {
    status = SimpleDirectedGraph::AddEdge(from_node, to_node, weight);
  } else {
    // Fine-grained lock during import.
    absl::MutexLock lock(&per_node_lock_[from_node / kLockGranularity]);
    status = SimpleDirectedGraph::AddEdge(from_node, to_node, weight);
  }
  if (!status.ok()) return status;
  if (to_node != from_node) {
    // SimpleDirectedGraph::AddEdge only fails for invalid indices, so adding
    // the reverse edge should succeed.
    if (per_node_lock_.empty()) {
      ABSL_CHECK_OK(SimpleDirectedGraph::AddEdge(to_node, from_node, weight));
    } else {
      // Fine-grained lock during import.
      absl::MutexLock lock(&per_node_lock_[to_node / kLockGranularity]);
      ABSL_CHECK_OK(SimpleDirectedGraph::AddEdge(to_node, from_node, weight));
    }
  }

  return absl::OkStatus();
}

absl::Status SimpleUndirectedGraph::SetEdgeWeight(NodeId from_node,
                                                  NodeId to_node,
                                                  double weight) {
  auto status = SimpleDirectedGraph::SetEdgeWeight(from_node, to_node, weight);
  if (!status.ok()) return status;
  if (to_node != from_node)
    // SimpleDirectedGraph::SetEdgeWeight only fails for invalid indices, so
    // changing the reverse edge should succeed.
    ABSL_CHECK_OK(
        SimpleDirectedGraph::SetEdgeWeight(to_node, from_node, weight));
  return absl::OkStatus();
}

absl::Status SimpleUndirectedGraph::PrepareImport(int64_t num_nodes) {
  if (absl::GetFlag(FLAGS_lock_free_import)) {
    int64_t num_buckets = (num_nodes + kLockGranularity - 1) / kLockGranularity;
    per_node_lock_ = std::vector<absl::Mutex>(num_buckets);
  }
  return SimpleDirectedGraph::PrepareImport(num_nodes);
}

absl::Status SimpleUndirectedGraph::FinishImport() {
  if (absl::GetFlag(FLAGS_lock_free_import)) {
    // Destroy the per-node lock vector because we are done with import.
    std::vector<absl::Mutex>().swap(per_node_lock_);
  }
  return SimpleDirectedGraph::FinishImport();
}

absl::Status CopyGraph(const SimpleDirectedGraph& in_graph,
                       InMemoryClusterer::Graph* out_graph) {
  RETURN_IF_ERROR(out_graph->PrepareImport(in_graph.NumNodes()));
  for (NodeId id = 0; id < in_graph.NumNodes(); id++) {
    InMemoryClusterer::AdjacencyList adjacency_list;
    adjacency_list.id = id;
    adjacency_list.weight = in_graph.NodeWeight(id);
    const auto node_part = in_graph.NodePart(id);
    if (node_part != kDefaultNodePartId) {
      adjacency_list.part = in_graph.NodePart(id);
    }
    const auto& neighbors = in_graph.Neighbors(id);
    adjacency_list.outgoing_edges.reserve(neighbors.size());
    for (auto [neighbor_id, weight] : neighbors) {
      adjacency_list.outgoing_edges.emplace_back(neighbor_id, weight);
    }
    RETURN_IF_ERROR(out_graph->Import(std::move(adjacency_list)));
  }
  return out_graph->FinishImport();
}

}  // namespace in_memory
}  // namespace graph_mining
