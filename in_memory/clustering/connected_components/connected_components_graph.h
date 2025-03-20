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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/status_macros.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

// Graph implementation used by `ParallelConnectedComponentsClusterer` (and
// tested indirectly through the unit tests of that class). Any public method
// other than `PrepareImport` requires that `PrepareImport` has been called.
class ConnectedComponentsGraph : public InMemoryClusterer::Graph {
 public:
  absl::Status PrepareImport(int64_t num_nodes) override {
    if (union_find_parent_.has_value())
      return absl::FailedPreconditionError("PrepareImport called twice");
    if (num_nodes > std::numeric_limits<NodeId>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Total number of nodes exceeds limit: ", num_nodes));
    }
    if (num_nodes < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Total number of nodes cannot be negative: ", num_nodes));
    }
    union_find_parent_ = AsynchronousUnionFind<NodeId>(num_nodes);
    return absl::OkStatus();
  }

  absl::Status Import(AdjacencyList adjacency_list) override {
    RETURN_IF_ERROR(ValidateThatPrepareImportWasCalled());
    RETURN_IF_ERROR(ValidateThatFinishImportWasNotCalled());
    RETURN_IF_ERROR(ValidateNodeId(adjacency_list.id));
    for (const auto& [neighbor_id, weight] : adjacency_list.outgoing_edges) {
      RETURN_IF_ERROR(ValidateNodeId(neighbor_id));
      union_find_parent_->Unite(adjacency_list.id, neighbor_id);
    }
    return absl::OkStatus();
  }

  absl::Status FinishImport() override {
    RETURN_IF_ERROR(ValidateThatPrepareImportWasCalled());
    RETURN_IF_ERROR(ValidateThatFinishImportWasNotCalled());
    finished_parents_ = std::move(*union_find_parent_).ComponentSequence();
    return absl::OkStatus();
  }

  absl::StatusOr<absl::Span<const NodeId>> ParentArray() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    RETURN_IF_ERROR(ValidateThatPrepareImportWasCalled());
    if (!finished_parents_.has_value()) {
      return absl::FailedPreconditionError(
          "ParentArray cannot be called before FinishImport.");
    }
    return absl::MakeSpan(*finished_parents_);
  }

 private:
  absl::Status ValidateThatPrepareImportWasCalled() const {
    if (!union_find_parent_.has_value()) {
      return absl::FailedPreconditionError(
          "Using ConnectedComponents requires calling PrepareImport.");
    }
    return absl::OkStatus();
  }

  absl::Status ValidateThatFinishImportWasNotCalled() const {
    if (finished_parents_.has_value()) {
      return absl::FailedPreconditionError("FinishImport already called.");
    }
    return absl::OkStatus();
  }

  // Returns a status indicating whether `node_id` is a valid node ID. Assumes
  // that `PrepareImport` has been called.
  absl::Status ValidateNodeId(const NodeId node_id) const {
    if (node_id < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Node IDs must be non-negative; found ", node_id));
    } else if (node_id >= union_find_parent_->NumberOfNodes()) {
      return absl::OutOfRangeError(absl::StrCat(
          "Node IDs must be in the range [0, ",
          union_find_parent_->NumberOfNodes(), "); found ", node_id));
    }
    return absl::OkStatus();
  }

  // Initialized in `PrepareImport`.
  std::optional<AsynchronousUnionFind<NodeId>> union_find_parent_;
  // Initialized in `FinishImport`.
  std::optional<parlay::sequence<NodeId>> finished_parents_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CONNECTED_COMPONENTS_CONNECTED_COMPONENTS_GRAPH_H_
