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

#include "in_memory/clustering/undirected_converter_graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gbbs/bridge.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/undirected_converter_graph.pb.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/streaming_writer.h"
#include "in_memory/status_macros.h"
#include "utils/status/thread_safe_status.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

namespace graph_mining {
namespace in_memory {

constexpr int kPerThreadBufferSize = 1024 * 1024;

UndirectedConverterGraph::UndirectedConverterGraph(
    const ::graph_mining::ConvertToUndirectedConfig& config,
    InMemoryClusterer::Graph* out_graph)
    : config_(config),
      edge_buffer_(kPerThreadBufferSize),
      node_buffer_(kPerThreadBufferSize),
      out_graph_(out_graph) {}

absl::Status UndirectedConverterGraph::Import(AdjacencyList adjacency_list) {
  if (adjacency_list.weight == std::numeric_limits<double>::infinity()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input adjacency_list node weight is infinity for node ",
                     adjacency_list.id));
  }

  node_buffer_.Add({adjacency_list.id, adjacency_list.weight});
  for (auto [neighbor_id, weight] : adjacency_list.outgoing_edges) {
    edge_buffer_.Add({adjacency_list.id, neighbor_id, weight});
    edge_buffer_.Add({neighbor_id, adjacency_list.id, weight});
  }
  return absl::OkStatus();
}

absl::Status UndirectedConverterGraph::FinishImport() {
  // TODO: Add a factory method and move the validation to an
  // earlier stage.
  if (config_.asymmetric_edge_treatment() !=
          graph_mining::ConvertToUndirectedConfig::MAX &&
      config_.asymmetric_edge_treatment() !=
          graph_mining::ConvertToUndirectedConfig::MIN) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported ConvertToUndirectedConfig: ",
                                ""));
  }

  // Sort edges before nodes. This is because the `nodes_` array at this stage
  // contains explicit nodes only. That is, `nodes_` does *not* contain a node
  // id if that node is added only as a neighbor node of an explicit node during
  // `Import`. We thus need to sort `edges_` first in order to identify the full
  // range of node ids.

  // Sort the edges according to source node id, target node id, and edge
  // weight. Note that in order for the subsequent parlay::unique call (which
  // keeps the first edge between a source-target node pair) to work correctly,
  // we rank the edges from the same source-target node pair according to the
  // `asymmetric_edge_treatment` configuration.
  edges_ = Flatten(edge_buffer_.Build());
  parlay::sample_sort_inplace(
      parlay::make_slice(edges_),
      [this](std::tuple<gbbs::uintE, gbbs::uintE, double> a,
             std::tuple<gbbs::uintE, gbbs::uintE, double> b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) <
                   std::tie(std::get<0>(b), std::get<1>(b)) ||
               (std::tie(std::get<0>(a), std::get<1>(a)) ==
                    std::tie(std::get<0>(b), std::get<1>(b)) &&
                (config_.asymmetric_edge_treatment() ==
                         graph_mining::ConvertToUndirectedConfig::MAX
                     ? std::get<2>(a) > std::get<2>(b)
                     : std::get<2>(a) < std::get<2>(b)));
      });

  edges_ = parlay::unique(parlay::make_slice(edges_),
                          [](std::tuple<gbbs::uintE, gbbs::uintE, double> a,
                             std::tuple<gbbs::uintE, gbbs::uintE, double> b) {
                            return std::get<0>(a) == std::get<0>(b) &&
                                   std::get<1>(a) == std::get<1>(b);
                          });

  // Sparify the graph.
  if (config_.has_sparsify()) {
    RETURN_IF_ERROR(Sparsify());
  }

  // Sort nodes.
  if (!edges_.empty()) {
    // Given that we do not know whether each node has already been added
    // explicitly during Import, we artificially add each node once with an
    // infinite weight. If a node weight is explicitly added, then the
    // artificially added weight will be removed by unique. Otherwise, the
    // artificially added weight will be reset to
    // AdjacencyList::kDefaultNodeWeight.
    //
    // Note that we cannot set the weight directly to
    // AdjacencyList::kDefaultNodeWeight in this step because otherwise there is
    // no guarantee that we can keep the user-provided weight by sort+unique.
    parlay::parallel_for(0, std::get<0>(edges_.back()) + 1, [&](std::size_t i) {
      node_buffer_.Add({i, std::numeric_limits<double>::infinity()});
    });
  }

  nodes_ = Flatten(node_buffer_.Build());

  parlay::sample_sort_inplace(
      parlay::make_slice(nodes_),
      [](std::tuple<gbbs::uintE, double> a, std::tuple<gbbs::uintE, double> b) {
        return a < b;
      });
  nodes_ = parlay::unique(
      parlay::make_slice(nodes_),
      [](std::tuple<gbbs::uintE, double> a, std::tuple<gbbs::uintE, double> b) {
        return std::get<0>(a) == std::get<0>(b);
      });

  // For the remaining nodes with infinite node weight, reset that to
  // AdjacencyList::kDefaultNodeWeight to remain compatible with AdjacencyList.
  parlay::parallel_for(0, nodes_.size(), [this](std::size_t i) {
    ABSL_CHECK_EQ(std::get<0>(nodes_[i]), i);
    auto& weight = std::get<1>(nodes_[i]);
    if (weight == std::numeric_limits<double>::infinity()) {
      weight = AdjacencyList::kDefaultNodeWeight;
    }
  });

  offsets_ = GetOffsets(
      [this](std::size_t i) -> gbbs::uintE { return std::get<0>(edges_[i]); },
      edges_.size(), nodes_.size());

  // The following is an implementation check on GetOffsets.
  ABSL_CHECK_EQ(offsets_.size(), nodes_.size() + 1);

  return CopyGraph();
}

// CopyGraph is implemented via the importer interface (i.e., PrepareImport -->
// Import --> FinishImport). We trade performance for extensibility of the API
// and intentionally avoid in-place construction of the underlying GbbsGraph
// object using graph_mining::in_memory::MakeGbbsGraph().
absl::Status UndirectedConverterGraph::CopyGraph() const {
  RETURN_IF_ERROR(out_graph_->PrepareImport(nodes_.size()));

  ThreadSafeStatus import_status;
  parlay::parallel_for(0, nodes_.size(), [&](std::size_t i) {
    AdjacencyList adjacency_list;
    adjacency_list.id = i;
    adjacency_list.weight = std::get<1>(nodes_[i]);
    ABSL_CHECK_LE(offsets_[i + 1], edges_.size());
    adjacency_list.outgoing_edges.reserve(offsets_[i + 1] - offsets_[i]);
    for (auto j = offsets_[i]; j < offsets_[i + 1]; ++j) {
      adjacency_list.outgoing_edges.emplace_back(std::get<1>(edges_[j]),
                                                 std::get<2>(edges_[j]));
    }
    import_status.Update(out_graph_->Import(std::move(adjacency_list)));
  });
  RETURN_IF_ERROR(import_status.status());

  return out_graph_->FinishImport();
}

namespace {

// Sparsifies the edge array.
//
// This function assumes that the input `edges` contains edges of an undirected
// graph.
parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, double>> SparsifyHelper(
    parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, double>>&& edges,
    const int32_t degree_threshold, const bool remove_one_sided_edges) {
  // Sort the edge array by source id, edge weight, and neighbor id.
  parlay::sample_sort_inplace(
      parlay::make_slice(edges),
      [](std::tuple<gbbs::uintE, gbbs::uintE, double> a,
         std::tuple<gbbs::uintE, gbbs::uintE, double> b) {
        return std::forward_as_tuple(std::get<0>(a), -std::get<2>(a),
                                     std::get<1>(a)) <
               std::forward_as_tuple(std::get<0>(b), -std::get<2>(b),
                                     std::get<1>(b));
      });

  gbbs::uintE num_nodes = std::get<0>(edges.back()) + 1;
  auto offsets = GetOffsets(
      [&edges](std::size_t i) -> gbbs::uintE { return std::get<0>(edges[i]); },
      edges.size(), num_nodes);
  ABSL_CHECK_EQ(offsets.size(), num_nodes + 1);

  // deleted_edges maintains the directed edge removal decision per source node.
  std::vector<absl::flat_hash_set<gbbs::uintE>> deleted_edges(num_nodes);

  parlay::parallel_for(
      0, num_nodes,
      [&edges, &deleted_edges, &offsets, degree_threshold](gbbs::uintE i) {
        std::size_t num_neighbors = offsets[i + 1] - offsets[i];
        std::size_t num_allowed_neighbors =
            std::min(num_neighbors, static_cast<std::size_t>(degree_threshold));
        deleted_edges[i].reserve(num_neighbors - num_allowed_neighbors);
        for (std::size_t j = num_allowed_neighbors; j < num_neighbors; ++j) {
          const auto& edge = edges[offsets[i] + j];
          deleted_edges[i].insert(std::get<1>(edge));
        }
      });

  // Trim edges.
  return parlay::filter(
      edges, [&deleted_edges, remove_one_sided_edges](
                 const std::tuple<gbbs::uintE, gbbs::uintE, double>& edge) {
        if (remove_one_sided_edges) {
          // Keep an edge if both end points decide to keep it.
          return !deleted_edges[std::get<0>(edge)].contains(
                     std::get<1>(edge)) &&
                 !deleted_edges[std::get<1>(edge)].contains(std::get<0>(edge));
        } else {
          // Keep an edge if either end point decides to keep it.
          return !(
              deleted_edges[std::get<0>(edge)].contains(std::get<1>(edge)) &&
              deleted_edges[std::get<1>(edge)].contains(std::get<0>(edge)));
        }
      });
}

}  // namespace

absl::Status UndirectedConverterGraph::Sparsify() {
  if (!config_.sparsify().has_soft_degree_threshold() &&
      !config_.sparsify().has_hard_degree_threshold()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid sparification configuration: either soft or hard "
                     "degree threshold must be specified: ",
                                ""));
  }

  if ((config_.sparsify().has_soft_degree_threshold() &&
       config_.sparsify().soft_degree_threshold() <= 0) ||
      (config_.sparsify().has_hard_degree_threshold() &&
       config_.sparsify().hard_degree_threshold() < 0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid sparification configuration: degree threshold "
                     "must be positive: ",
                                ""));
  }

  if (config_.sparsify().keep_lowest_weight_edges()) {
    return absl::UnimplementedError(
        "keep_lowest_weight_edges is not yet supported in parallel graph "
        "sparsification.");
  }

  if (config_.sparsify().has_soft_degree_threshold()) {
    edges_ = SparsifyHelper(std::move(edges_),
                            config_.sparsify().soft_degree_threshold(),
                            /*remove_one_sided_edges=*/false);
  }

  if (config_.sparsify().has_hard_degree_threshold()) {
    edges_ = SparsifyHelper(std::move(edges_),
                            config_.sparsify().hard_degree_threshold(),
                            /*remove_one_sided_edges=*/true);
  }

  return absl::OkStatus();
}

}  // namespace in_memory
}  // namespace graph_mining
