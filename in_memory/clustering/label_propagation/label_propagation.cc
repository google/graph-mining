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

#include "in_memory/clustering/label_propagation/label_propagation.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "benchmarks/GraphColoring/Hasenplaugh14/GraphColoring.h"
#include "gbbs/bridge.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/label_propagation/label_propagation.pb.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/delayed_sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

namespace internal {

constexpr NodeId kInvalidLabel = std::numeric_limits<NodeId>::max();

// Computes a new label for the vertex given the current label set.
NodeId ComputeNewLabel(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const parlay::sequence<NodeId>& cur_labels, NodeId node_id) {
  auto node = graph.get_vertex(node_id);
  size_t degree = node.out_degree();
  if (degree == 0) {
    return kInvalidLabel;
  }

  // Collect the set of neighboring labels, and weights to them.
  using Elt = std::pair<NodeId, double>;
  parlay::sequence<Elt> label_and_weight(degree);
  auto map_f = [&](const NodeId& u, const NodeId& v, const float& weight,
                   size_t index) {
    label_and_weight[index] = Elt{cur_labels[v], double{weight}};
  };
  node.out_neighbors().map_with_index(map_f);

  // Sum up the weights.
  parlay::sort_inplace(label_and_weight);
  auto copy_f = [](Elt a, Elt b) {
    if (a.first == b.first) {
      return Elt{a.first, a.second + b.second};
    }
    return b;
  };
  Elt identity = {std::numeric_limits<NodeId>::max(), 0};
  auto copy_monoid = parlay::make_monoid(copy_f, identity);
  parlay::scan_inclusive_inplace(label_and_weight, copy_monoid);

  // Collect the total weight per-label.
  auto label_ends =
      parlay::pack_index(parlay::delayed_seq<bool>(degree, [&](size_t i) {
        return (i == degree - 1) ||
               (label_and_weight[i].first != label_and_weight[i + 1].first);
      }));
  auto weights_and_labels = parlay::map(label_ends, [&](size_t index) {
    return std::make_pair(label_and_weight[index].second,
                          label_and_weight[index].first);
  });
  parlay::sort_inplace(weights_and_labels, std::greater<>{});

  return weights_and_labels[0].second;
}

// Expects an undirected (possibly weighted) graph.
absl::StatusOr<Clustering> LabelPropagationImplementation(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const parlay::sequence<NodeId>& initial_labels,
    const LabelPropagationConfig& label_propagation_config) {
  const NodeId n = graph.n;

  // Create two copies of the labels.
  parlay::sequence<NodeId> cur_labels =
      parlay::sequence<NodeId>(initial_labels);

  parlay::sequence<bool> cur_active(n, true);
  parlay::sequence<bool> next_active(n, false);

  // Try the graph-coloring variant.
  using color = gbbs::uintE;
  parlay::sequence<std::pair<color, NodeId>> color_and_node;
  parlay::sequence<size_t> color_starts;
  if (label_propagation_config.use_graph_coloring()) {
    parlay::sequence<color> coloring = gbbs::Coloring(graph);
    color_and_node = parlay::tabulate<std::pair<color, NodeId>>(
        coloring.size(),
        [&](NodeId i) { return std::make_pair(coloring[i], i); });
    parlay::sort_inplace(coloring);
    color_starts = parlay::pack_index(
        parlay::delayed_seq<bool>(color_and_node.size(), [&](size_t i) {
          return (i == 0) ||
                 (color_and_node[i].first != color_and_node[i - 1].first);
        }));
  }

  ABSL_LOG(INFO) << "Starting LabelPropagation. Parameters:";
  ABSL_LOG(INFO) << label_propagation_config;

  for (size_t round = 0; round < label_propagation_config.max_rounds();
       ++round) {
    ABSL_LOG(INFO) << "Running round: " << round;
    std::atomic<bool> changed = false;

    auto process_node = [&](NodeId i) {
      // One of our neighbors changed in the last round (or this is
      // the first round). Recompute this node's label.
      if (cur_active[i]) {
        // Reset our flag for the next round.
        cur_active[i] = false;

        // Computes the next label based on cur_labels.
        NodeId new_label = internal::ComputeNewLabel(graph, cur_labels, i);

        if (new_label != kInvalidLabel && cur_labels[i] != new_label) {
          // Set our label for the next round by updating the current label set
          // asynchronously. Note that we use a CAS here (and below, when
          // updating next_active) to avoid a data race when running under tsan.
          ABSL_CHECK(
              gbbs::CAS<NodeId>(&cur_labels[i], cur_labels[i], new_label));
          // Mark our neighbors to be active in the next round.
          auto activate_f = [&](const NodeId& u, const NodeId& v,
                                const float& weight) {
            if (!next_active[v]) {
              // TODO: benchmark using std::atomics throughout instead
              // of gbbs::CAS (both here, and above). Note that we should not
              // call CHECK on the CAS below, since the CAS below can fail if
              // multiple vertices are trying to update next_active for the same
              // neighbor.
              gbbs::CAS<bool>(&next_active[v], false, true);
            }
          };
          graph.get_vertex(i).out_neighbors().map(activate_f);
          if (!changed) {
            changed = true;
          }
        }
      }
    };

    if (!label_propagation_config.use_graph_coloring()) {
      // Map over all nodes.
      parlay::parallel_for(0, n, [&](size_t i) { process_node(i); }, 1);
    } else {
      // Map over each color set, one after the other.
      for (size_t c = 0; c < color_starts.size(); ++c) {
        size_t start_offset = color_starts[c];
        size_t end_offset = (c == color_starts.size() - 1)
                                ? color_and_node.size()
                                : color_starts[c + 1];
        // Map over all vertices of the same color in parallel.
        parlay::parallel_for(
            start_offset, end_offset,
            [&](size_t i) {
              NodeId node_id = color_and_node[i].second;
              process_node(node_id);
            },
            1);
      }
    }

    std::swap(cur_active, next_active);

    // Check convergence. If no labels changed in this round, quit.
    if (!changed) break;
  }

  ASSIGN_OR_RETURN(Clustering result,
                   ClusterIdSequenceToClustering(absl::MakeSpan(cur_labels)));
  return result;
}

}  // namespace internal

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelLabelPropagationClusterer::Cluster(
    const graph_mining::in_memory::ClustererConfig& clusterer_config) const {
  

  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
      graph_.Graph();

  const LabelPropagationConfig& label_propagation_config =
      clusterer_config.label_propagation_clusterer_config();

  auto initial_labels =
      parlay::tabulate(current_graph->n, [&](NodeId i) { return i; });

  InMemoryClusterer::Clustering clustering;
  ASSIGN_OR_RETURN(clustering, internal::LabelPropagationImplementation(
                                   *current_graph, initial_labels,
                                   label_propagation_config));
  return clustering;
}

}  // namespace graph_mining::in_memory
