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

#include "in_memory/clustering/parline/minla.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "parlay/primitives.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/parline/minla_cost_metric.h"
#include "in_memory/clustering/parline/minla.pb.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {
namespace {

using uintE = ::gbbs::uintE;

const int kDefaultMaxIterations = 20;

MinimumLinearArrangementConfig MinimumLinearArrangementConfigWithDefaults(
    MinimumLinearArrangementConfig minla_config) {
  // Set some defaults if needed.
  if (!minla_config.has_max_iterations()) {
    minla_config.set_max_iterations(kDefaultMaxIterations);
  }
  return minla_config;
}

}  // namespace

MinimumLinearArrangement::MinimumLinearArrangement(
    const MinimumLinearArrangementConfig& config) {
  config_ = MinimumLinearArrangementConfigWithDefaults(config);
}

std::vector<uintE>
MinimumLinearArrangement::Compute(const GbbsGraph& graph) const {
  
  std::size_t num_nodes = graph.Graph()->n;
  // Entry i is the node id at location i. This is the minla ordering we are
  // interested in computing.
  std::vector<uintE> minla(num_nodes);
  // We set the initial location of a node to its own id.
  parlay::copy(parlay::iota<uintE>(num_nodes), minla);
  Improve(graph, minla);
  return minla;
}

// Implements a modified version of the iterative median (or mean) algorithm
// described in:
//    https://www.wisdom.weizmann.ac.il/~harel/papers/min_la.pdf
// The differences in this implementation are:
//   - All node locations are updated in parallel
//   - Node locations are rescaled to prevent collapsing
//   - Node's own location is also included in improved location computations
void MinimumLinearArrangement::Improve(const GbbsGraph& graph,
                                       std::vector<uintE>& minla) const {
  
  std::size_t num_nodes = graph.Graph()->n;
  // Node id to location array for fast lookup.
  std::vector<double> node_locations(num_nodes);
  parlay::parallel_for(0, num_nodes,
                       [&](std::size_t i) { node_locations[minla[i]] = i; });
  // Entry i is the new computed node location (as double) of node with id i.
  std::vector<double> new_node_locations(num_nodes);
  auto minla_cost_metric = CreateMinlaCostMetric(config_);

  double cost_diff = std::numeric_limits<double>::max();
  double prev_cost =
      minla_cost_metric->ComputeCostFromNodeLocations(node_locations, graph);
  for (int i = 0; i < config_.max_iterations() &&
                  cost_diff > config_.placement_convergence_delta();
       ++i) {
    // 1) Iterate through all the nodes and compute their new locations.
    parlay::parallel_for(0, num_nodes, [&](std::size_t i) {
      new_node_locations[i] =
          minla_cost_metric->ImproveNodeLocation(i, node_locations, graph);
    });

    // 2) Rescale the node locations to [0,n-1] range.
    auto min_max = parlay::minmax_element(new_node_locations);
    double min_value = *min_max.first;
    double max_value = *min_max.second;
    ABSL_CHECK_GT(max_value, min_value)
        << "All new locations collapsed to a single point at iteration " << i;
    const double scaling_factor = (num_nodes - 1) / (max_value - min_value);
    parlay::parallel_for(0, num_nodes, [&](std::size_t j) {
      new_node_locations[j] =
          (new_node_locations[j] - min_value) * scaling_factor;
    });

    // 3) Compute cost difference.
    double cost = minla_cost_metric->ComputeCostFromNodeLocations(
        new_node_locations, graph);
    cost_diff = std::abs(cost - prev_cost);
    prev_cost = cost;
    node_locations.swap(new_node_locations);
  }
  // Generate improved minla using the final node locations.
  parlay::sort_inplace(minla, [&node_locations](std::size_t i, std::size_t j) {
    return node_locations[i] < node_locations[j];
  });
}

}  // namespace graph_mining::in_memory
