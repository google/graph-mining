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

#include "in_memory/clustering/parline/minla_cost_metric.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "in_memory/clustering/gbbs_graph.h"

namespace graph_mining::in_memory {
namespace {

using uintE = gbbs::uintE;

// TODO: Move it to:
// /third_party/graph_mining/in_memory/parallel/parallel_graph_utils.h
template <class vertex_type>
double WeightedOutDegree(vertex_type& vertex) {
  return vertex.out_neighbors().reduce(
      [&](uintE id, uintE neighbor_id, float weight) -> double {
        return weight;
      },
      parlay::addm<double>());
}
}  // namespace

// This implementation treats each value as a point in the middle of its weight
// segment (after the values are aligned along a line giving each one a segment
// equal to its weight). An interval in this ordering is defined as the segment
// between two adjacent values. The weighted median is then computed by linearly
// interpolating the two end-points of the interval in which the half-position
// falls into.
double MinlaCostMetricL1::WeightedMedian(
    std::vector<WeightedValue> weighted_values) {
  if (weighted_values.empty()) return 0;
  if (weighted_values.size() == 1) return weighted_values[0].value;
  parlay::sort_inplace(weighted_values,
                       [](const WeightedValue& a, const WeightedValue& b) {
                         return a.value < b.value;
                       });
  // Compute interval sums to be used for determining half position interval
  // in parallel. (The values in interval_sums are scaled by 2.)
  std::vector<double> interval_sums(weighted_values.size());
  parlay::parallel_for(0, weighted_values.size(), [&](std::size_t i) {
    interval_sums[i] = (i == 0 ? 0 : weighted_values[i - 1].weight) +
                       weighted_values[i].weight;
  });
  parlay::scan_inclusive_inplace(interval_sums, parlay::addm<double>());
  // The last entry in weighted_values is included only once in interval_sums.
  const double total_weight =
      interval_sums.back() + weighted_values.back().weight;
  const double half_position = total_weight / 2;
  // Find half position interval and compute the weighted median.
  auto it = std::upper_bound(interval_sums.begin(), interval_sums.end(),
                             half_position);
  if (it == interval_sums.begin()) {
    return weighted_values.front().value;
  } else if (it == interval_sums.end()) {
    return weighted_values.back().value;
  } else {
    int i = it - interval_sums.begin();
    // half_position is between (i-1)th and ith values so we linearly
    // interpolate corresponding values.
    const double fraction = (half_position - *(it - 1)) / (*it - *(it - 1));
    return (1 - fraction) * weighted_values[i - 1].value +
           fraction * weighted_values[i].value;
  }
}

double MinlaCostMetricL1::ComputeCostFromNodeLocations(
    const std::vector<double>& node_locations, const GbbsGraph& graph) const {
  auto map_fn = [&](uintE u, uintE v, float weight) -> double {
    return weight * std::abs(node_locations[u] - node_locations[v]);
  };
  return graph.Graph()->reduceEdges(map_fn, parlay::addm<double>());
}

double MinlaCostMetricL1::ImproveNodeLocation(
    uintE node_id, const std::vector<double>& node_locations,
    const GbbsGraph& graph) const {
  // An improved node location for L1 cost metric is the weighted median of its
  // neighbors' locations including its own.
  auto vertex = graph.Graph()->get_vertex(node_id);
  auto neighbors = vertex.out_neighbors();
  std::vector<WeightedValue> weighted_locations(neighbors.get_degree() + 1);
  parlay::parallel_for(0, neighbors.get_degree(), [&](std::size_t i) {
    weighted_locations[i] = {node_locations[neighbors.get_neighbor(i)],
                             neighbors.get_weight(i)};
  });
  double self_weight = neighbors.get_degree() > 0
                           ? WeightedOutDegree(vertex) / neighbors.get_degree()
                           : 1;
  weighted_locations.back() = {node_locations[node_id], self_weight};
  return WeightedMedian(std::move(weighted_locations));
}

double MinlaCostMetricL2::ComputeCostFromNodeLocations(
    const std::vector<double>& node_locations, const GbbsGraph& graph) const {
  auto map_fn = [&](uintE u, uintE v, float weight) -> double {
    auto diff = node_locations[u] - node_locations[v];
    return weight * diff * diff;
  };
  return graph.Graph()->reduceEdges(map_fn, parlay::addm<double>());
}

double MinlaCostMetricL2::ImproveNodeLocation(
    uintE node_id, const std::vector<double>& node_locations,
    const GbbsGraph& graph) const {
  // An improved node location for L2 cost metric is the weighted mean of its
  // neighbors' locations including its own.
  auto vertex = graph.Graph()->get_vertex(node_id);
  // Maps neighbor node to <weighted-location, weight> pair.
  auto map_fn = [&](uintE u, uintE v, float weight) -> WeightedValue {
    return {weight * node_locations[v], weight};
  };
  // Sums weighted locations and weights into a single pair.
  auto reduce_fn = parlay::make_monoid(
      [](const WeightedValue& a, const WeightedValue& b) -> WeightedValue {
        return {a.value + b.value, a.weight + b.weight};
      },
      WeightedValue({0, 0}));
  auto weighted_sum_pair = vertex.out_neighbors().reduce(map_fn, reduce_fn);
  // Include node's own location in the mean.
  double self_weight = vertex.out_degree() > 0
                           ? weighted_sum_pair.weight / vertex.out_degree()
                           : 1;
  return (weighted_sum_pair.value + node_locations[node_id]) /
         (weighted_sum_pair.weight + self_weight);
}

std::unique_ptr<MinlaCostMetric> CreateMinlaCostMetric(
    const MinimumLinearArrangementConfig& config) {
  switch (config.cost_metric()) {
    case MinimumLinearArrangementConfig::COST_METRIC_UNSPECIFIED:
    case MinimumLinearArrangementConfig::L1_COST_METRIC:
      return std::make_unique<MinlaCostMetricL1>();
    case MinimumLinearArrangementConfig::L2_COST_METRIC:
      return std::make_unique<MinlaCostMetricL2>();
  }
  ABSL_LOG(FATAL) << "A valid cost metric must be specified in "
                     "MinimumLinearArrangementConfig.cost_metric, found: "
                  << config.cost_metric();
}

}  // namespace graph_mining::in_memory
