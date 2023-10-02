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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_COST_METRIC_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_COST_METRIC_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/parline/minla.pb.h"

namespace graph_mining::in_memory {

// Base class for implementing a Minimum Linear Arrangement cost metric.
class MinlaCostMetric {
 public:
  virtual ~MinlaCostMetric() = default;

  // Computes the cost of a given minimum linear arrangement.
  virtual double ComputeCost(const std::vector<gbbs::uintE>& minla,
                             const GbbsGraph& graph) const {
    const std::size_t num_nodes = graph.Graph()->n;
    std::vector<double> node_locations(num_nodes);
    parlay::parallel_for(0, num_nodes,
                         [&](std::size_t i) { node_locations[minla[i]] = i; });
    return ComputeCostFromNodeLocations(node_locations, graph);
  }

  // Computes the cost of a given input node locations.
  virtual double ComputeCostFromNodeLocations(
      const std::vector<double>& node_locations,
      const GbbsGraph& graph) const = 0;

  // Computes an improved node location for a given node id wrt to the input
  // node locations. The improvement is based on the underlying cost metric.
  virtual double ImproveNodeLocation(gbbs::uintE node_id,
                                     const std::vector<double>& node_locations,
                                     const GbbsGraph& graph) const = 0;

 protected:
  // Convenience struct to hold weighted values that can be used by subclasses.
  struct WeightedValue {
    double value;
    double weight;
  };
};

// Implements a weighted L1 cost metric. Please see the description of
// DEFAULT_L1_COST_METRIC in minla.proto.
class MinlaCostMetricL1 : public MinlaCostMetric {
 public:
  double ComputeCostFromNodeLocations(const std::vector<double>& node_locations,
                                      const GbbsGraph& graph) const override;
  double ImproveNodeLocation(gbbs::uintE node_id,
                             const std::vector<double>& node_locations,
                             const GbbsGraph& graph) const override;

 private:
  // Computes the weighted median of a list of values given as a vector of
  // WeightedValue objects. Note that 'weighted_values' vector is passed by
  // value so callers should use std::move to avoid the copy.
  static double WeightedMedian(std::vector<WeightedValue> weighted_values);

  // For testing purposes only.
  friend class MinlaCostMetricL1Test;
};

// Implements a weighted L2 cost metric. Please see the description of
// L2_COST_METRIC in minla.proto.
class MinlaCostMetricL2 : public MinlaCostMetric {
 public:
  double ComputeCostFromNodeLocations(const std::vector<double>& node_locations,
                                      const GbbsGraph& graph) const override;
  double ImproveNodeLocation(gbbs::uintE node_id,
                             const std::vector<double>& node_locations,
                             const GbbsGraph& graph) const override;
};

// Creates and returns a specific implementation of a MinlaCostMetric class
// based on the passed config. See MinimumLinearArrangementConfig.cost_metric
// field in minla.proto.
std::unique_ptr<MinlaCostMetric> CreateMinlaCostMetric(
    const MinimumLinearArrangementConfig& config);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_COST_METRIC_H_
