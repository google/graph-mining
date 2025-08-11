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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_H_

#include <vector>

#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/parline/minla.pb.h"

namespace graph_mining::in_memory {

// Implements the Minimum Linear Arrangement heuristic where the goal is to
// compute an ordering (i.e. a permutation) of the nodes of an input graph such
// that neighboring nodes in the input graph are as close to each other as
// possible. For the specific cost metrics that are minimized to quantify this
// closeness please see the CostMetric enum in the minla.proto file.
class MinimumLinearArrangement {
 public:
  explicit MinimumLinearArrangement(
      const MinimumLinearArrangementConfig& config);
  ~MinimumLinearArrangement() = default;

  // Computes and returns a minimum linear arrangement of an input graph. The
  // returned vector is a permutation of the node ids in the input graph
  // computed to minimize the configured cost metric.
  std::vector<gbbs::uintE> Compute(const GbbsGraph& graph) const;

  // Improves an existing minimum linear arrangement by minimizing the cost
  // metric. The input minla vector is assumed to contain an existing
  // permutation of the node ids in the input graph.
  void Improve(const GbbsGraph& graph, std::vector<gbbs::uintE>& minla) const;

 private:
  MinimumLinearArrangementConfig config_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_MINLA_H_
