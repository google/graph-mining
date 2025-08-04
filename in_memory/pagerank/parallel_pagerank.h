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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_

#include <optional>

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gbbs/helpers/progress_reporting.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"
#include "in_memory/pagerank/pagerank.pb.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

class ParallelPageRank {
 public:
  DirectedGbbsGraph* absl_nonnull MutableGraph() ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return &graph_;
  }

  // Returns a sequence where the i-th element contains the pagerank value for
  // the i-th node.
  //
  // The `source_nodes` are the nodes from which the centrality is computed.
  // When no source nodes are provided, the centrality is computed from all
  // nodes in the graph. The elements of `source_nodes` must be distinct and
  // correspond to valid nodes in the graph.
  //
  // The length of `source_nodes` is expected to have very limited impact on the
  // performance of the algorithm (e.g. in terms of running time and memory
  // usage).
  //
  // If provided,`report_progress` will be called periodically to report the
  // progress of the computation.
  absl::StatusOr<parlay::sequence<double>> Run(
      absl::Span<const NodeId> source_nodes, const PageRankConfig& config,
      std::optional<gbbs::ReportProgressCallback> report_progress =
          std::nullopt) const;

 private:
  DirectedGbbsGraph graph_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PAGERANK_PARALLEL_PAGERANK_H_
