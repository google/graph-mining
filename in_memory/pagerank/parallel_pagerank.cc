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

#include "in_memory/pagerank/parallel_pagerank.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "benchmarks/PageRank/PageRank.h"
#include "gbbs/helpers/parallel_for_with_status.h"
#include "gbbs/helpers/progress_reporting.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/types.h"
#include "in_memory/pagerank/pagerank.pb.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

absl::StatusOr<parlay::sequence<double>> ParallelPageRank::Run(
    const absl::Span<const NodeId> source_nodes, const PageRankConfig& config,
    std::optional<gbbs::ReportProgressCallback> report_progress) const {
  

  if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "'graph_' must be initialized before running ParallelPageRank");
  }

  if (std::isnan(config.damping_factor())) {
    return absl::InvalidArgumentError("'damping_factor' must not be NaN");
  }
  if (config.damping_factor() < 0.0 || config.damping_factor() >= 1.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("'damping_factor' must be in the range [0, 1); found ",
                     config.damping_factor()));
  }

  if (config.num_iterations() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("'num_iterations' must be nonnegative; found ",
                     config.num_iterations()));
  }

  if (std::isnan(config.approx_precision())) {
    return absl::InvalidArgumentError("'approx_precision' must not be NaN");
  }
  if (config.approx_precision() < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("'approx_precision' must be nonnegative; found ",
                     config.approx_precision()));
  }

  // Convert the source node IDs to `gbbs::uintE`.
  std::vector<gbbs::uintE> source_nodes_gbbs(source_nodes.size());
  const std::size_t num_nodes = graph_.Graph()->n;
  RETURN_IF_ERROR(gbbs::parallel_for_with_status(
      0, source_nodes.size(), [&](std::size_t i) {
        const NodeId source_node_id = source_nodes[i];
        if (source_node_id < 0 || source_node_id >= num_nodes) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid source node ID: ", source_node_id));
        } else {
          source_nodes_gbbs[i] = static_cast<gbbs::uintE>(source_node_id);
          return absl::OkStatus();
        }
      }));
  return gbbs::PageRank_edgeMap(
      *(graph_.Graph()), config.approx_precision(), source_nodes_gbbs,
      config.damping_factor(), config.num_iterations(),
      /*has_in_edges=*/true, std::move(report_progress));
}

}  // namespace graph_mining::in_memory
