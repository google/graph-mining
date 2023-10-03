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

#include "in_memory/generation/add_edge_weights.h"

#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

absl::Status AddUniformWeights(SimpleUndirectedGraph& graph, double low,
                               double high) {
  absl::BitGen gen;
  for (size_t i = 0; i < graph.NumNodes(); ++i) {
    for (const auto& [neighbor_id, weight] : graph.Neighbors(i)) {
      if (i < neighbor_id) {
        RETURN_IF_ERROR(
            graph.SetEdgeWeight(i, neighbor_id, absl::Uniform(gen, low, high)));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace graph_mining::in_memory
