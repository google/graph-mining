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

#include "in_memory/clustering/gbbs_graph_test_utils.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"

namespace graph_mining::in_memory {

void CheckGbbsGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph,
    std::size_t num_vertices,
    const std::vector<std::vector<std::tuple<gbbs::uintE, float>>>& neighbors) {
  ASSERT_EQ(graph->n, num_vertices);
  for (std::size_t i = 0; i < neighbors.size(); i++) {
    EXPECT_EQ(graph->get_vertex(i).out_degree(), neighbors[i].size());
  }
  for (std::size_t i = 0; i < neighbors.size(); i++) {
    auto graph_neighbors = std::vector<std::tuple<gbbs::uintE, float>>(
        graph->get_vertex(i).out_neighbors().neighbors,
        graph->get_vertex(i).out_neighbors().neighbors +
            graph->get_vertex(i).out_degree());
    auto sorted_neighbors = neighbors[i];
    std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
    EXPECT_THAT(graph_neighbors, testing::WhenSorted(sorted_neighbors));
  }
}

}  // namespace graph_mining::in_memory
