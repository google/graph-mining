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

#include "in_memory/clustering/affinity/parallel_affinity_internal.h"

#include <initializer_list>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "gbbs/graph.h"
#include "gbbs/macros.h"
#include "gbbs/vertex.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/gbbs_graph_test_utils.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/parse_proto/parse_text_proto.h"
#include "parlay/internal/group_by.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {
namespace {

using gbbs::uintE;
#define ParseTextProtoOrDie(STR) PARSE_TEXT_PROTO(STR)
using ::graph_mining::in_memory::AffinityClustererConfig;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;


std::vector<std::pair<AffinityClustererConfig, float>>
CreateAffinityTestScenarios(std::vector<std::pair<absl::string_view, float>>
                                string_configs_and_weights) {
  std::vector<std::pair<AffinityClustererConfig, float>> configs_and_weights;

  for (const auto& [string_config, weight] : string_configs_and_weights) {
    configs_and_weights.push_back({ParseTextProtoOrDie(string_config), weight});
  }

  return configs_and_weights;
}

using ParallelAffinityInternalTest = ::testing::Test;

class CompressGraphTest : public ParallelAffinityInternalTest {};
class NearestNeighborLinkageTest : public ParallelAffinityInternalTest {};
class ComputeClustersTest : public ParallelAffinityInternalTest {};
class FindFinishedClustersTest : public ParallelAffinityInternalTest {};
class ComputeFinishedClusterStatsTest : public ParallelAffinityInternalTest {};
class EnforceMaxClusterSizeTest : public ParallelAffinityInternalTest {};

using CompressGraphDeathTest = CompressGraphTest;

TEST_F(CompressGraphDeathTest,
       NegativeAverageMaxDegreeBoundedWeightMultiplier) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  int num_vertices = 3;
  int num_edges = 4;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{2}, float{2}),
                               std::make_tuple(uintE{0}, float{1}),
                               std::make_tuple(uintE{0}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 1}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<gbbs::uintE> cluster_ids = {0, 2, 2};
  std::vector<double> node_weights;
  EXPECT_DEATH(
      auto compressed_graph = CompressGraph(
          G, node_weights, cluster_ids,
          ParseTextProtoOrDie(
              "edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
              "max_degree_bounded_weight_multiplier: -1.0 "
              "weight_threshold: 1.0")),
      HasSubstr("Check failed: "
                "affinity_config.max_degree_bounded_weight_multiplier()"
                " > 0 (-1 vs. 0)"));
}

TEST_F(CompressGraphDeathTest, ZeroAverageMaxDegreeBoundedWeightMultiplier) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  int num_vertices = 3;
  int num_edges = 4;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{2}, float{2}),
                               std::make_tuple(uintE{0}, float{1}),
                               std::make_tuple(uintE{0}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 1}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<gbbs::uintE> cluster_ids = {0, 2, 2};
  std::vector<double> node_weights;
  EXPECT_DEATH(
      auto compressed_graph = CompressGraph(
          G, node_weights, cluster_ids,
          ParseTextProtoOrDie(
              "edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
              "max_degree_bounded_weight_multiplier: 0.0 "
              "weight_threshold: 1.0")),
      HasSubstr("Check failed: "
                "affinity_config.max_degree_bounded_weight_multiplier()"
                " > 0 (0 vs. 0)"));
}

TEST_F(CompressGraphTest, EdgeAggregationThreeNodes) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  int num_vertices = 3;
  int num_edges = 4;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{2}, float{2}),
                               std::make_tuple(uintE{0}, float{1}),
                               std::make_tuple(uintE{0}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 1}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<gbbs::uintE> cluster_ids = {0, 2, 2};
  std::vector<double> node_weights;

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.5},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 3.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 2.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 1.0 "
             "weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 3.0 "
             "weight_threshold: 2.0",
             1.5}})) {
    ASSERT_OK_AND_ASSIGN(
        auto compressed_graph,
        CompressGraph(G, node_weights, cluster_ids, clusterer_config));
    std::vector<std::vector<GbbsEdge>> neighbors = {
        {std::make_tuple(uintE{2}, expected_weight)},
        {},
        {std::make_tuple(uintE{0}, expected_weight)}};
    graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 3,
                                            neighbors);
  }
}

TEST_F(CompressGraphTest, EdgeAggregationFourNodes) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 8;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{3}, float{4}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{1}, float{2}), std::make_tuple(uintE{3}, float{3}),
       std::make_tuple(uintE{0}, float{4}),
       std::make_tuple(uintE{2}, float{3})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[6]), gbbs::vertex_data{0, 2}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {1, 1, 3, 3};
  std::vector<double> node_weights;

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.5},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 6.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 4.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 1.0 "
             "weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 3.0 "
             "weight_threshold: 2.0",
             1.5}})) {
    ASSERT_OK_AND_ASSIGN(
        auto compressed_graph,
        CompressGraph(G, node_weights, cluster_ids, clusterer_config));
    std::vector<std::vector<GbbsEdge>> neighbors = {
        {},
        {std::make_tuple(uintE{3}, expected_weight)},
        {},
        {std::make_tuple(uintE{1}, expected_weight)}};
    graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 4,
                                            neighbors);
  }
}

TEST_F(CompressGraphTest, EdgeAggregationThreeNodesWithNodeWeights) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  int num_vertices = 3;
  int num_edges = 4;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{2}, float{2}),
                               std::make_tuple(uintE{0}, float{1}),
                               std::make_tuple(uintE{0}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 1}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<gbbs::uintE> cluster_ids = {0, 2, 2};
  std::vector<double> node_weights = {1, 3, 5};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.625},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 3.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 2.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 1.0 "
             "weight_threshold: 1.0",
             3.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 10.0 "
             "weight_threshold: 1.0",
             1.625}})) {
    ASSERT_OK_AND_ASSIGN(
        auto compressed_graph,
        CompressGraph(G, node_weights, cluster_ids, clusterer_config));
    std::vector<std::vector<GbbsEdge>> neighbors = {
        {std::make_tuple(uintE{2}, expected_weight)},
        {},
        {std::make_tuple(uintE{0}, expected_weight)}};
    graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 3,
                                            neighbors);
  }
}

TEST_F(CompressGraphTest, EdgeAggregationFourNodesWithNodeWeights) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 8;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{3}, float{4}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{1}, float{2}), std::make_tuple(uintE{3}, float{3}),
       std::make_tuple(uintE{0}, float{4}),
       std::make_tuple(uintE{2}, float{3})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[6]), gbbs::vertex_data{0, 2}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {1, 1, 3, 3};
  std::vector<double> node_weights = {1, 3, 2, 3};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.2},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 6.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 4.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             2.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 1.0 "
             "weight_threshold: 1.0",
             2.0},
            {"edge_aggregation_function: AVERAGE_WITH_MAX_DEGREE_BOUNDED "
             "max_degree_bounded_weight_multiplier: 10.0 "
             "weight_threshold: 1.0",
             1.2}})) {
    ASSERT_OK_AND_ASSIGN(
        auto compressed_graph,
        CompressGraph(G, node_weights, cluster_ids, clusterer_config));
    std::vector<std::vector<GbbsEdge>> neighbors = {
        {},
        {std::make_tuple(uintE{3}, expected_weight)},
        {},
        {std::make_tuple(uintE{1}, expected_weight)}};
    graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 4,
                                            neighbors);
  }
}

TEST_F(CompressGraphTest, RemoveNode) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 3;
  int num_edges = 4;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{0}, float{1}),
                               std::make_tuple(uintE{2}, float{2}),
                               std::make_tuple(uintE{1}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 1}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {0, 2, UINT_E_MAX};
  std::vector<double> node_weights;

  ASSERT_OK_AND_ASSIGN(
      auto compressed_graph,
      CompressGraph(G, node_weights, cluster_ids,
                    ParseTextProtoOrDie("edge_aggregation_function: SUM")));
  std::vector<std::vector<GbbsEdge>> neighbors = {
      {std::make_tuple(uintE{2}, 1.0)}, {}, {std::make_tuple(uintE{0}, 1.0)}};
  graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 3,
                                          neighbors);
}

TEST_F(CompressGraphTest, RemoveTwoNodes) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 3;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{1}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {UINT_E_MAX, 2, UINT_E_MAX};
  std::vector<double> node_weights;

  ASSERT_OK_AND_ASSIGN(
      auto compressed_graph,
      CompressGraph(G, node_weights, cluster_ids,
                    ParseTextProtoOrDie("edge_aggregation_function: SUM")));

  std::vector<std::vector<GbbsEdge>> neighbors = {{}, {}, {}};
  graph_mining::in_memory::CheckGbbsGraph(compressed_graph.graph.get(), 3,
                                          neighbors);
}

TEST_F(NearestNeighborLinkageTest, NoEdges) {
  int num_vertices = 3;
  int num_edges = 0;
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(nullptr, gbbs::vertex_data{0, 0}, 0),
       gbbs::symmetric_vertex<float>(nullptr, gbbs::vertex_data{0, 0}, 1),
       gbbs::symmetric_vertex<float>(nullptr, gbbs::vertex_data{0, 0}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  EXPECT_THAT(NearestNeighborLinkage(G, 1.0),
              IsOkAndHolds(ElementsAreArray<uintE>({0, 1, 2})));
}

TEST_F(NearestNeighborLinkageTest, AppliesWeightThreshold) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 2;
  int num_edges = 2;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{1}),
                               std::make_tuple(uintE{0}, float{1})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 1}, 1)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  ASSERT_OK_AND_ASSIGN(auto cluster_ids, NearestNeighborLinkage(G, 1.0));
  ASSERT_EQ(cluster_ids.size(), 2);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_GE(cluster_ids[0], 0);
  EXPECT_LE(cluster_ids[0], 1);

  ASSERT_OK_AND_ASSIGN(cluster_ids, NearestNeighborLinkage(G, 1.1));

  ASSERT_EQ(cluster_ids.size(), 2);
  EXPECT_NE(cluster_ids[0], cluster_ids[1]);
}

TEST_F(NearestNeighborLinkageTest, ChoosesBestNeighbor) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{3}), std::make_tuple(uintE{0}, float{3}),
       std::make_tuple(uintE{2}, float{1}), std::make_tuple(uintE{1}, float{1}),
       std::make_tuple(uintE{3}, float{3}),
       std::make_tuple(uintE{2}, float{3})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[5]), gbbs::vertex_data{0, 1}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  ASSERT_OK_AND_ASSIGN(auto cluster_ids, NearestNeighborLinkage(G, 0.0));

  ASSERT_EQ(cluster_ids.size(), 4);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_EQ(cluster_ids[2], cluster_ids[3]);
  EXPECT_NE(cluster_ids[1], cluster_ids[2]);
}

TEST_F(NearestNeighborLinkageTest, BreaksTies) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 5;
  int num_edges = 8;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{1}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{3}, float{5}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{4}, float{5}),
       std::make_tuple(uintE{1}, float{5}),
       std::make_tuple(uintE{2}, float{5})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[6]), gbbs::vertex_data{0, 1}, 3),
       gbbs::symmetric_vertex<float>(&(edges[7]), gbbs::vertex_data{0, 1}, 4)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  ASSERT_OK_AND_ASSIGN(auto cluster_ids, NearestNeighborLinkage(G, 0.0));

  ASSERT_EQ(cluster_ids.size(), 5);
  EXPECT_EQ(cluster_ids[0], cluster_ids[2]);
  EXPECT_EQ(cluster_ids[2], cluster_ids[4]);
  EXPECT_EQ(cluster_ids[1], cluster_ids[3]);
  EXPECT_NE(cluster_ids[0], cluster_ids[1]);
}

TEST_F(NearestNeighborLinkageTest, ChoosesBestNeighborInCircle) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{3}, float{3}), std::make_tuple(uintE{2}, float{1}),
       std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{3}, float{2}),
       std::make_tuple(uintE{0}, float{3}),
       std::make_tuple(uintE{2}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  ASSERT_OK_AND_ASSIGN(auto cluster_ids, NearestNeighborLinkage(G, 0.0));

  ASSERT_EQ(cluster_ids.size(), 4);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_EQ(cluster_ids[1], cluster_ids[2]);
  EXPECT_EQ(cluster_ids[2], cluster_ids[3]);
}

TEST_F(NearestNeighborLinkageTest, ChoosesBestNeighborWithMinSizeConstraint) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 3));
  ASSERT_OK(graph.AddEdge(1, 2, 1));
  ASSERT_OK(graph.AddEdge(2, 3, 3));
  GbbsGraph gbbs_graph;
  ASSERT_OK(CopyGraph(graph, &gbbs_graph));
  ASSERT_OK(gbbs_graph.FinishImport());

  // Create node weights such that the current weight of nodes 2 and 3 exceeds
  // the min-size threshold. Thus nodes 2 and 3 will remain in their isolated
  // clusters after NearestNeighborLinkage.
  AffinityClustererConfig::SizeConstraint size_constraint;
  size_constraint.set_min_cluster_size(2.0);
  std::vector<double> node_weights({1, 1, 3, 3});
  const internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                              node_weights};

  ASSERT_OK_AND_ASSIGN(
      auto cluster_ids,
      NearestNeighborLinkage(*gbbs_graph.Graph(), 0.0, size_constraint_config));

  ASSERT_EQ(cluster_ids.size(), 4);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_NE(cluster_ids[2], cluster_ids[3]);
  EXPECT_NE(cluster_ids[1], cluster_ids[2]);
  EXPECT_NE(cluster_ids[1], cluster_ids[3]);
}

TEST_F(NearestNeighborLinkageTest, ChoosesBestNeighborWithMaxSizeConstraint) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 3));
  ASSERT_OK(graph.AddEdge(1, 2, 1));
  ASSERT_OK(graph.AddEdge(2, 3, 3));
  GbbsGraph gbbs_graph;
  ASSERT_OK(CopyGraph(graph, &gbbs_graph));
  ASSERT_OK(gbbs_graph.FinishImport());

  // Create node weights such that the sum of current weights of nodes 2 and 3
  // exceeds the max-size threshold. Thus nodes 2 and 3 cannot join one cluster.
  // This causes node 2 to join node 1 and node 3 to remain isolated.
  AffinityClustererConfig::SizeConstraint size_constraint;
  size_constraint.set_max_cluster_size(5.0);
  std::vector<double> node_weights({1, 1, 3, 3});
  const internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                              node_weights};

  ASSERT_OK_AND_ASSIGN(
      auto cluster_ids,
      NearestNeighborLinkage(*gbbs_graph.Graph(), 0.0, size_constraint_config));

  ASSERT_EQ(cluster_ids.size(), 4);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_EQ(cluster_ids[1], cluster_ids[2]);
  EXPECT_NE(cluster_ids[2], cluster_ids[3]);
}

TEST_F(NearestNeighborLinkageTest, MinMaxSizeConstraint) {
  // Prepare.
  SimpleUndirectedGraph graph;

  ASSERT_OK(graph.AddEdge(0, 2, 1));
  ASSERT_OK(graph.AddEdge(0, 6, 1));
  ASSERT_OK(graph.AddEdge(0, 7, 1));
  ASSERT_OK(graph.AddEdge(1, 5, 1));
  ASSERT_OK(graph.AddEdge(2, 4, 2));

  graph.SetNodeWeight(0, 1);
  graph.SetNodeWeight(1, 2);
  graph.SetNodeWeight(2, 5);
  graph.SetNodeWeight(3, 1);
  graph.SetNodeWeight(4, 1);
  graph.SetNodeWeight(5, 2);
  graph.SetNodeWeight(6, 1);
  graph.SetNodeWeight(7, 1);

  GbbsGraph gbbs_graph;
  ASSERT_OK(CopyGraph(graph, &gbbs_graph));
  ASSERT_OK(gbbs_graph.FinishImport());

  std::vector<double> node_weights{1, 2, 3, 1, 2, 2, 1, 1};
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(5);
  size_constraint.set_min_cluster_size(2);
  size_constraint.set_prefer_min_cluster_size(true);

  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  // SUT.
  //
  // Two connected-component-based clusterings are conducted inside
  // NearestNeighborLinkage. This test demonstrates that both min and max
  // cluster size enforcements work as intended.
  //
  // min_cluster_size constraint breaks up nodes 1 and 5 in the 2nd connected
  // component iteration. Those two nodes form one cluster in the 1st
  // connected component computation.
  //
  // max_cluster_size constraint breaks up nodes 2 and 4 in the 2nd connected
  // component iteration. Those two nodes form one cluster in the 1st connected
  // component computation.
  EXPECT_THAT(
      NearestNeighborLinkage(*gbbs_graph.Graph(), 0.0, size_constraint_config),
      IsOkAndHolds(ElementsAreArray<uintE>({0, 1, 2, 3, 4, 5, 0, 0})));
}

TEST_F(NearestNeighborLinkageTest, SizeConstraintWithEmptyNodeWeights) {
  // Prepare.
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1));

  GbbsGraph gbbs_graph;
  ASSERT_OK(CopyGraph(graph, &gbbs_graph));
  ASSERT_OK(gbbs_graph.FinishImport());

  // Empty node weights vector.
  std::vector<double> node_weights;
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(1);

  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  // SUT.
  //
  // An empty node_weights vector means that all nodes have default weights.
  // Thus nodes 0 and 1 cannot form one cluster.
  EXPECT_THAT(
      NearestNeighborLinkage(*gbbs_graph.Graph(), 0.0, size_constraint_config),
      IsOkAndHolds(ElementsAreArray<uintE>({0, 1})));
}

using Cluster = std::initializer_list<InMemoryClusterer::NodeId>;

TEST_F(NearestNeighborLinkageTest, GeneratesConsecutiveClusterIds) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 2, 1.0));
  ASSERT_OK(graph.AddEdge(0, 6, 3.0));
  ASSERT_OK(graph.AddEdge(2, 4, 3.0));
  ASSERT_OK(graph.AddEdge(4, 6, 1.0));
  ASSERT_OK(graph.AddEdge(1, 3, 1.0));
  ASSERT_OK(graph.AddEdge(3, 5, 1.0));
  GbbsGraph gbbs_graph;
  ASSERT_OK(CopyGraph(graph, &gbbs_graph));
  ASSERT_OK(gbbs_graph.FinishImport());
  std::vector<double> node_weights;
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(2);
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};
  auto cluster_ids =
      NearestNeighborLinkage(*gbbs_graph.Graph(), 0.0, size_constraint_config);
  ASSERT_OK(cluster_ids);
  EXPECT_THAT(parlay::remove_duplicates(cluster_ids.value()),
              UnorderedElementsAreArray<uintE>({0, 1, 2, 3}));
  EXPECT_THAT(CanonicalizeClustering(ComputeClusters(
                  cluster_ids.value(), [&](uintE i) { return true; })),
              ElementsAreArray<Cluster>({{0, 6}, {1, 3}, {2, 4}, {5}}));
}

TEST_F(ComputeClustersTest, ConvertsToClusters) {
  std::vector<uintE> cluster_ids = {0};
  auto is_active = [&](gbbs::uintE i) { return false; };
  EXPECT_THAT(ComputeClusters(cluster_ids, is_active), IsEmpty());

  cluster_ids = {0, 1, 2};
  auto finished_vertices =
      std::unique_ptr<bool[]>(new bool[3]{true, true, true});
  auto is_finished = [&](gbbs::uintE i) { return finished_vertices[i]; };
  EXPECT_THAT(ComputeClusters(cluster_ids, is_finished),
              ElementsAreArray<Cluster>({{0}, {1}, {2}}));

  cluster_ids = {3, 0, 3, 3, 0, UINT_E_MAX};
  finished_vertices =
      std::unique_ptr<bool[]>(new bool[6]{true, true, true, true, true, false});
  EXPECT_THAT(ComputeClusters(cluster_ids, is_finished),
              ElementsAreArray<Cluster>({{1, 4}, {0, 2, 3}}));
}

TEST_F(FindFinishedClustersTest, NoConditions) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 2;
  int num_edges = 2;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{2}),
                               std::make_tuple(uintE{0}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 1}, 1)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  AffinityClustererConfig config;

  std::vector<uintE> cluster_ids = {0, 0};
  std::vector<uintE> compressed_cluster_ids = {0, 0};
  EXPECT_THAT(
      FindFinishedClusters(G, config, cluster_ids, compressed_cluster_ids),
      IsEmpty());

  cluster_ids = {0, 1};
  compressed_cluster_ids = {0, 1};
  EXPECT_THAT(
      FindFinishedClusters(G, config, cluster_ids, compressed_cluster_ids),
      IsEmpty());
}

TEST_F(FindFinishedClustersTest, SingleCondition) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{3}, float{2}),
       std::make_tuple(uintE{2}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[5]), gbbs::vertex_data{0, 1}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  // cluster 0 of density 1 and conductance 0.5
  // cluster 1 of density 2 and conductance 0.5
  std::vector<uintE> cluster_ids = {0, 0, 1, 1};
  std::vector<uintE> compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.0 }"),
          cluster_ids, compressed_cluster_ids),
      IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.1 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_conductance: 0.5 }"),
                  cluster_ids, compressed_cluster_ids),
              IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_conductance: 0.6 }"),
                  cluster_ids, compressed_cluster_ids),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX,
                                                    UINT_E_MAX, UINT_E_MAX}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>(
                  {UINT_E_MAX, UINT_E_MAX, UINT_E_MAX, UINT_E_MAX}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.0 "
                              "min_conductance: 0.5 }"),
          cluster_ids, compressed_cluster_ids),
      IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.0 "
                              "min_conductance: 0.6 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX,
                                                    UINT_E_MAX, UINT_E_MAX}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>(
                  {UINT_E_MAX, UINT_E_MAX, UINT_E_MAX, UINT_E_MAX}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.1 "
                              "min_conductance: 0.5 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie("active_cluster_conditions { min_density: 1.1 "
                              "min_conductance: 0.6 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX,
                                                    UINT_E_MAX, UINT_E_MAX}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>(
                  {UINT_E_MAX, UINT_E_MAX, UINT_E_MAX, UINT_E_MAX}));
}

TEST_F(FindFinishedClustersTest, MultipleConditions) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{3}, float{2}),
       std::make_tuple(uintE{2}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[5]), gbbs::vertex_data{0, 1}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  // cluster 0 of density 1 and conductance 0.5
  // cluster 1 of density 2 and conductance 0.5
  std::vector<uintE> cluster_ids = {0, 0, 1, 1};
  std::vector<uintE> compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_density: 1.0 }"
                      "active_cluster_conditions { min_conductance: 0.6 }"),
                  cluster_ids, compressed_cluster_ids),
              IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_density: 1.1 }"
                      "active_cluster_conditions { min_conductance: 0.6 }"
                      "active_cluster_conditions { min_conductance: 0.5 }"),
                  cluster_ids, compressed_cluster_ids),
              IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie(
              "active_cluster_conditions { min_density: 1.1 }"
              "active_cluster_conditions { min_conductance: 0.6 }"
              "active_cluster_conditions { min_conductance: 0.5 min_density: "
              "1.1 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
}

TEST_F(FindFinishedClustersTest, ConditionsWithCompressedIds) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{3}, float{2}),
       std::make_tuple(uintE{2}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[5]), gbbs::vertex_data{0, 1}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  // cluster 0 of density 1 and conductance 0.5
  // cluster 1 of density 2 and conductance 0.5
  std::vector<uintE> cluster_ids = {0, 0, 1, 1};
  std::vector<uintE> compressed_cluster_ids = {0, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_density: 1.0 }"
                      "active_cluster_conditions { min_conductance: 0.6 }"),
                  cluster_ids, compressed_cluster_ids),
              IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 0, 1};
  EXPECT_THAT(FindFinishedClusters(
                  G,
                  ParseTextProtoOrDie(
                      "active_cluster_conditions { min_density: 1.0 }"
                      "active_cluster_conditions { min_conductance: 0.6 }"),
                  cluster_ids, compressed_cluster_ids),
              IsEmpty());
  EXPECT_THAT(cluster_ids, ElementsAreArray<uintE>({0, 0, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({0, 0, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie(
              "active_cluster_conditions { min_density: 1.1 }"
              "active_cluster_conditions { min_conductance: 0.6 }"
              "active_cluster_conditions { min_conductance: 0.5 min_density: "
              "1.1 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids, ElementsAreArray<uintE>({UINT_E_MAX, 1}));

  cluster_ids = {0, 0, 1, 1};
  compressed_cluster_ids = {0, 1, 1};
  EXPECT_THAT(
      FindFinishedClusters(
          G,
          ParseTextProtoOrDie(
              "active_cluster_conditions { min_density: 1.1 }"
              "active_cluster_conditions { min_conductance: 0.6 }"
              "active_cluster_conditions { min_conductance: 0.5 min_density: "
              "1.1 }"),
          cluster_ids, compressed_cluster_ids),
      ElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, UINT_E_MAX, 1, 1}));
  EXPECT_THAT(compressed_cluster_ids,
              ElementsAreArray<uintE>({UINT_E_MAX, 1, 1}));
}

std::vector<testing::Matcher<internal::ClusterStats>> GetClusterStatsMatcher(
    const std::vector<internal::ClusterStats>& expected_stats) {
  std::vector<testing::Matcher<internal::ClusterStats>> stats_matcher;
  for (const auto& stats : expected_stats) {
    stats_matcher.push_back(testing::AllOf(
        testing::Field(&internal::ClusterStats::density, stats.density),
        testing::Field(&internal::ClusterStats::conductance,
                       stats.conductance)));
  }
  return stats_matcher;
}

TEST_F(ComputeFinishedClusterStatsTest, TwoNodes) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 2;
  int num_edges = 2;
  std::vector<GbbsEdge> edges({std::make_tuple(uintE{1}, float{3}),
                               std::make_tuple(uintE{0}, float{3})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 1}, 0),
       gbbs::symmetric_vertex<float>(&(edges[1]), gbbs::vertex_data{0, 1}, 1)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {0, 1};
  std::vector<internal::ClusterStats> expected_stats = {{0, 1}, {0, 1}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 2),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));

  cluster_ids = {0, 0};
  expected_stats = {{3, 1}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 1),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));
}

TEST_F(ComputeFinishedClusterStatsTest, ThreeNodes) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 3;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{2}, float{3}),
       std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{1}, float{3})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 2}, 1),
       gbbs::symmetric_vertex<float>(&(edges[4]), gbbs::vertex_data{0, 2}, 2)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {0, 1, 1};
  std::vector<internal::ClusterStats> expected_stats = {{0, 1}, {3, 1}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 2),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));

  cluster_ids = {0, 0, 0};
  expected_stats = {{2, 1}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 1),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));

  cluster_ids = {0, 0, 1};
  expected_stats = {{1, 1}, {0, 1}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 2),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));
}

// Smallest test with nontrivial conductance
TEST_F(ComputeFinishedClusterStatsTest, FourNodes) {
  using GbbsEdge = std::tuple<uintE, float>;
  int num_vertices = 4;
  int num_edges = 6;
  std::vector<GbbsEdge> edges(
      {std::make_tuple(uintE{1}, float{1}), std::make_tuple(uintE{2}, float{2}),
       std::make_tuple(uintE{0}, float{1}), std::make_tuple(uintE{0}, float{2}),
       std::make_tuple(uintE{3}, float{2}),
       std::make_tuple(uintE{2}, float{2})});
  std::vector<gbbs::symmetric_vertex<float>> v(
      {gbbs::symmetric_vertex<float>(&(edges[0]), gbbs::vertex_data{0, 2}, 0),
       gbbs::symmetric_vertex<float>(&(edges[2]), gbbs::vertex_data{0, 1}, 1),
       gbbs::symmetric_vertex<float>(&(edges[3]), gbbs::vertex_data{0, 2}, 2),
       gbbs::symmetric_vertex<float>(&(edges[5]), gbbs::vertex_data{0, 1}, 3)});
  auto G = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      num_vertices, num_edges, v.data(), []() {});

  std::vector<uintE> cluster_ids = {0, 0, 1, 1};
  std::vector<internal::ClusterStats> expected_stats = {{1, 0.5}, {2, 0.5}};
  EXPECT_THAT(internal::ComputeFinishedClusterStats(G, cluster_ids, 2),
              ElementsAreArray(GetClusterStatsMatcher(expected_stats)));
}

TEST_F(EnforceMaxClusterSizeTest, NoSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  std::vector<double> node_weights{1, 2, 3, 1, 2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 1, 0, 3, 0, 1};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{2, 1}, {5, 1}, {0, 1},
                                                  {3, 0}, {2, 2}, {1, 1}};

  // SUT.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 1, 0, 3, 0, 1}));
}

TEST_F(EnforceMaxClusterSizeTest, MinSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_min_cluster_size(2);
  size_constraint.set_prefer_min_cluster_size(true);
  std::vector<double> node_weights{1, 2, 3, 1, 2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 1, 0, 3, 0, 1};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{2, 1}, {5, 1}, {0, 1},
                                                  {3, 0}, {2, 2}, {1, 1}};

  // SUT.
  // The min cluster size contraint prevents nodes 1 and 5 to form a cluster. It
  // also prevents nodes 4 to conform a cluster with nodes 0 and 2.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 1, 0, 3, 4, 5}));
}

TEST_F(EnforceMaxClusterSizeTest, MaxSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(5);
  std::vector<double> node_weights{1, 2, 3, 1, 2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 1, 0, 3, 0, 1};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{2, 1}, {5, 1}, {0, 1},
                                                  {3, 0}, {2, 2}, {1, 1}};

  // SUT.
  // The max cluster size contraint prevents nodes 0 to conform a cluster with
  // nodes 2 and 4 (nodes 2 and 4 are joined first because the best neighbor
  // edge connecting the two has a higher weight).
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 1, 2, 3, 2, 1}));
}

TEST_F(EnforceMaxClusterSizeTest, MinMaxSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(5);
  size_constraint.set_min_cluster_size(2);
  size_constraint.set_prefer_min_cluster_size(true);
  std::vector<double> node_weights{1, 2, 3, 1, 2, 2, 1, 1};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 1, 0, 3, 0, 1, 0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{
      {2, 1}, {5, 1}, {0, 1}, {3, 0}, {2, 2}, {1, 1}, {0, 1}, {0, 1}};

  // SUT.
  //
  // The min cluster size constraint prevents nodes 1 and 5 to form a cluster.
  // The min cluster size constraint also prevents node 4 to join nodes 0, 2,
  // and 6 to form a cluster.
  //
  // The max cluster size constraint prevents node 7 to join nodes 0, 2, and 6
  // to form a cluster.
  //
  // This test also demonstrates that min_cluster_size has precedence over
  // max_cluster_size. Otherwise, nodes 2 and 4 should be able to join the same
  // cluster because the best neighbor edge connecting the two has the highest
  // edge weight and thus this pair is considered prior to the other nodes 0, 6,
  // and 7 sharing the same original input cluster.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 1, 0, 3, 4, 5, 0, 7}));
}

TEST_F(EnforceMaxClusterSizeTest, TargetSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_target_cluster_size(5);

  std::vector<double> node_weights{4, 4, 2, 2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0, 0, 0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{
      {1, 3}, {0, 3}, {1, 2.1}, {2, 2}, {3, 1}};

  // SUT.
  //
  // The affinity tree is 4->3->2->1<->0
  //
  // The target cluster size constraint prevents nodes 4,3,2 to join nodes 1,0.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 0, 2, 2, 2}));
}

TEST_F(EnforceMaxClusterSizeTest, UnweightedNodesTargetSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_target_cluster_size(2);

  std::vector<double> node_weights;
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0, 0, 0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{
      {1, 4}, {0, 4}, {1, 3}, {2, 2}, {3, 1}};

  // SUT.
  //
  // The affinity tree is 4->3->2->1<->0
  //
  // The target cluster size constraint prevents nodes 4,3,2 to join nodes 1,0.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 0, 0, 3, 3}));
}

TEST_F(EnforceMaxClusterSizeTest, StarsTargetSizeConstraint) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_target_cluster_size(3);

  std::vector<double> node_weights{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{
      {1, 5}, {0, 5}, {0, 4}, {0, 3}, {0, 2},
      {6, 5}, {5, 5}, {5, 4}, {5, 3}, {5, 2}};

  // SUT.
  //
  // There are two stars the first with center at 0 and the second with the
  // center at 5.
  //
  // The target cluster size constraint prevents nodes 4,3 to join nodes 2,1,0,
  // and prevents nodes 9,8 to join nodes 7,6,5.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 0, 0, 3, 3, 5, 5, 5, 8, 8}));
}

TEST_F(EnforceMaxClusterSizeTest, MaxSizeConstraintTestTailingCompression) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(5);
  std::vector<double> node_weights{1, 1, 1};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{1, 1}, {2, 2}, {1, 2}};

  // SUT.
  // The three nodes belong to the same cluster before and after the size
  // constraint logic. This test is for verifying that the tailing
  // find_compress_atomic logic works as intended. Without that, the size
  // constraint logic would incorrectly break down the nodes into two clusters.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 0, 0}));
}

TEST_F(EnforceMaxClusterSizeTest, MaxSizeConstraintSelfEdge) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(5);
  std::vector<double> node_weights{2, 2};
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{0, 2}, {0, 1}};

  // SUT.
  // The two nodes belong to the same cluster before and after the size
  // constraint logic. This test is for verifying that the self-edge skipping
  // logic works as intended. Without that, the size constraint logic would
  // incorrectly break down the nodes into two clusters, because of
  // double-counting of the parent node weight.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 0}));
}

TEST_F(EnforceMaxClusterSizeTest, EmptyNodeWeights) {
  // Prepare.
  graph_mining::in_memory::AffinityClustererConfig::SizeConstraint
      size_constraint;
  size_constraint.set_max_cluster_size(1);
  std::vector<double> node_weights;
  internal::SizeConstraintConfig size_constraint_config{size_constraint,
                                                        node_weights};

  parlay::sequence<gbbs::uintE> cluster_ids{0, 0};
  AsynchronousUnionFind<gbbs::uintE> labels(std::move(cluster_ids));
  parlay::sequence<internal::Edge> best_neighbors{{1, 1}, {0, 1}};

  // SUT.
  //
  // Empty node weights mean that all nodes have default weights. Given the max
  // constraint set to 1, nodes 0 and 1 cannot form one cluster.
  EXPECT_THAT(
      EnforceMaxClusterSize(size_constraint_config, labels.ComponentIds(),
                            std::move(best_neighbors)),
      ElementsAreArray<uintE>({0, 1}));
}

}  // namespace
}  // namespace graph_mining::in_memory
