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

#include "in_memory/clustering/affinity/affinity_internal.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/parse_proto/parse_text_proto.h"
#include "src/farmhash.h"

namespace graph_mining::in_memory {
namespace {

using ::graph_mining::in_memory::AffinityClustererConfig;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::Pair;

using NodeId = InMemoryClusterer::NodeId;

TEST(FlattenClusteringTest, FlattensClusters) {
  std::vector<NodeId> cluster_ids = {0};
  std::vector<NodeId> compressed_cluster_ids = {2};
  EXPECT_THAT(FlattenClustering(cluster_ids, compressed_cluster_ids),
              ElementsAre(2));
  cluster_ids = {0, 2, -1};
  compressed_cluster_ids = {0, 2, 1};
  EXPECT_THAT(FlattenClustering(cluster_ids, compressed_cluster_ids),
              ElementsAre(0, 1, -1));
}

std::vector<std::pair<AffinityClustererConfig, float>>
CreateAffinityTestScenarios(std::vector<std::pair<absl::string_view, float>>
                                string_configs_and_weights) {
  std::vector<std::pair<AffinityClustererConfig, float>> configs_and_weights;

  for (const auto& [string_config, weight] : string_configs_and_weights) {
    configs_and_weights.push_back({PARSE_TEXT_PROTO(string_config), weight});
  }

  return configs_and_weights;
}

TEST(CompressGraphTest, EdgeAggregationThreeNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  std::vector<NodeId> cluster_ids = {0, 2, 2};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.5},
            {"edge_aggregation_function: EXPLICIT_AVERAGE weight_threshold: "
             "1.0",
             1.5},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 3.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 2.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             3.0}})) {
    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSERT_OK_AND_ASSIGN(compressed_graph,
                         CompressGraph(graph, cluster_ids, clusterer_config));
    ASSERT_EQ(compressed_graph->NumNodes(), 3);
    EXPECT_THAT(compressed_graph->Neighbors(0),
                ElementsAre(Pair(2, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(1), IsEmpty());
    EXPECT_THAT(compressed_graph->Neighbors(2),
                ElementsAre(Pair(0, expected_weight)));
  }
}

TEST(CompressGraphTest, EdgeAggregationFourNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(1, 2, 2.0));
  ASSERT_OK(graph.AddEdge(2, 3, 3.0));
  ASSERT_OK(graph.AddEdge(3, 0, 4.0));

  std::vector<NodeId> cluster_ids = {1, 1, 3, 3};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: DEFAULT_AVERAGE weight_threshold: 1.0",
             1.5},
            {"edge_aggregation_function: EXPLICIT_AVERAGE weight_threshold: "
             "1.0",
             3.0},
            {"edge_aggregation_function: SUM weight_threshold: 1.0", 6.0},
            {"edge_aggregation_function: MAX weight_threshold: 1.0", 4.0},
            {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
             3.0}})) {
    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSERT_OK_AND_ASSIGN(compressed_graph,
                         CompressGraph(graph, cluster_ids, clusterer_config));
    EXPECT_EQ(compressed_graph->NumNodes(), 4);
    EXPECT_THAT(compressed_graph->Neighbors(0), IsEmpty());
    EXPECT_THAT(compressed_graph->Neighbors(1),
                ElementsAre(Pair(3, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(2), IsEmpty());
    EXPECT_THAT(compressed_graph->Neighbors(3),
                ElementsAre(Pair(1, expected_weight)));
  }
}

// Expect MAX and PERCENTILE with percentile linkage value 1.0 to be equally.
TEST(CompressGraphTest, EdgeAggregationMaxAndPercentile) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  std::vector<NodeId> cluster_ids = {0, 2, 2};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: MAX weight_threshold: 1.0", 2.0},
            {"edge_aggregation_function: PERCENTILE weight_threshold: 1.0 "
             "percentile_linkage_value: 1.0 "
             "min_edge_count_for_percentile_linkage: 2",
             2.0}})) {
    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSERT_OK_AND_ASSIGN(compressed_graph,
                         CompressGraph(graph, cluster_ids, clusterer_config));
    ASSERT_EQ(compressed_graph->NumNodes(), 3);
    EXPECT_THAT(compressed_graph->Neighbors(0),
                ElementsAre(Pair(2, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(1), IsEmpty());
    EXPECT_THAT(compressed_graph->Neighbors(2),
                ElementsAre(Pair(0, expected_weight)));
  }
}

// Expect PERCENTILE linkage to pick:
// - the lowest weight when percentile linkage value is 0
// - the middle weight when percentile linkage value is 0.5
// - the highest weight when percentile linkage value is 1.0
TEST(CompressGraphTest, EdgeAggregationPercentile) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));
  ASSERT_OK(graph.AddEdge(0, 3, 3.0));

  std::vector<NodeId> cluster_ids = {0, 2, 2, 2};

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios(
           {{"edge_aggregation_function: PERCENTILE weight_threshold: 0.0 "
             "percentile_linkage_value: 0.0 "
             "min_edge_count_for_percentile_linkage: 2",
             1.0},
            {"edge_aggregation_function: PERCENTILE weight_threshold: 0.0 "
             "percentile_linkage_value: 0.5 "
             "min_edge_count_for_percentile_linkage: 2",
             2.0},
            {"edge_aggregation_function: PERCENTILE weight_threshold: 0.0 "
             "percentile_linkage_value: 1.0 "
             "min_edge_count_for_percentile_linkage: 2",
             3.0}})) {
    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSERT_OK_AND_ASSIGN(compressed_graph,
                         CompressGraph(graph, cluster_ids, clusterer_config));
    ASSERT_EQ(compressed_graph->NumNodes(), 4);
    EXPECT_THAT(compressed_graph->Neighbors(0),
                ElementsAre(Pair(2, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(1), IsEmpty());
    EXPECT_THAT(compressed_graph->Neighbors(2),
                ElementsAre(Pair(0, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(3), IsEmpty());
  }
}

TEST(CompressGraphTest, RemoveNode) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(1, 2, 2.0));

  std::vector<NodeId> cluster_ids = {0, 2, -1};
  std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
  ASSERT_OK_AND_ASSIGN(
      compressed_graph,
      CompressGraph(graph, cluster_ids,
                    PARSE_TEXT_PROTO("edge_aggregation_function: SUM")));
  EXPECT_EQ(compressed_graph->NumNodes(), 3);
  EXPECT_THAT(compressed_graph->Neighbors(0), ElementsAre(Pair(2, 1.0)));
  EXPECT_THAT(compressed_graph->Neighbors(1), IsEmpty());
  EXPECT_THAT(compressed_graph->Neighbors(2), ElementsAre(Pair(0, 1.0)));
}

TEST(CompressGraphTest, RemoveTwoNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(1, 2, 2.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  std::vector<NodeId> cluster_ids = {-1, 2, -1};
  std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
  ASSERT_OK_AND_ASSIGN(
      compressed_graph,
      CompressGraph(graph, cluster_ids,
                    PARSE_TEXT_PROTO("edge_aggregation_function: SUM")));
  EXPECT_EQ(compressed_graph->NumNodes(), 3);
  EXPECT_THAT(compressed_graph->Neighbors(0), IsEmpty());
  EXPECT_THAT(compressed_graph->Neighbors(1), IsEmpty());
  EXPECT_THAT(compressed_graph->Neighbors(2), IsEmpty());
}

std::string StringId(NodeId node_id) { return std::to_string(node_id); }

TEST(NearestNeighborLinkageTest, NoEdges) {
  SimpleUndirectedGraph graph;
  graph.SetNumNodes(3);
  std::vector<NodeId> cluster_ids = NearestNeighborLinkage(
      graph, 1.0, [](NodeId id) { return StringId(id); });
  ASSERT_EQ(cluster_ids.size(), 3);
  EXPECT_EQ(cluster_ids[0], 0);
  EXPECT_EQ(cluster_ids[1], 1);
  EXPECT_EQ(cluster_ids[2], 2);
}

TEST(NearestNeighborLinkageTest, AppliesWeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));

  std::vector<NodeId> cluster_ids = NearestNeighborLinkage(
      graph, 1.0, [](NodeId id) { return StringId(id); });
  ASSERT_EQ(cluster_ids.size(), 2);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_GE(cluster_ids[0], 0);
  EXPECT_LE(cluster_ids[0], 1);
  cluster_ids = NearestNeighborLinkage(graph, 1.1,
                                       [](NodeId id) { return StringId(id); });

  ASSERT_EQ(cluster_ids.size(), 2);
  EXPECT_NE(cluster_ids[0], cluster_ids[1]);
}

TEST(NearestNeighborLinkageTest, ChoosesBestNeighbor) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 3.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 3.0));

  std::vector<NodeId> cluster_ids = NearestNeighborLinkage(
      graph, 0.0, [](NodeId id) { return StringId(id); });
  ASSERT_EQ(cluster_ids.size(), 4);
  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_EQ(cluster_ids[2], cluster_ids[3]);
  EXPECT_NE(cluster_ids[1], cluster_ids[2]);
}

TEST(NearestNeighborLinkageTest, BreaksTies) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 1.0));
  ASSERT_OK(graph.AddEdge(1, 3, 5.0));
  ASSERT_OK(graph.AddEdge(2, 4, 5.0));

  std::vector<NodeId> cluster_ids = NearestNeighborLinkage(
      graph, 0.0, [](NodeId id) { return StringId(id); });
  ASSERT_EQ(cluster_ids.size(), 5);
  // Clusters are {1, 3}, {0, 2, 4}, since 0 prefers to join 1, given that
  // farmhash::Fingerprint("2") < farmhash:::Fingerprint64("1")
  EXPECT_LT(util::Fingerprint64("2", 1), util::Fingerprint64("1", 1));

  EXPECT_EQ(cluster_ids[0], cluster_ids[1]);
  EXPECT_EQ(cluster_ids[2], cluster_ids[4]);
  EXPECT_EQ(cluster_ids[1], cluster_ids[3]);
  EXPECT_NE(cluster_ids[0], cluster_ids[2]);
}

using Cluster = std::initializer_list<InMemoryClusterer::NodeId>;

TEST(ComputeClustersTest, ConvertsToClusters) {
  std::vector<NodeId> cluster_ids = {-1};
  EXPECT_THAT(ComputeClusters(cluster_ids), IsEmpty());

  cluster_ids = {0, 1, 2};
  EXPECT_THAT(ComputeClusters(cluster_ids),
              ElementsAreArray<Cluster>({{0}, {1}, {2}}));

  cluster_ids = {3, 0, 3, 3, 0, -1};
  EXPECT_THAT(ComputeClusters(cluster_ids),
              ElementsAreArray<Cluster>({{1, 4}, {0, 2, 3}}));
}

TEST(ComputeClusterQualityIndicatorsTest, TwoNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 3.0));

  auto qi =
      ComputeClusterQualityIndicators(std::vector<NodeId>({0}), graph, 2 * 3.0);
  EXPECT_EQ(qi.density, 0.0);
  EXPECT_EQ(qi.conductance, 1.0);

  qi = ComputeClusterQualityIndicators(std::vector<NodeId>({0, 1}), graph,
                                       2 * 3.0);
  EXPECT_EQ(qi.density, 3.0);
  EXPECT_EQ(qi.conductance, 1.0);
}

TEST(ComputeClusterQualityIndicatorsTest, ThreeNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 3.0));

  auto qi = ComputeClusterQualityIndicators(std::vector<NodeId>({1, 2}), graph,
                                            2 * 6.0);
  EXPECT_EQ(qi.density, 3.0);
  EXPECT_EQ(qi.conductance, 1.0);

  qi = ComputeClusterQualityIndicators(std::vector<NodeId>({0, 1, 2}), graph,
                                       2 * 6.0);
  EXPECT_EQ(qi.density, 2.0);
  EXPECT_EQ(qi.conductance, 1.0);

  qi = ComputeClusterQualityIndicators(std::vector<NodeId>({0, 1}), graph,
                                       2 * 6.0);
  EXPECT_EQ(qi.density, 1.0);
  EXPECT_EQ(qi.conductance, 1.0);
}

TEST(CompressGraphTest, EdgeAggregationManyMergedEdges) {
  SimpleUndirectedGraph graph;
  // Two nodes 0 and 1, with nodes 2 ... n-1 connected to both. The even nodes
  // merge with node 0, and odd nodes merge with node 1.
  // 0 --- i --- 1
  size_t n = 8;
  std::vector<NodeId> cluster_ids(n);
  cluster_ids[0] = 0;
  cluster_ids[1] = 1;
  for (size_t i = 2; i < n; ++i) {
    ASSERT_OK(graph.AddEdge(0, i, i));
    ASSERT_OK(graph.AddEdge(1, i, i));
    cluster_ids[i] = i % 2;
  }

  for (auto [clusterer_config, expected_weight] : CreateAffinityTestScenarios({
           {"edge_aggregation_function: "
            "EXPLICIT_AVERAGE weight_threshold: 1.0",
            4.5},
           {"edge_aggregation_function: PERCENTILE weight_threshold: 1.0 "
            "percentile_linkage_value: 1.0 "
            "min_edge_count_for_percentile_linkage: 100",
            7.0},
           {"edge_aggregation_function: PERCENTILE weight_threshold: 1.0 "
            "percentile_linkage_value: 0.5 "
            "min_edge_count_for_percentile_linkage: 2",
            4.0},
           {"edge_aggregation_function: SUM weight_threshold: 1.0", 27.0},
           {"edge_aggregation_function: MAX weight_threshold: 1.0", 7.0},
           {"edge_aggregation_function: CUT_SPARSITY weight_threshold: 1.0",
            27.0 / 4},
       })) {
    std::unique_ptr<SimpleUndirectedGraph> compressed_graph;
    ASSERT_OK_AND_ASSIGN(compressed_graph,
                         CompressGraph(graph, cluster_ids, clusterer_config));
    EXPECT_EQ(compressed_graph->NumNodes(), n);
    EXPECT_THAT(compressed_graph->Neighbors(0),
                ElementsAre(Pair(1, expected_weight)));
    EXPECT_THAT(compressed_graph->Neighbors(1),
                ElementsAre(Pair(0, expected_weight)));
    for (size_t i = 2; i < n; ++i) {
      EXPECT_THAT(compressed_graph->Neighbors(i), IsEmpty());
    }
  }
}

// Smallest test with nontrivial conductance
TEST(ComputeClusterQualityIndicatorsTest, FourNodes) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  auto qi = ComputeClusterQualityIndicators(std::vector<NodeId>({0, 1}), graph,
                                            2 * 5.0);
  EXPECT_EQ(qi.density, 1.0);
  EXPECT_EQ(qi.conductance, 0.5);

  qi = ComputeClusterQualityIndicators(std::vector<NodeId>({2, 3}), graph,
                                       2 * 5.0);
  EXPECT_EQ(qi.density, 2.0);
  EXPECT_EQ(qi.conductance, 0.5);
}

TEST(IsActiveClusterTest, NoConditions) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  AffinityClustererConfig config;

  EXPECT_TRUE(IsActiveCluster(std::vector<NodeId>({0, 1}), graph, config,
                              /*graph_volume=*/4.0));
  EXPECT_TRUE(IsActiveCluster(std::vector<NodeId>({0}), graph, config,
                              /*graph_volume=*/4.0));
  EXPECT_TRUE(IsActiveCluster(std::vector<NodeId>({1}), graph, config,
                              /*graph_volume*/ 4.0));
}

TEST(IsActiveClusterTest, SingleCondition) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  // cluster of density 1 and conductance 0.5
  std::vector<NodeId> cluster({0, 1});

  EXPECT_TRUE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.0 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.1 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_TRUE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_conductance: 0.5 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_conductance: 0.6 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_TRUE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.0 "
                       "min_conductance: 0.5 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.0 "
                       "min_conductance: 0.6 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.1 "
                       "min_conductance: 0.5 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.1 "
                       "min_conductance: 0.6 }"),
      /*graph_volume*/ 2 * 5.0));
}

TEST(IsActiveClusterTest, MultipleConditions) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  ASSERT_OK(graph.AddEdge(0, 2, 2.0));

  // cluster of density 1 and conductance 0.5
  std::vector<NodeId> cluster({0, 1});

  EXPECT_TRUE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.0 }"
                       "active_cluster_conditions { min_conductance: 0.6 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_TRUE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO("active_cluster_conditions { min_density: 1.1 }"
                       "active_cluster_conditions { min_conductance: 0.6 }"
                       "active_cluster_conditions { min_conductance: 0.5 }"),
      /*graph_volume*/ 2 * 5.0));
  EXPECT_FALSE(IsActiveCluster(
      cluster, graph,
      PARSE_TEXT_PROTO(
          "active_cluster_conditions { min_density: 1.1 }"
          "active_cluster_conditions { min_conductance: 0.6 }"
          "active_cluster_conditions { min_conductance: 0.5 min_density: "
          "1.1 }"),
      /*graph_volume*/ 2 * 5.0));
}

}  // namespace
}  // namespace graph_mining::in_memory
