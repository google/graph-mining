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

#include "in_memory/clustering/affinity/parallel_affinity.h"

#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/parse_proto/parse_text_proto.h"

namespace graph_mining::in_memory {
namespace {

using NodeId = InMemoryClusterer::NodeId;
using ::graph_mining::in_memory::CanonicalizeClustering;
using ::testing::ElementsAreArray;

using Cluster = std::initializer_list<InMemoryClusterer::NodeId>;

std::vector<std::tuple<NodeId, NodeId, double>> GetAggregationTestGraph() {
  return {
      {0, 1, 5.0},  // forms cluster1 in round1.
      {2, 3, 5.0},  // forms cluster2 in round1.
      {4, 5, 5.0},  // forms cluster3 in round1.
      {6, 7, 5.0},  // forms cluster4 in round1.
      {0, 2, 2.0},  // cluster1 -> cluster2.
      {0, 3, 2.0},  // cluster1 -> cluster2.
      {1, 2, 1.0},  // cluster1 -> cluster2.
      {1, 3, 2.0},  // cluster1 -> cluster2.
      {4, 6, 2.0},  // cluster3 -> cluster4.
      {4, 7, 1.0},  // cluster3 -> cluster4.
      {5, 7, 2.0},  // cluster3 -> cluster4.
      {0, 4, 3.0},  // cluster1 -> cluster3.
      {0, 5, 1.0},  // cluster1 -> cluster3.
      {2, 6, 3.0},  // cluster2 -> cluster4.
      {3, 7, 1.0},  // cluster2 -> cluster4.
      {1, 7, 2.0},  // cluster1 -> cluster4.
      {3, 5, 2.0},  // cluster2 -> cluster3.
  };
}

TEST(ParallelAffinityTest, NoEdges) {
  SimpleUndirectedGraph graph;
  graph.AddNode();
  graph.AddNode();
  graph.AddNode();
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());
  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(graph_mining::in_memory::ClustererConfig()));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0}, {1}, {2}}));
}

TEST(ParallelAffinityTest, NumIterations) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 0 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0}, {1}, {2}, {3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 1 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, Hierarchy) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(auto clustering_hierarchy,
                       clusterer->HierarchicalFlatCluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  ASSERT_EQ(clustering_hierarchy.size(), 2);
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[0]),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[1]),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, HierarchyZeroIterations) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(auto clustering_hierarchy,
                       clusterer->HierarchicalFlatCluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 0 }")));
  ASSERT_EQ(clustering_hierarchy.size(), 1);
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[0]),
              ElementsAreArray<Cluster>({{0}, {1}, {2}, {3}}));
}

TEST(ParallelAffinityTest, HierarchyFinishEarly) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(
      auto clustering_hierarchy,
      clusterer->HierarchicalFlatCluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { active_cluster_conditions { "
          "min_density: 3.0 } num_iterations: 2 }")));
  ASSERT_EQ(clustering_hierarchy.size(), 1);
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[0]),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
}

TEST(ParallelAffinityTest, WeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { weight_threshold: 2.0 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}}));

  ASSERT_OK_AND_ASSIGN(
      clustering, clusterer->Cluster(PARSE_TEXT_PROTO(
                      "affinity_clusterer_config { weight_threshold: 2.1 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0}, {1}}));
}

TEST(ParallelAffinityTest, MultiWeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 10.0));
  ASSERT_OK(graph.AddEdge(2, 3, 10.0));
  ASSERT_OK(graph.AddEdge(0, 2, 4.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(
      Clustering clustering, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
        affinity_clusterer_config {
          num_iterations: 2
          edge_aggregation_function: MAX
          per_iteration_weight_thresholds: { thresholds: 10.0 thresholds: 4.0 }
        })pb")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));

  // Prevent a complete clustering in the second round with a too-high second
  // weight threshold.
  ASSERT_OK_AND_ASSIGN(clustering, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      edge_aggregation_function: MAX
      per_iteration_weight_thresholds: { thresholds: 10.0 thresholds: 4.1 }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
}

TEST(ParallelAffinityTest, WeightThresholdWithInitialNodeWeightDisabled) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 20.0));
  ASSERT_OK(graph.AddEdge(2, 3, 20.0));
  ASSERT_OK(graph.AddEdge(0, 2, 10.0));
  graph.SetNodeWeight(1, 2.0);
  graph.SetNodeWeight(3, 2.0);
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());
  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { edge_aggregation_function: "
          "DEFAULT_AVERAGE weight_threshold: 2.0 num_iterations: 2 "
          "use_node_weight_for_cluster_size: false}")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, WeightThresholdWithInitialNodeWeightEnabled) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 20.0));
  ASSERT_OK(graph.AddEdge(2, 3, 20.0));
  ASSERT_OK(graph.AddEdge(0, 2, 10.0));
  graph.SetNodeWeight(1, 2.0);
  graph.SetNodeWeight(3, 2.0);
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());
  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { edge_aggregation_function: "
          "DEFAULT_AVERAGE weight_threshold: 2.0 num_iterations: 2}")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
}

TEST(ParallelAffinityTest, EdgeAggregationFunction) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 10.0));
  ASSERT_OK(graph.AddEdge(0, 2, 4.0));
  ASSERT_OK(graph.AddEdge(2, 3, 10.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 1.1 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 1.0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: MAX weight_threshold: "
                           "4.1 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: MAX weight_threshold: "
                           "4.0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, OutputClusterEarly) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { active_cluster_conditions { "
          "min_density: 3.0 } num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, OutputClusterEarlyInIntermediateStep) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 20.0));
  ASSERT_OK(graph.AddEdge(1, 2, 5.0));
  ASSERT_OK(graph.AddEdge(2, 3, 10.0));
  ASSERT_OK(graph.AddEdge(3, 4, 5.0));
  ASSERT_OK(graph.AddEdge(4, 5, 20.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(
      Clustering clustering,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { active_cluster_conditions { "
          "min_density: 11.0 } num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}, {4, 5}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3, 4, 5}}));
}

TEST(ParallelAffinityTest, DynamicWeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
                         affinity_clusterer_config {
                           num_iterations: 2
                           dynamic_weight_threshold_config {
                             weight_decay_function: EXPONENTIAL_DECAY
                             upper_bound: 2.0
                             lower_bound: 0.51
                           }
                         })pb")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2}}));

  ASSERT_OK_AND_ASSIGN(clustering, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      dynamic_weight_threshold_config {
        weight_decay_function: EXPONENTIAL_DECAY
        upper_bound: 2.0
        lower_bound: 0.5
      }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2}}));
}

TEST(ParallelAffinityTest, MaxEdgeAggregation) {
  SimpleUndirectedGraph graph;
  auto edges = GetAggregationTestGraph();
  for (auto& edge : edges) {
    ASSERT_OK(
        graph.AddEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge)));
  }
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: MAX "
                           "weight_threshold: 0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 4, 5}, {2, 3, 6, 7}}));
}

TEST(ParallelAffinityTest, SumEdgeAggregation) {
  SimpleUndirectedGraph graph;
  auto edges = GetAggregationTestGraph();
  for (auto& edge : edges) {
    ASSERT_OK(
        graph.AddEdge(std::get<0>(edge), std::get<1>(edge), std::get<2>(edge)));
  }
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: SUM "
                           "weight_threshold: 0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}, {4, 5, 6, 7}}));
}

TEST(ParallelAffinityTest, CutSparsityEdgeAggregation) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));  // contracted in the first round
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));  // contracted in the first round
  ASSERT_OK(graph.AddEdge(0, 2, 1.0));
  ASSERT_OK(graph.AddEdge(1, 3, 0.5));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: CUT_SPARSITY "
                           "weight_threshold: 0.76 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: CUT_SPARSITY "
                           "weight_threshold: 0.75 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(ParallelAffinityTest, AverageEdgeAggregation) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));  // contracted in the first round
  ASSERT_OK(graph.AddEdge(1, 2, 2.0));  // contracted in the first round
  ASSERT_OK(graph.AddEdge(3, 4, 2.0));  // contracted in the first round
  ASSERT_OK(graph.AddEdge(3, 0, 1));
  ASSERT_OK(graph.AddEdge(4, 2, 0.5));
  auto clusterer = std::make_unique<ParallelAffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK(clusterer->MutableGraph()->FinishImport());

  ASSERT_OK_AND_ASSIGN(Clustering clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 0.26 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2}, {3, 4}}));

  ASSERT_OK_AND_ASSIGN(clustering,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 0.24 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clustering),
              ElementsAreArray<Cluster>({{0, 1, 2, 3, 4}}));
}

}  // namespace
}  // namespace graph_mining::in_memory
