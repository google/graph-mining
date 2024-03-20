// Copyright 2024 Google LLC
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

#include "in_memory/clustering/dynamic/hac/hac.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dendrogram_test_utils.h"
#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/generation/add_edge_weights.h"
#include "in_memory/generation/erdos_renyi.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {
namespace {

using ParentEdge = DynamicDendrogram::ParentEdge;
using Cluster =
    std::initializer_list<graph_mining::in_memory::InMemoryClusterer::NodeId>;
using Clustering = graph_mining::in_memory::Clustering;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
using NodeColor = DynamicHacNodeColor::NodeColor;
using graph_mining::in_memory::Dendrogram;


using DynamicHacTest = ::testing::Test;

// Test the goodness of `dendrogram` with respect to `graph` and `config`.
void TestDendrogram(const SimpleUndirectedGraph& graph,
                    const DynamicHacConfig& config,
                    const Dendrogram& dendrogram,
                    std::vector<double> min_merge_similarities) {
  auto copy = std::make_unique<SimpleUndirectedGraph>();
  EXPECT_OK(graph_mining::in_memory::CopyGraph(graph, copy.get()));

  if (min_merge_similarities.empty()) {
    min_merge_similarities.resize(graph.NumNodes(),
                                  std::numeric_limits<double>::infinity());
  }

  auto goodness = DendrogramGoodness(
      dendrogram, *copy, config.weight_threshold(), min_merge_similarities);

  ASSERT_OK(goodness.status());

  if (min_merge_similarities.empty()) {
    EXPECT_THAT(ClosenessApproximationFactor(dendrogram, *copy,
                                             config.weight_threshold()),
                IsOkAndHolds(testing::DoubleNear(*goodness, 1e-6)));
  }

  double global_approx =
      graph_mining::in_memory::GlobalApproximationFactor(dendrogram);

  EXPECT_PRED_FORMAT2(testing::DoubleLE, *goodness,
                      1.0 + config.epsilon() + 0.000003)
      << " epsilon = " << config.epsilon()
      << " weight_threshold = " << config.weight_threshold()
      << " global approx = " << global_approx;
}

// Insert all edges from `graph` in one batch, and delete one node at a time.
void RunBatchInsertionAndDeletionTest(
    const SimpleUndirectedGraph& graph, const DynamicHacConfig& config,
    std::vector<double> min_merge_similarities = {}) {
  ASSERT_OK_AND_ASSIGN(const auto adj_lists, ConvertToAdjList(graph));
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  ASSERT_OK(clusterer->Insert(adj_lists));
  ASSERT_EQ(clusterer->Dendrogram().NumNodes(), graph.NumNodes());
  ASSERT_OK_AND_ASSIGN(auto dendrogram,
                       clusterer->Dendrogram().ConvertToDendrogram());
  TestDendrogram(graph, config, dendrogram.first, min_merge_similarities);

  const auto num_nodes = graph.NumNodes();
  NodeId last_node = num_nodes;  // Nodes >= `last_node` have been deleted.
  absl::BitGen gen;
  // Delete from node n-1 to 0.
  while (last_node > 0) {
    NodeId num_remove =
        absl::Uniform<NodeId>(absl::IntervalClosedClosed, gen, 1, 5);
    if (last_node < num_remove) num_remove = last_node;
    for (NodeId i = last_node - 1; i >= last_node - num_remove; i--) {
      ASSERT_OK(clusterer->Remove({i}));
      ASSERT_OK_AND_ASSIGN(auto dendrogram,
                           clusterer->Dendrogram().ConvertToDendrogram());

      ASSERT_EQ(clusterer->Dendrogram().NumNodes(), i);
      if (i > 0) {
        SimpleUndirectedGraph graph_test;
        graph_test.SetNumNodes(i);
        for (int j = 0; j < i; ++j) {
          for (const auto& [k, w] : graph.Neighbors(j)) {
            if (k < i) {
              ASSERT_OK(graph_test.AddEdge(j, k, w));
            }
          }
        }
        ABSL_CHECK(graph_test.NumNodes() == i);
        TestDendrogram(graph_test, config, dendrogram.first,
                       min_merge_similarities);
      }
    }
    last_node -= num_remove;
  }
}

// Insert a single node at a time.
void RunSingleInsertionTest(const SimpleUndirectedGraph& full_graph,
                            const DynamicHacConfig& config,
                            std::vector<double> min_merge_similarities = {}) {
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  ASSERT_OK(clusterer->Insert({{0, 1, {}}}));
  absl::flat_hash_set<NodeId> inserted_nodes{0};
  SimpleUndirectedGraph graph;
  graph.AddNode();
  std::size_t edges_inserted = 0;
  std::size_t num_total_edges = full_graph.Neighbors(0).size();

  // Insert from node 1 ... n-1.
  for (NodeId i = 1; i < full_graph.NumNodes(); ++i) {
    std::vector<AdjacencyList> adj_lists(1);
    adj_lists[0].id = i;
    adj_lists[0].weight = 1;
    inserted_nodes.insert(i);
    graph.AddNode();
    const auto& i_neighbors = full_graph.Neighbors(i);
    for (const auto& neighbor : i_neighbors) {
      if (inserted_nodes.contains(neighbor.first)) {
        ASSERT_OK(graph.AddEdge(i, neighbor.first, neighbor.second));
        adj_lists[0].outgoing_edges.push_back(neighbor);
      }
    }
    edges_inserted += 2 * adj_lists[0].outgoing_edges.size();
    num_total_edges += i_neighbors.size();

    ASSERT_OK(clusterer->Insert(adj_lists));
    ASSERT_EQ(clusterer->Dendrogram().NumNodes(), graph.NumNodes());
    ASSERT_OK_AND_ASSIGN(auto dendrogram,
                         clusterer->Dendrogram().ConvertToDendrogram());

    TestDendrogram(graph, config, dendrogram.first, min_merge_similarities);
  }
  // Check that all edges are inserted.
  EXPECT_EQ(edges_inserted, num_total_edges);
  EXPECT_EQ(clusterer->NumEdges(), num_total_edges);
}

DynamicHacConfig MakeConfig(double epsilon, double weight_threshold) {
  DynamicHacConfig result;
  result.set_epsilon(epsilon);
  result.set_weight_threshold(weight_threshold);
  return result;
}

TEST_F(DynamicHacTest, TestInsertSmallEdge) {
  size_t num_nodes = 100;
  double p = 0.1;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  const auto config = MakeConfig(0, 0);
  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(*graph));
  ABSL_LOG(INFO) << "Graph created";

  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  ASSERT_OK(clusterer->Insert(adj_lists));
  ABSL_LOG(INFO) << "Num Rounds = " << clusterer->NumRounds();
  ABSL_LOG(INFO) << "======================================================";

  ASSERT_OK(clusterer->Insert({{100000, 1, {{1, 0.01}}}}));
  ABSL_LOG(INFO) << "Num Rounds = " << clusterer->NumRounds();
}

TEST_F(DynamicHacTest, TestInactiveNodeWeight) {
  size_t num_nodes = 17;
  double multiplier = 1.5;  // scales from 0.01 to ~0.2.
  SimpleUndirectedGraph graph;
  graph.SetNumNodes(num_nodes);
  EXPECT_OK(graph.AddEdge(0, 11, 0.357352));
  EXPECT_OK(graph.AddEdge(1, 6, 0.480908));
  EXPECT_OK(graph.AddEdge(2, 15, 0.223191));
  EXPECT_OK(graph.AddEdge(2, 9, 0.572059));
  EXPECT_OK(graph.AddEdge(3, 12, 0.412659));
  EXPECT_OK(graph.AddEdge(4, 11, 0.146563));
  EXPECT_OK(graph.AddEdge(5, 14, 0.736268));
  EXPECT_OK(graph.AddEdge(6, 1, 0.480908));
  EXPECT_OK(graph.AddEdge(7, 13, 0.284956));
  EXPECT_OK(graph.AddEdge(9, 13, 0.566927));
  EXPECT_OK(graph.AddEdge(9, 10, 0.788921));
  EXPECT_OK(graph.AddEdge(9, 2, 0.572059));
  EXPECT_OK(graph.AddEdge(10, 9, 0.788921));
  EXPECT_OK(graph.AddEdge(11, 4, 0.146563));
  EXPECT_OK(graph.AddEdge(11, 0, 0.357352));
  EXPECT_OK(graph.AddEdge(12, 14, 0.301005));
  EXPECT_OK(graph.AddEdge(12, 3, 0.412659));
  EXPECT_OK(graph.AddEdge(13, 7, 0.284956));
  EXPECT_OK(graph.AddEdge(13, 9, 0.566927));
  EXPECT_OK(graph.AddEdge(14, 5, 0.736268));
  EXPECT_OK(graph.AddEdge(14, 12, 0.301005));
  EXPECT_OK(graph.AddEdge(15, 2, 0.223191));
  EXPECT_OK(graph.AddEdge(15, 16, 0.80117));
  EXPECT_OK(graph.AddEdge(16, 15, 0.80117));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon = epsilon_exponent < 0
                           ? 0.0
                           : 0.01 * std::pow(multiplier, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(graph, config);
      RunSingleInsertionTest(graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestInactiveNodeMerge) {
  size_t num_nodes = 23;
  double multiplier = 1.5;  // scales from 0.01 to ~0.2.
  SimpleUndirectedGraph graph;
  graph.SetNumNodes(num_nodes);
  EXPECT_OK(graph.AddEdge(0, 9, 0.709901));
  EXPECT_OK(graph.AddEdge(0, 4, 0.429697));
  EXPECT_OK(graph.AddEdge(0, 21, 0.0772143));
  EXPECT_OK(graph.AddEdge(1, 5, 0.167472));
  EXPECT_OK(graph.AddEdge(2, 3, 0.980541));
  EXPECT_OK(graph.AddEdge(2, 20, 0.0377098));
  EXPECT_OK(graph.AddEdge(2, 9, 0.525744));
  EXPECT_OK(graph.AddEdge(2, 12, 0.201285));
  EXPECT_OK(graph.AddEdge(3, 17, 0.201533));
  EXPECT_OK(graph.AddEdge(3, 15, 0.790602));
  EXPECT_OK(graph.AddEdge(3, 2, 0.980541));
  EXPECT_OK(graph.AddEdge(3, 22, 0.308301));
  EXPECT_OK(graph.AddEdge(4, 14, 0.0833762));
  EXPECT_OK(graph.AddEdge(4, 8, 0.627443));
  EXPECT_OK(graph.AddEdge(4, 0, 0.429697));
  EXPECT_OK(graph.AddEdge(4, 10, 0.334445));
  EXPECT_OK(graph.AddEdge(5, 15, 0.519206));
  EXPECT_OK(graph.AddEdge(5, 1, 0.167472));
  EXPECT_OK(graph.AddEdge(6, 17, 0.632601));
  EXPECT_OK(graph.AddEdge(7, 16, 0.447963));
  EXPECT_OK(graph.AddEdge(7, 20, 0.637021));
  EXPECT_OK(graph.AddEdge(7, 8, 0.397911));
  EXPECT_OK(graph.AddEdge(8, 16, 0.342045));
  EXPECT_OK(graph.AddEdge(8, 4, 0.627443));
  EXPECT_OK(graph.AddEdge(8, 7, 0.397911));
  EXPECT_OK(graph.AddEdge(9, 2, 0.525744));
  EXPECT_OK(graph.AddEdge(9, 0, 0.709901));
  EXPECT_OK(graph.AddEdge(10, 4, 0.334445));
  EXPECT_OK(graph.AddEdge(10, 18, 0.665931));
  EXPECT_OK(graph.AddEdge(11, 12, 0.196954));
  EXPECT_OK(graph.AddEdge(11, 15, 0.0856146));
  EXPECT_OK(graph.AddEdge(11, 21, 0.715773));
  EXPECT_OK(graph.AddEdge(12, 2, 0.201285));
  EXPECT_OK(graph.AddEdge(12, 11, 0.196954));
  EXPECT_OK(graph.AddEdge(14, 20, 0.246394));
  EXPECT_OK(graph.AddEdge(14, 4, 0.0833762));
  EXPECT_OK(graph.AddEdge(15, 5, 0.519206));
  EXPECT_OK(graph.AddEdge(15, 3, 0.790602));
  EXPECT_OK(graph.AddEdge(15, 11, 0.0856146));
  EXPECT_OK(graph.AddEdge(16, 8, 0.342045));
  EXPECT_OK(graph.AddEdge(16, 7, 0.447963));
  EXPECT_OK(graph.AddEdge(17, 3, 0.201533));
  EXPECT_OK(graph.AddEdge(17, 6, 0.632601));
  EXPECT_OK(graph.AddEdge(18, 10, 0.665931));
  EXPECT_OK(graph.AddEdge(20, 2, 0.0377098));
  EXPECT_OK(graph.AddEdge(20, 14, 0.246394));
  EXPECT_OK(graph.AddEdge(20, 7, 0.637021));
  EXPECT_OK(graph.AddEdge(21, 11, 0.715773));
  EXPECT_OK(graph.AddEdge(21, 0, 0.0772143));
  EXPECT_OK(graph.AddEdge(22, 3, 0.308301));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon = epsilon_exponent < 0
                           ? 0.0
                           : 0.01 * std::pow(multiplier, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(graph, config);
      RunSingleInsertionTest(graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestNeighboringDirtyPartitions) {
  size_t num_nodes = 20;
  double multiplier = 1.5;  // scales from 0.01 to ~0.2.
  SimpleUndirectedGraph graph;
  graph.SetNumNodes(num_nodes);
  EXPECT_OK(graph.AddEdge(6, 7, 0.773841));
  EXPECT_OK(graph.AddEdge(7, 9, 0.708377));
  EXPECT_OK(graph.AddEdge(7, 15, 0.513506));
  EXPECT_OK(graph.AddEdge(7, 6, 0.773841));
  EXPECT_OK(graph.AddEdge(7, 16, 0.367374));
  EXPECT_OK(graph.AddEdge(9, 7, 0.708377));
  EXPECT_OK(graph.AddEdge(9, 19, 0.685238));
  EXPECT_OK(graph.AddEdge(10, 15, 0.284053));
  EXPECT_OK(graph.AddEdge(10, 16, 0.296604));
  EXPECT_OK(graph.AddEdge(10, 4, 0.977272));
  EXPECT_OK(graph.AddEdge(15, 10, 0.284053));
  EXPECT_OK(graph.AddEdge(15, 7, 0.513506));
  EXPECT_OK(graph.AddEdge(15, 4, 0.136645));
  EXPECT_OK(graph.AddEdge(16, 7, 0.367374));
  EXPECT_OK(graph.AddEdge(16, 10, 0.296604));
  EXPECT_OK(graph.AddEdge(19, 9, 0.685238));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon = epsilon_exponent < 0
                           ? 0.0
                           : 0.01 * std::pow(multiplier, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(graph, config);
      RunSingleInsertionTest(graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestMultipleApproximationUnweightedNoisyErdosRenyi) {
  size_t num_nodes = 100;
  double p = 0.1;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 1, 1.001));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon =
          epsilon_exponent < 0 ? 0.0 : 0.01 * std::pow(1.5, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(*graph, config);
      RunSingleInsertionTest(*graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestMultipleApproximationUnweightedErdosRenyi) {
  size_t num_nodes = 100;
  double p = 0.1;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon =
          epsilon_exponent < 0 ? 0.0 : 0.01 * std::pow(1.5, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(*graph, config);
      RunSingleInsertionTest(*graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestMultipleApproximationWeightedErdosRenyi) {
  size_t num_nodes = 100;
  double p = 0.1;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon =
          epsilon_exponent < 0 ? 0.0 : 0.01 * std::pow(1.5, epsilon_exponent);
      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(*graph, config);
      RunSingleInsertionTest(*graph, config);
    }
  }
}

TEST_F(DynamicHacTest, TestMultipleApproximationFactorsWeightedClique) {
  size_t num_nodes = 100;
  double p = 1.0;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  for (int epsilon_exponent = -1; epsilon_exponent < 8; ++epsilon_exponent) {
    for (double weight_threshold : {0.0, 0.25, 0.5, 0.75}) {
      double epsilon =
          epsilon_exponent < 0 ? 0.0 : 0.01 * std::pow(1.5, epsilon_exponent);

      const auto config = MakeConfig(epsilon, weight_threshold);
      RunBatchInsertionAndDeletionTest(*graph, config);
      RunSingleInsertionTest(*graph, config);
    }
  }
}

}  // namespace
}  // namespace graph_mining::in_memory
