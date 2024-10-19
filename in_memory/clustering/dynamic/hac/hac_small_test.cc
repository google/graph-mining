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

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/dynamic/hac/dynamic_hac.pb.h"
#include "in_memory/clustering/dynamic/hac/hac.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/generation/add_edge_weights.h"
#include "in_memory/generation/erdos_renyi.h"
#include "in_memory/parallel/scheduler.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {
namespace {

using ::testing::FieldsAre;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;



using ParentEdge = DynamicDendrogram::ParentEdge;
using Cluster =
    std::initializer_list<graph_mining::in_memory::InMemoryClusterer::NodeId>;
using Clustering = graph_mining::in_memory::Clustering;
using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using NodeId = graph_mining::in_memory::InMemoryClusterer::NodeId;
using NodeColor = DynamicHacNodeColor::NodeColor;

inline void TestDendrogram(const DynamicDendrogram& dendrogram,
                           const NodeId& id1, const NodeId& id2,
                           const NodeId& parent, float weight) {
  EXPECT_THAT(dendrogram.Parent(id1),
              IsOkAndHolds(FieldsAre(parent, FloatEq(weight))))
      << id1;
  EXPECT_THAT(dendrogram.Parent(id2),
              IsOkAndHolds(FieldsAre(parent, FloatEq(weight))))
      << id2;
}

DynamicHacConfig MakeConfig(double epsilon, double weight_threshold) {
  DynamicHacConfig result;
  result.set_epsilon(epsilon);
  result.set_weight_threshold(weight_threshold);
  return result;
}

TEST(DynamicHacTest, NoEdge) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0.01, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
}

TEST(DynamicHacTest, OneEdge) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  const auto clustering1 = Clustering({{0, 1}});
  const auto clustering2 = Clustering({{0}, {1}});
  EXPECT_THAT(clusterer->FlatCluster(0.5), IsOkAndHolds(clustering1));
  EXPECT_THAT(clusterer->FlatCluster(1), IsOkAndHolds(clustering2));

  auto dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.6);

  EXPECT_OK(clusterer->Remove({0}));
  dendrogram = clusterer->Dendrogram();

  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, OneEdgeNonconsecutiveIds) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {2, 1, {{3, 0.6}}};
  const AdjacencyList adj_list_1 = {3, 1, {{2, 0.6}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  const auto clustering1 = Clustering({{2, 3}});
  const auto clustering2 = Clustering({{2}, {3}});
  EXPECT_THAT(clusterer->FlatCluster(0.5), IsOkAndHolds(clustering1));
  EXPECT_THAT(clusterer->FlatCluster(1), IsOkAndHolds(clustering2));

  auto dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 2, 3, large_id + 1, 0.6);

  EXPECT_OK(clusterer->Remove({2}));
  dendrogram = clusterer->Dendrogram();

  EXPECT_THAT(dendrogram.HasValidParent(3), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, TwoEdges) {
  // This graph looks like 0 -- 1 -- 2.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));

  EXPECT_THAT(clusterer->FlatCluster(0.7),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({0}),
                                                UnorderedElementsAre(2, 1))));

  const auto& dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 1, 2, large_id + 1, 0.8);
  TestDendrogram(dendrogram, 0, large_id + 1, large_id + 2, 0.3);
}

TEST(DynamicHacTest, RemoveNoRecluster) {
  // This graph looks like 0 -- 1 -- 2.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};
  NodeId large_id = clusterer->LargestAvailableNodeId();

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));
  EXPECT_OK(clusterer->Remove({0}));
  auto dendrogram = clusterer->Dendrogram();
  TestDendrogram(dendrogram, 1, 2, large_id + 1, 0.8);
  EXPECT_THAT(dendrogram.Parent(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, RemoveSingletonClusters) {
  // This graph looks like 0 -- 1 -- 2.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));
  EXPECT_OK(clusterer->Remove({1}));
  auto dendrogram = clusterer->Dendrogram();
  EXPECT_THAT(dendrogram.HasValidParent(0), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, RemoveRecluster) {
  // This graph looks like 0 -- 1 -- 2.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};
  NodeId large_id = clusterer->LargestAvailableNodeId();

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));
  EXPECT_OK(clusterer->Remove({2}));
  auto dendrogram = clusterer->Dendrogram();
  TestDendrogram(dendrogram, 0, 1, large_id + 3, 0.6);
  EXPECT_THAT(dendrogram.Parent(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, Triangle) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}, {2, 1}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}, {0, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));

  EXPECT_THAT(clusterer->FlatCluster(0.75),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({1}),
                                                UnorderedElementsAre(2, 0))));

  const auto& dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 0, 2, large_id + 1, 1);
  TestDendrogram(dendrogram, 1, large_id + 1, large_id + 2, 0.7);

  EXPECT_OK(clusterer->Remove({2}));
  const auto& dendrogram2 = clusterer->Dendrogram();
  TestDendrogram(dendrogram2, 0, 1, large_id + 3, 0.6);
  EXPECT_THAT(dendrogram2.Parent(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_OK(clusterer->Remove({0}));
  const auto& dendrogram3 = clusterer->Dendrogram();
  EXPECT_THAT(dendrogram3.HasValidParent(1), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram3.Parent(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, RemoveMultipleNodes) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}, {2, 1}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}, {0, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));
  EXPECT_OK(clusterer->Remove({0, 1}));
  const auto& dendrogram = clusterer->Dendrogram();
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Parent(1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(DynamicHacTest, RemoveAllNodes) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}, {2, 1}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}, {0, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));
  EXPECT_OK(clusterer->Remove({0, 1, 2}));
  const auto& dendrogram = clusterer->Dendrogram();
  EXPECT_THAT(dendrogram.NumNodes(), 0);
}

TEST(DynamicHacTest, TriangleLargeNodeId) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {10, 1, {{11, 0.6}, {13, 1}}};
  const AdjacencyList adj_list_1 = {11, 1, {{10, 0.6}, {13, 0.8}}};
  const AdjacencyList adj_list_2 = {13, 1, {{11, 0.8}, {10, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));

  EXPECT_THAT(clusterer->FlatCluster(0.75),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({11}),
                                                UnorderedElementsAre(10, 13))));

  const auto& dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 10, 13, large_id + 1, 1);
  TestDendrogram(dendrogram, 11, large_id + 1, large_id + 2, 0.7);
}

TEST(DynamicHacTest, ClustersIsolatedGraph) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {0, 1, {}};
  const AdjacencyList adj_list_1 = {1, 1, {}};
  const AdjacencyList adj_list_2 = {2, 1, {}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));

  EXPECT_THAT(clusterer->FlatCluster(0.75),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({0}),
                                                std::vector<NodeId>({1}),
                                                std::vector<NodeId>({2}))));

  const auto& dendrogram = clusterer->Dendrogram();

  EXPECT_THAT(dendrogram.HasValidParent(0), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
}

TEST(DynamicHacTest, TriangleInsertTwice) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}, {0, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));
  ASSERT_OK(clusterer->Insert({adj_list_2}));

  const auto& dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();

  TestDendrogram(dendrogram, 0, 2, large_id + 2, 1);
  TestDendrogram(dendrogram, 1, large_id + 2, large_id + 3, 0.7);
}

TEST(DynamicHacTest, RemoveInsert) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}, {0, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));
  for (int i = 0; i < 100; ++i) {
    ASSERT_OK(clusterer->Insert({adj_list_2}));
    ASSERT_OK(clusterer->Remove({2}));
  }
  ASSERT_OK(clusterer->Insert({adj_list_2}));

  ASSERT_OK_AND_ASSIGN(const auto& p0, clusterer->Dendrogram().Parent(0));
  const auto& [parent0, w0] = p0;
  ASSERT_OK_AND_ASSIGN(const auto& p2, clusterer->Dendrogram().Parent(2));
  const auto& [parent2, w2] = p2;
  ASSERT_OK_AND_ASSIGN(const auto& p1, clusterer->Dendrogram().Parent(1));
  const auto& [parent1, w1] = p1;
  EXPECT_EQ(parent0, parent2);
  EXPECT_EQ(w0, 1);
  EXPECT_EQ(w2, 1);
  EXPECT_THAT(w1, FloatEq(0.7));
}

TEST(DynamicHacTest, HasEdgeToCleanPartition) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  auto dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();
  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.6);

  ASSERT_OK(clusterer->Insert({adj_list_2}));

  dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 1, 2, large_id + 2, 0.8);
  TestDendrogram(dendrogram, 0, large_id + 2, large_id + 3, 0.3);
}

TEST(DynamicHacTest, InvalidMergeInFutureRound) {
  // 2 - 3 merge in round 1.
  // The insertion of node 4 will make the 2 - 3 merge invalid in round 0
  // because node 4 is in partition 0, but merge 2-3 is actually valid.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  NodeId large_id = clusterer->LargestAvailableNodeId();
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{2, NodeColor::kRed, 1},
                               {3, NodeColor::kRed, 1},
                               {4, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{2, NodeColor::kBlue, 1},
                               {3, NodeColor::kRed, 1},
                               {4, NodeColor::kRed, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kBlue, 1}, {4, NodeColor::kRed, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 2, NodeColor::kBlue, 1}})));
  clusterer->SetColors(colors);
  const AdjacencyList adj_list_2 = {2, 1, {{3, 0.6}}};
  const AdjacencyList adj_list_3 = {3, 1, {{2, 0.6}}};
  const AdjacencyList adj_list_4 = {4, 1, {{2, 0.001}}};

  ASSERT_OK(clusterer->Insert({adj_list_2, adj_list_3}));

  ASSERT_OK(clusterer->Insert({adj_list_4}));
}

TEST(DynamicHacTest, EdgeWightOptimized) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.8}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.6}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  auto dendrogram = clusterer->Dendrogram();
  NodeId large_id = clusterer->LargestAvailableNodeId();
  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);

  ASSERT_OK(clusterer->Insert({adj_list_2}));

  dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);
  TestDendrogram(dendrogram, 2, large_id + 1, large_id + 2, 0.3);
}

TEST(DynamicHacTest, InsertToNewRoundBlueLastRound) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  NodeId large_id = clusterer->LargestAvailableNodeId();
  // The colors are set so we need one additional round when inserting the new
  // edge. The root node after the first insertion is blue in the last round.
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kRed, 1},
                               {1, NodeColor::kBlue, 1},
                               {2, NodeColor::kBlue, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kBlue, 1}, {2, NodeColor::kBlue, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kRed, 1}, {2, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 2, NodeColor::kRed, 1}})));
  clusterer->SetColors(colors);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.8}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.6}}};

  ASSERT_OK_AND_ASSIGN(auto update_stats,
                       clusterer->Insert({adj_list_0, adj_list_1}));

  auto dendrogram = clusterer->Dendrogram();
  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);
  EXPECT_THAT(update_stats, FieldsAre(1, 2, 2, 0));

  ASSERT_OK_AND_ASSIGN(update_stats, clusterer->Insert({adj_list_2}));

  dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);
  TestDendrogram(dendrogram, 2, large_id + 1, large_id + 2, 0.3);

  EXPECT_THAT(update_stats, FieldsAre(5, 11, 12, 0));
}

TEST(DynamicHacTest, DeleteLastMerge) {
  // Deleting node 3 does not change lower dendrogram.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);
  NodeId large_id = clusterer->LargestAvailableNodeId();
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kRed, 1},
                               {1, NodeColor::kBlue, 1},
                               {2, NodeColor::kBlue, 1},
                               {3, NodeColor::kRed, 1},
                               {4, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 1, NodeColor::kRed, 1},
                               {2, NodeColor::kBlue, 1},
                               {3, NodeColor::kRed, 1},
                               {large_id + 2, NodeColor::kBlue, 1},
                               {4, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 2, NodeColor::kRed, 1},
                               {3, NodeColor::kBlue, 1},
                               {4, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 3, NodeColor::kRed, 1},
                               {large_id + 4, NodeColor::kRed, 1}})));
  clusterer->SetColors(colors);

  const std::vector<std::tuple<NodeId, NodeId, double>> edges{
      {0, 1, 1}, {0, 2, 0.1}, {1, 2, 0.5}, {2, 4, 0.24}, {2, 3, 0.2}};
  const absl::flat_hash_map<NodeId, NodeId> node_sizes{
      {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};

  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(edges, node_sizes));

  ASSERT_OK(clusterer->Insert(adj_lists));

  // Delete node 3.
  ASSERT_OK(clusterer->Remove({3}));

  auto dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 0, 1, large_id + 1, 1);
  TestDendrogram(dendrogram, 2, large_id + 1, large_id + 2, 0.3);
  TestDendrogram(dendrogram, 4, large_id + 2, large_id + 3, 0.08);
}

TEST(DynamicHacTest, InsertToNewRoundRedLastRound) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  NodeId large_id = clusterer->LargestAvailableNodeId();
  // The colors are set so we need one additional round when inserting the new
  // edge. The root node after the first insertion is red in the last round.
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kRed, 1},
                               {1, NodeColor::kBlue, 1},
                               {2, NodeColor::kBlue, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kRed, 1}, {2, NodeColor::kRed, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kRed, 1}, {2, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 2, NodeColor::kRed, 1}})));
  clusterer->SetColors(colors);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.8}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.6}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  auto dendrogram = clusterer->Dendrogram();
  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);

  ASSERT_OK(clusterer->Insert({adj_list_2}));

  dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 0, 1, large_id + 1, 0.8);
  TestDendrogram(dendrogram, 2, large_id + 1, large_id + 2, 0.3);
}

TEST(DynamicHacTest, TestWeightedErdosRenyiInsertion) {
  std::size_t num_nodes = 10;
  double p = 0.8;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(*graph));

  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  ASSERT_OK(clusterer->Insert(adj_lists));
  ASSERT_OK(clusterer->Insert({{11, 1, {{0, 1.1}}}}));
  ASSERT_THAT(clusterer->Dendrogram().Sibling(11), Optional(0));
  ASSERT_OK_AND_ASSIGN(const auto parent, clusterer->Dendrogram().Parent(0));
  ASSERT_OK(clusterer->Insert({{12, 1, {{0, 0.001}}}}));
  // The same parent is preserved.
  EXPECT_THAT(
      clusterer->Dendrogram().Parent(0),
      IsOkAndHolds(FieldsAre(parent.parent_id, parent.merge_similarity)));
}

TEST(DynamicHacTest, TestWeightedErdosRenyiDeletion) {
  std::size_t num_nodes = 10;
  double p = 0.8;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(*graph));

  auto config = MakeConfig(0, 0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  ASSERT_OK(clusterer->Insert(adj_lists));
  for (int i = 0; i < num_nodes; ++i) {
    ASSERT_OK(clusterer->Remove({i}));
  }
}

TEST(DynamicHacTest, ThresholdNoMerge) {
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, /*weight_threshold=*/1);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1}));

  const auto& dendrogram = clusterer->Dendrogram();

  // Threshold is higher than all edge weights, so no merge happens.
  EXPECT_THAT(dendrogram.HasValidParent(0), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(false));
}

TEST(DynamicHacTest, ThresholdTwoEdges) {
  // This graph looks like 0 -- 1 -- 2. Only one merge happens because of
  // threshold.
  graph_mining::in_memory::SimpleUndirectedGraph graph;
  auto config = MakeConfig(0, /*weight_threshold=*/0.7);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  NodeId large_id = clusterer->LargestAvailableNodeId();
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kBlue, 1},
                               {1, NodeColor::kRed, 1},
                               {2, NodeColor::kRed, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kRed, 1},
                               {1, NodeColor::kRed, 1},
                               {2, NodeColor::kBlue, 1}})));
  colors.push_back(
      std::make_unique<DynamicHacNodeColorTest>(DynamicHacNodeColorTest(
          {{large_id + 1, NodeColor::kBlue, 1}, {0, NodeColor::kRed, 1}})));
  clusterer->SetColors(colors);

  const AdjacencyList adj_list_0 = {0, 1, {{1, 0.6}}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.6}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{1, 0.8}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_1, adj_list_2}));

  EXPECT_THAT(clusterer->FlatCluster(1),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({0}),
                                                std::vector<NodeId>({1}),
                                                std::vector<NodeId>({2}))));

  EXPECT_THAT(clusterer->FlatCluster(0.75),
              IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({0}),
                                                UnorderedElementsAre(2, 1))));

  const auto& dendrogram = clusterer->Dendrogram();

  TestDendrogram(dendrogram, 1, 2, large_id + 1, 0.8);
  EXPECT_THAT(dendrogram.HasValidParent(0), IsOkAndHolds(false));
}

TEST(DynamicHacTest, SingletonBluePartition) {
  auto config = MakeConfig(0, /*weight_threshold=*/0);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  NodeId large_id = clusterer->LargestAvailableNodeId();
  std::vector<std::unique_ptr<DynamicHacNodeColorTest>> colors;
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kBlue, 1},
                               {1, NodeColor::kBlue, 1},
                               {2, NodeColor::kRed, 1},
                               {3, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{0, NodeColor::kRed, 1},
                               {large_id + 1, NodeColor::kBlue, 1},
                               {large_id + 2, NodeColor::kBlue, 1}})));
  colors.push_back(std::make_unique<DynamicHacNodeColorTest>(
      DynamicHacNodeColorTest({{large_id + 3, NodeColor::kBlue, 1}})));
  clusterer->SetColors(colors);

  // This graph looks like 0 -- 1 -- 2 -- 3.
  const AdjacencyList adj_list_0 = {0, 1, {}};
  const AdjacencyList adj_list_1 = {1, 1, {{0, 0.3}, {2, 0.8}}};
  const AdjacencyList adj_list_2 = {2, 1, {{3, 1}}};
  const AdjacencyList adj_list_3 = {3, 1, {{2, 1}}};

  ASSERT_OK(clusterer->Insert({adj_list_0, adj_list_2, adj_list_3}));
  ASSERT_OK(clusterer->Insert({adj_list_1}));

  EXPECT_THAT(
      clusterer->FlatCluster(0),
      IsOkAndHolds(UnorderedElementsAre(std::vector<NodeId>({0, 1, 2, 3}))));
}

TEST(DynamicHacTest, TestWeightedErdosRenyiThresholdInsertion) {
  std::size_t num_nodes = 10;
  double p = 0.8;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(*graph));

  auto config = MakeConfig(0, /*weight_threshold=*/0.1);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  ASSERT_OK(clusterer->Insert(adj_lists));
  ASSERT_OK(clusterer->Insert({{11, 1, {{0, 1.1}}}}));
  ASSERT_THAT(clusterer->Dendrogram().Sibling(11), Optional(0));
  ASSERT_OK_AND_ASSIGN(const auto parent, clusterer->Dendrogram().Parent(0));
  ASSERT_OK(clusterer->Insert({{12, 1, {{0, 0.001}}}}));
  // The same parent is preserved.
  EXPECT_THAT(
      clusterer->Dendrogram().Parent(0),
      IsOkAndHolds(FieldsAre(parent.parent_id, parent.merge_similarity)));
}

TEST(DynamicHacTest, TestWeightedErdosRenyiThresholdInsertionDeletion) {
  std::size_t num_nodes = 10;
  double p = 0.8;
  ASSERT_OK_AND_ASSIGN(
      auto graph, graph_mining::in_memory::UnweightedErdosRenyi(num_nodes, p));
  ASSERT_OK(graph_mining::in_memory::AddUniformWeights(*graph, 0.01, 1));

  ASSERT_OK_AND_ASSIGN(auto adj_lists, ConvertToAdjList(*graph));

  auto config = MakeConfig(0, 0.1);
  auto clusterer = std::make_unique<DynamicHacClusterer>(config);

  ASSERT_OK(clusterer->Insert(adj_lists));
  for (int i = 0; i < num_nodes; ++i) {
    ASSERT_OK(clusterer->Remove({i}));
  }
}

}  // namespace
}  // namespace graph_mining::in_memory
