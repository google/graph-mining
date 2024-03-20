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

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_graph.h"

#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/math.h"

namespace graph_mining {
namespace in_memory {
namespace {

using NodeId = InMemoryClusterer::NodeId;
using Cluster = std::initializer_list<NodeId>;
using Dendrogram = graph_mining::in_memory::Dendrogram;
using DendrogramNode = graph_mining::in_memory::DendrogramNode;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

MATCHER_P2(IsNode, parent_id, merge_similarity, "") {
  return parent_id == arg.parent_id &&
         AlmostEquals(merge_similarity, arg.merge_similarity);
}
const auto IsEmptyNode =
    IsNode(Dendrogram::kNoParentId, std::numeric_limits<double>::infinity());

TEST(ApproximateSubgraphHacGraphTest, IsolatedGraph) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  NodeId num_nodes = 3;
  graph->SetNumNodes(num_nodes);
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, 1);

  std::vector<bool> is_active({true, true, false});

  std::vector<double> min_merge_similarities(
      num_nodes, std::numeric_limits<double>::max());

  ApproximateSubgraphHacGraph apx_graph(*graph, num_nodes, 0.1, 0.1,
                                        std::move(is_active),
                                        min_merge_similarities);
  EXPECT_EQ(apx_graph.NumNodes(), num_nodes);
  EXPECT_THAT(apx_graph.GetGoodEdge(),
              testing::FieldsAre(std::numeric_limits<NodeId>::max(),
                                 std::numeric_limits<NodeId>::max(),
                                 apx_graph.kDefaultGoodness));
  EXPECT_TRUE(apx_graph.IsActive(0));
  EXPECT_TRUE(apx_graph.IsActive(1));
  EXPECT_FALSE(apx_graph.IsActive(2));
  EXPECT_EQ(apx_graph.CurrentClusterSize(0), 1);
  EXPECT_EQ(apx_graph.CurrentClusterSize(1), 1);
  EXPECT_EQ(apx_graph.CurrentClusterSize(2), 1);
  for (NodeId i = 0; i < num_nodes; ++i) {
    EXPECT_THAT(apx_graph.Neighbors(i), IsEmpty());
  }
}

TEST(ApproximateSubgraphHacGraphTest, TestTriangleWithLowWeightSatellite) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  NodeId num_nodes = 3;
  graph->SetNumNodes(num_nodes);
  ASSERT_OK(graph->AddEdge(0, 1, 1.0));
  ASSERT_OK(graph->AddEdge(0, 2, 0.5));
  ASSERT_OK(graph->AddEdge(1, 2, 0.25));
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, 1);

  std::vector<bool> is_active({true, true, false});

  std::vector<double> min_merge_similarities(
      num_nodes, std::numeric_limits<double>::max());

  Dendrogram dendrogram(3);
  // Map from an active node id to its current cluster id. Initially just the
  // identity mapping.
  std::vector<NodeId> to_cluster_id(num_nodes);
  absl::c_iota(to_cluster_id, 0);

  ApproximateSubgraphHacGraph apx_graph(*graph, num_nodes, 0.1, 0.1,
                                        std::move(is_active),
                                        min_merge_similarities);
  EXPECT_EQ(apx_graph.NumNodes(), num_nodes);

  EXPECT_EQ(apx_graph.EdgeWeight(0, 1), 1.0);
  EXPECT_EQ(apx_graph.EdgeWeight(1, 0), 1.0);
  EXPECT_EQ(apx_graph.EdgeWeight(0, 2), 0.5);
  EXPECT_EQ(apx_graph.EdgeWeight(1, 2), 0.25);
  EXPECT_EQ(apx_graph.CurrentClusterSize(0), 1);
  EXPECT_EQ(apx_graph.CurrentClusterSize(1), 1);
  EXPECT_EQ(apx_graph.CurrentClusterSize(2), 1);
  EXPECT_EQ(apx_graph.EdgeWeightUnnormalized(0, 2), 0.5);
  EXPECT_EQ(apx_graph.EdgeWeightUnnormalized(0, 1), 1);
  EXPECT_EQ(apx_graph.EdgeWeightUnnormalized(1, 2), 0.25);
  EXPECT_EQ(apx_graph.EdgeWeightUnnormalized(1, 0), 1.0);
  EXPECT_THAT(apx_graph.Neighbors(0), UnorderedElementsAre(2, 1));
  EXPECT_THAT(apx_graph.Neighbors(1), UnorderedElementsAre(2, 0));
  EXPECT_THAT(apx_graph.Neighbors(2), IsEmpty());

  ASSERT_OK(apx_graph.Merge(&dendrogram, &to_cluster_id,
                            &min_merge_similarities, 0, 1));
  EXPECT_NE(apx_graph.IsActive(0), apx_graph.IsActive(1));
  EXPECT_NE(apx_graph.IsActive(0), apx_graph.IsActive(1));

  EXPECT_EQ(apx_graph.CurrentClusterSize(1), 2);
  EXPECT_EQ(apx_graph.CurrentClusterSize(2), 1);
  EXPECT_EQ(apx_graph.EdgeWeightUnnormalized(1, 2), 0.75);
  EXPECT_THAT(apx_graph.Neighbors(0), IsEmpty());
}

TEST(ApproximateSubgraphHacGraphTest, TestLine) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  NodeId num_nodes = 5;
  double epsilon = 0;
  graph->SetNumNodes(num_nodes);
  ASSERT_OK(graph->AddEdge(0, 1, 1.0));
  ASSERT_OK(graph->AddEdge(1, 2, 1.1));
  ASSERT_OK(graph->AddEdge(2, 3, 1.2));
  ASSERT_OK(graph->AddEdge(3, 4, 1.3));
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, 1);
  graph->SetNodeWeight(3, 1);
  graph->SetNodeWeight(4, 1);

  std::vector<bool> is_active(num_nodes, true);

  std::vector<double> min_merge_similarities(
      num_nodes, std::numeric_limits<double>::max());

  Dendrogram dendrogram(num_nodes);
  // Map from an active node id to its current cluster id. Initially just the
  // identity mapping.
  std::vector<NodeId> to_cluster_id(num_nodes);
  absl::c_iota(to_cluster_id, 0);

  ApproximateSubgraphHacGraph apx_graph(*graph, num_nodes, epsilon, epsilon,
                                        std::move(is_active),
                                        min_merge_similarities);

  EXPECT_EQ(apx_graph.NumNodes(), num_nodes);

  EXPECT_EQ(apx_graph.EdgeWeight(0, 1), 1.0);
  EXPECT_EQ(apx_graph.EdgeWeight(1, 2), 1.1);
  EXPECT_EQ(apx_graph.EdgeWeight(2, 3), 1.2);
  EXPECT_EQ(apx_graph.EdgeWeight(3, 4), 1.3);

  EXPECT_EQ(apx_graph.GetGoodEdge(), std::make_tuple(3, 4, 1.0));
  EXPECT_EQ(apx_graph.EdgeWeight(3, 4), 1.3);
  ASSERT_OK(apx_graph.Merge(&dendrogram, &to_cluster_id,
                            &min_merge_similarities, 3, 4));
  EXPECT_NE(apx_graph.IsActive(3), apx_graph.IsActive(4));
  EXPECT_EQ(apx_graph.IsActive(3), true);
  EXPECT_EQ(apx_graph.EdgeWeight(2, 3), 0.6);

  EXPECT_EQ(apx_graph.GetGoodEdge(), std::make_tuple(1, 2, 1.0));
  EXPECT_EQ(apx_graph.EdgeWeight(1, 2), 1.1);
  ASSERT_OK(apx_graph.Merge(&dendrogram, &to_cluster_id,
                            &min_merge_similarities, 1, 2));
  EXPECT_NE(apx_graph.IsActive(1), apx_graph.IsActive(2));
  EXPECT_EQ(apx_graph.IsActive(2), true);
  EXPECT_EQ(apx_graph.EdgeWeight(0, 2), 0.5);
  EXPECT_EQ(apx_graph.EdgeWeight(2, 3), 1.2 / 4);

  EXPECT_EQ(apx_graph.GetGoodEdge(), std::make_tuple(0, 2, 1.0));
  ASSERT_OK(apx_graph.Merge(&dendrogram, &to_cluster_id,
                            &min_merge_similarities, 0, 2));
  EXPECT_EQ(apx_graph.IsActive(2), true);

  EXPECT_EQ(apx_graph.GetGoodEdge(), std::make_tuple(2, 3, 1.0));
  EXPECT_EQ(apx_graph.EdgeWeight(2, 3), 1.2 / 6);
  ASSERT_OK(apx_graph.Merge(&dendrogram, &to_cluster_id,
                            &min_merge_similarities, 2, 3));
}

}  // namespace
}  // namespace in_memory
}  // namespace graph_mining
