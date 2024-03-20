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

#include "in_memory/clustering/dynamic/hac/hac_internal.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"
#include "in_memory/clustering/dynamic/hac/color_utils.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using graph_mining::in_memory::NodeId;
using NodeColor = DynamicHacNodeColorBase::NodeColor;
constexpr auto kRed = NodeColor::kRed;
constexpr auto kBlue = NodeColor::kBlue;
using M = std::tuple<NodeId, NodeId, double>;

constexpr int32_t kDefaultNumberOfWorkers = 6;

using ParallelTest = ::testing::Test;

class RunSubgraphHacTest : public ParallelTest {};

TEST_F(RunSubgraphHacTest, OneNode) {
  double epsilon = 1;
  auto partition_graph =
      std::make_unique<graph_mining::in_memory::SimpleUndirectedGraph>();
  std::vector<NodeId> node_map;
  absl::flat_hash_map<NodeId, double> min_merge_similarities;
  partition_graph->SetNumNodes(1);
  EXPECT_OK(partition_graph->FinishImport());
  node_map.push_back(10);
  min_merge_similarities[10] = 2;

  auto min_merge_similarities_partition_map =
      SubgraphMinMergeSimilarity(node_map, min_merge_similarities);

  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      RunSubgraphHac(partition_graph, min_merge_similarities_partition_map,
                     epsilon));
  const auto& [merges, clustering, dendrogram, contracted_graph] =
      subgraph_hac_result;
  EXPECT_THAT(merges, IsEmpty());
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 1);
}

TEST_F(RunSubgraphHacTest, TwoNodesMerge) {
  // A graph with edge 10 -- 11. The two nodes are both active.

  double epsilon = 1;
  auto partition_graph =
      std::make_unique<graph_mining::in_memory::SimpleUndirectedGraph>();
  std::vector<NodeId> node_map;
  absl::flat_hash_map<NodeId, double> min_merge_similarities;

  EXPECT_OK(partition_graph->AddEdge(0, 1, 0.5));
  EXPECT_OK(partition_graph->FinishImport());
  node_map.push_back(10);
  node_map.push_back(11);
  min_merge_similarities[10] = 2;
  min_merge_similarities[11] = 1;

  auto min_merge_similarities_partition_map =
      SubgraphMinMergeSimilarity(node_map, min_merge_similarities);

  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      RunSubgraphHac(partition_graph, min_merge_similarities_partition_map,
                     epsilon));
  const auto& [merges, clustering, dendrogram, contracted_graph] =
      subgraph_hac_result;

  std::vector<std::tuple<NodeId, NodeId, double>> expected_merges = {
      std::make_tuple(0, 1, 0.5)};
  EXPECT_THAT(merges, ElementsAre(Eq(expected_merges[0])));
  EXPECT_THAT(dendrogram.GetParent(0), FieldsAre(2, 0.5));
  EXPECT_THAT(dendrogram.GetParent(1), FieldsAre(2, 0.5));
}

TEST_F(RunSubgraphHacTest, TwoActiveNodesNoMerge) {
  // A graph with edges 10 -- 11 -- 12. The two nodes 10, 11 are both active,
  // but they will not merge because of epsilon.
  const double kInf = std::numeric_limits<double>::infinity();
  double epsilon = 0;  // The true epsilon is 1 + epsilon in SubgraphHac.
  auto partition_graph =
      std::make_unique<graph_mining::in_memory::SimpleUndirectedGraph>();
  std::vector<NodeId> node_map;
  absl::flat_hash_map<NodeId, double> min_merge_similarities;

  EXPECT_OK(partition_graph->AddEdge(0, 1, 0.5));
  EXPECT_OK(partition_graph->AddEdge(1, 2, 0.8));
  partition_graph->SetNodeWeight(2, -1);

  EXPECT_OK(partition_graph->FinishImport());
  node_map.push_back(10);
  node_map.push_back(11);
  node_map.push_back(12);
  min_merge_similarities[10] = kInf;
  min_merge_similarities[11] = kInf;
  min_merge_similarities[12] = kInf;

  auto min_merge_similarities_partition_map =
      SubgraphMinMergeSimilarity(node_map, min_merge_similarities);

  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      RunSubgraphHac(partition_graph, min_merge_similarities_partition_map,
                     epsilon));
  const auto& [merges, clustering, dendrogram, contracted_graph] =
      subgraph_hac_result;
  EXPECT_THAT(merges, IsEmpty());
  EXPECT_EQ(dendrogram.HasValidParent(0), false);
  EXPECT_EQ(dendrogram.HasValidParent(1), false);
}

TEST(UpdatePartitionTest, UsingGraphWithNoEdges) {
  // Add a single node 0 with weight 1. It does not have any edge.
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0};
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  const absl::flat_hash_set<NodeId> neighbors_deleted;
  const DynamicHacNodeColorTest color({{0, kBlue, 8}});
  absl::flat_hash_map<NodeId, NodeId> partition_map;

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(adj_list_vec, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_THAT(partition_map, UnorderedElementsAre(FieldsAre(0, 0)));
  EXPECT_THAT(changes, UnorderedElementsAre(FieldsAre(0, -1, 0)));
}

TEST(UpdatePartitionTest, AddOneBlueNode) {
  // Add blue node 0 to a graph with a single red node 2.
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const DynamicHacNodeColorTest color({{0, kBlue, 5}, {2, kRed, 8}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{2, 2}};

  const std::vector<AdjacencyList> insert_list = {adj_list_0};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(0, -1, 2), FieldsAre(2, 2, 2)));
  EXPECT_THAT(partition_map,
              UnorderedElementsAre(FieldsAre(0, 2), FieldsAre(2, 2)));
}

TEST(UpdatePartitionTest, AddOneRedNode) {
  // Add red node 2 to a graph with a single blue node 0.
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const DynamicHacNodeColorTest color({{0, kBlue, 5}, {2, kRed, 8}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 0}};

  const std::vector<AdjacencyList> insert_list = {adj_list_2};

  // Test Highest Similarity
  partition_map.clear();
  partition_map[0] = 0;

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(2, -1, 2), FieldsAre(0, 0, 2)));
  EXPECT_THAT(partition_map,
              UnorderedElementsAre(FieldsAre(2, 2), FieldsAre(0, 2)));
}

TEST(UpdatePartitionTest, AddMultipleNodesAssignToNew) {
  // Add node 3 and node 4 to a graph with an edge (0, 2).
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  const AdjacencyList adj_list_3 = {/*id=*/3, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}, {4, 0.3}}};
  const AdjacencyList adj_list_4 = {/*id=*/4, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 0.4}, {3, 3}}};
  const std::vector<AdjacencyList> insert_list = {adj_list_3, adj_list_4};
  EXPECT_OK(graph.AddNodes(insert_list));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const DynamicHacNodeColorTest color(
      {{0, kBlue, 5}, {2, kRed, 8}, {3, kRed, 6}, {4, kBlue, 7}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 2}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 2);
  EXPECT_EQ(partition_map[2], 2);
  EXPECT_EQ(partition_map[3], 3);  // Node 3 is red and is in its own partition.
  EXPECT_EQ(partition_map[4], 3);
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(2, 2, 2), FieldsAre(3, -1, 3),
                                   FieldsAre(4, -1, 3)));
}

TEST(UpdatePartitionTest, AddMultipleNodesAssignToOld) {
  // Add node 3 and node 4 to a graph with an edge (0, 2).
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/1,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const AdjacencyList adj_list_3 = {/*id=*/3, /*weight=*/2,
                                    /*outgoing_edges=*/{{2, 2}, {4, 0.4}}};
  const AdjacencyList adj_list_4 = {/*id=*/4, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 0.3}, {3, 0.4}}};

  const std::vector<AdjacencyList> insert_list = {adj_list_3, adj_list_4};
  EXPECT_OK(graph.AddNodes(insert_list));

  const DynamicHacNodeColorTest color(
      {{0, kBlue, 5}, {2, kRed, 8}, {3, kRed, 9}, {4, kBlue, 7}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 2}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 2);
  EXPECT_EQ(partition_map[2], 2);
  EXPECT_EQ(partition_map[3], 3);  // Node 3 is red and is in its own partition.
  EXPECT_EQ(partition_map[4], 2);
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(2, 2, 2), FieldsAre(3, -1, 3),
                                   FieldsAre(4, -1, 2)));
}

TEST(UpdatePartitionTest, AddMultipleNodesNoAssign2) {
  // Add node 3 and node 4 to a graph with an edge (0, 2).
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const AdjacencyList adj_list_3 = {/*id=*/3, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_4 = {/*id=*/4, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};
  const std::vector<AdjacencyList> insert_list = {adj_list_3, adj_list_4};
  EXPECT_OK(graph.AddNodes(insert_list));

  const DynamicHacNodeColorTest color(
      {{0, kBlue, 5}, {2, kRed, 8}, {3, kRed, 9}, {4, kBlue, 7}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 2}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 2);
  EXPECT_EQ(partition_map[2], 2);
  EXPECT_EQ(partition_map[3], 3);  // Node 3 is red and is in its own partition.
  EXPECT_EQ(partition_map[4], 4);
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(0, 2, 2), FieldsAre(2, 2, 2),
                                   FieldsAre(3, -1, 3), FieldsAre(4, -1, 4)));
}

TEST(UpdatePartitionTest, PartitionReassignBlue) {
  // Add node 3 to a graph with an edge (0, 2). Node 0's partition will update
  // from 2 to 3. We should have Partition 2 in the dirty partition.
  DynamicClusteredGraph graph;
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/2,
                                    /*outgoing_edges=*/{{2, 4}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  const std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  const AdjacencyList adj_list_3 = {/*id=*/3, /*weight=*/1,
                                    /*outgoing_edges=*/{{0, 3}}};

  std::vector<AdjacencyList> insert_list = {adj_list_3};
  EXPECT_OK(graph.AddNodes(insert_list));

  const DynamicHacNodeColorTest color(
      {{0, kBlue, 5}, {2, kRed, 8}, {3, kRed, 7}});
  absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 2}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 3);
  EXPECT_EQ(partition_map[2], 2);
  EXPECT_EQ(partition_map[3], 3);  // Node 3 is red and is in its own partition.
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(0, 2, 3), FieldsAre(3, -1, 3)));
}

TEST(UpdatePartitionTest, DeleteNodes) {
  // Graph 4-5-6-0-1-2-3. Node 6 is added. Nodes 2 and 4 are deleted.
  DynamicClusteredGraph graph;

  ASSERT_OK_AND_ASSIGN(
      const auto adj_list_vec,
      ConvertToAdjList({{4, 5}, {5, 6}, {6, 0}, {0, 1}, {1, 2}, {2, 3}}));
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const absl::flat_hash_set<NodeId> deleted_nodes({2, 4});

  const AdjacencyList adj_list_6 = {/*id=*/6, /*weight=*/1,
                                    /*outgoing_edges=*/{{0, 2}, {5, 2}}};

  std::vector<AdjacencyList> insert_list = {adj_list_6};

  const DynamicHacNodeColorTest color({{0, kBlue, 5},
                                       {1, kRed, 10},
                                       {2, kBlue, 8},
                                       {3, kRed, 1},
                                       {4, kRed, 6},
                                       {5, kBlue, 9},
                                       {6, kRed, 9}});

  ASSERT_OK_AND_ASSIGN(auto neighbors_deleted, graph.Neighbors(deleted_nodes));
  absl::erase_if(neighbors_deleted,
                 [&](auto k) { return deleted_nodes.contains(k); });

  for (const auto v : deleted_nodes) {
    EXPECT_OK(graph.RemoveNode(v));
  }

  absl::flat_hash_map<NodeId, NodeId> partition_map = {{0, 1}, {1, 1}, {2, 3},
                                                       {3, 3}, {4, 4}, {5, 4}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(insert_list, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 6);  // Node 0 changed to new node 6.
  EXPECT_EQ(partition_map[1], 1);
  EXPECT_EQ(partition_map[3], 3);
  EXPECT_EQ(partition_map[5], 6);  // Node 5 changed to new node 6.
  EXPECT_EQ(partition_map[6], 6);

  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(0, 1, 6), FieldsAre(5, 4, 6),
                                   FieldsAre(1, 1, 1), FieldsAre(3, 3, 3),
                                   FieldsAre(6, -1, 6)));
}

TEST(UpdatePartitionTest, Empty) {
  // Add a single node 0 with weight 1. It does not have any edge.
  DynamicClusteredGraph graph;
  const std::vector<AdjacencyList> adj_list_vec;
  const absl::flat_hash_set<NodeId> neighbors_deleted;
  absl::flat_hash_map<NodeId, NodeId> partition_map;
  const DynamicHacNodeColorTest color({});

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(adj_list_vec, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_THAT(partition_map, IsEmpty());
  EXPECT_THAT(changes, IsEmpty());
}

TEST(UpdatePartitionTest, InsertSingleBlue) {
  // Insert node 3 with three neighbors (0 1 2) to empty graph.
  const std::vector<AdjacencyList> new_nodes = {
      {/*id=*/3, /*weight=*/1,
       /*outgoing_edges=*/{{0, 0.5}, {1, 0.6}, {2, 0.7}}}};
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  DynamicClusteredGraph graph;
  absl::flat_hash_map<NodeId, NodeId> node_sizes = {
      {0, 1}, {1, 1}, {2, 1}, {3, 1}};
  ASSERT_OK_AND_ASSIGN(
      const auto adj_list_vec,
      ConvertToAdjList({{0, 3, 0.5}, {1, 3, 0.6}, {2, 3, 0.7}}, node_sizes));
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  // Blue node should point to the red neighbor with largest similarity and skip
  // the blue neighbors.
  const DynamicHacNodeColorTest color(
      {{0, kRed, 0}, {1, kRed, 0}, {2, kBlue, 0}, {3, kBlue, 0}});
  absl::flat_hash_map<NodeId, NodeId> partition_map = {{0, 0}, {1, 1}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(new_nodes, neighbors_deleted, graph,
                                        color, partition_map));

  EXPECT_EQ(partition_map[3], 1);
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(3, -1, 1), FieldsAre(0, 0, 0),
                                   FieldsAre(1, 1, 1), FieldsAre(2, 2, 2)));

  // All neighbors are also blue, the blue node should point to itself.
  const DynamicHacNodeColorTest color2(
      {{0, kBlue, 0}, {1, kBlue, 0}, {2, kBlue, 0}, {3, kBlue, 0}});
  partition_map = {{0, 0}, {1, 1}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes2,
                       UpdatePartitions(new_nodes, neighbors_deleted, graph,
                                        color2, partition_map));

  EXPECT_EQ(partition_map[3], 3);
  EXPECT_THAT(changes2,
              UnorderedElementsAre(FieldsAre(3, -1, 3), FieldsAre(0, 0, 0),
                                   FieldsAre(1, 1, 1), FieldsAre(2, 2, 2)));
}

TEST(UpdatePartitionTest, InsertSingleBlueClusterSize) {
  // Insert node 3 with three neighbors (0 1 2) to empty graph. It has weight 2.
  const std::vector<AdjacencyList> new_nodes = {
      {/*id=*/3, /*weight=*/2,
       /*outgoing_edges=*/{{0, 0.5}, {1, 0.6}, {2, 0.7}}}};
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  DynamicClusteredGraph graph;
  absl::flat_hash_map<NodeId, NodeId> node_sizes = {
      {0, 1}, {1, 2}, {2, 1}, {3, 2}};
  ASSERT_OK_AND_ASSIGN(
      const auto adj_list_vec,
      ConvertToAdjList({{0, 3, 0.5}, {1, 3, 0.6}, {2, 3, 0.7}}, node_sizes));
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  // Blue node should point to the red neighbor with largest similarity and skip
  // the blue neighbors.
  const DynamicHacNodeColorTest color(
      {{0, kRed, 0}, {1, kRed, 0}, {2, kBlue, 0}, {3, kBlue, 0}});
  absl::flat_hash_map<NodeId, NodeId> partition_map = {{0, 0}, {1, 1}, {2, 2}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(new_nodes, neighbors_deleted, graph,
                                        color, partition_map));

  EXPECT_EQ(partition_map[3], 0);
  EXPECT_THAT(changes,
              UnorderedElementsAre(FieldsAre(3, -1, 0), FieldsAre(0, 0, 0),
                                   FieldsAre(1, 1, 1), FieldsAre(2, 2, 2)));
}

TEST(UpdatePartitionTest, InsertSingleRed) {
  const std::vector<AdjacencyList> new_nodes = {
      {/*id=*/3, /*weight=*/1,
       /*outgoing_edges=*/{{0, 0.5}, {1, 0.6}, {2, 0.7}}}};
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  DynamicClusteredGraph graph;
  absl::flat_hash_map<NodeId, NodeId> node_sizes = {{0, 1}, {1, 1}, {2, 1},
                                                    {3, 1}, {4, 1}, {5, 1}};
  ASSERT_OK_AND_ASSIGN(
      const auto adj_list_vec,
      ConvertToAdjList(
          {{0, 3, 0.5}, {1, 3, 0.6}, {2, 3, 0.7}, {0, 4, 0.4}, {5, 4, 0.6}},
          node_sizes));
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  // Node 1 is red, so it does not change partition
  // Node 0 changes from 4 to 3
  // Node 2 changes from 2 to 3
  // Node 5 is blue and also does not change
  const DynamicHacNodeColorTest color({{0, kBlue, 0},
                                       {1, kRed, 0},
                                       {2, kBlue, 0},
                                       {3, kRed, 0},
                                       {4, kRed, 0},
                                       {5, kBlue, 0}});
  absl::flat_hash_map<NodeId, NodeId> partition_map = {
      {1, 1}, {0, 4}, {2, 3}, {4, 4}, {5, 4}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(new_nodes, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 3);
  EXPECT_EQ(partition_map[1], 1);
  EXPECT_EQ(partition_map[2], 3);
  EXPECT_EQ(partition_map[3], 3);
  EXPECT_EQ(partition_map[4], 4);
  EXPECT_EQ(partition_map[5], 4);
}

TEST(UpdatePartitionTest, InsertBlueAndRed) {
  DynamicClusteredGraph graph;
  absl::flat_hash_map<NodeId, NodeId> node_sizes = {{0, 1}, {1, 1}, {2, 1},
                                                    {3, 2}, {4, 1}, {5, 1}};
  const absl::flat_hash_set<NodeId> neighbors_deleted;

  ASSERT_OK_AND_ASSIGN(const auto new_nodes, ConvertToAdjList({{0, 3, 0.5},
                                                               {1, 3, 0.6},
                                                               {2, 3, 0.7},
                                                               {0, 4, 0.4},
                                                               {5, 3, 0.5},
                                                               {5, 4, 0.6}},
                                                              node_sizes));
  EXPECT_OK(graph.AddNodes(new_nodes));

  const DynamicHacNodeColorTest color({{0, kBlue, 0},
                                       {1, kRed, 0},
                                       {2, kBlue, 0},
                                       {3, kRed, 0},
                                       {4, kRed, 0},
                                       {5, kBlue, 0}});
  absl::flat_hash_map<NodeId, NodeId> partition_map;

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions(new_nodes, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 4);
  EXPECT_EQ(partition_map[1], 1);
  EXPECT_EQ(partition_map[2], 3);
  EXPECT_EQ(partition_map[3], 3);
  EXPECT_EQ(partition_map[4], 4);
  EXPECT_EQ(partition_map[5], 4);
}

TEST(UpdatePartitionTest, TargetDeleted) {
  // Node 1 pointed to 2, but 2 is now deleted. Edge (0,1) is inserted.
  DynamicClusteredGraph graph;
  absl::flat_hash_map<NodeId, NodeId> node_sizes = {{0, 1}, {1, 1}, {2, 1}};
  const absl::flat_hash_set<NodeId> deleted_nodes = {2};
  const absl::flat_hash_set<NodeId> neighbors_deleted = {1};

  ASSERT_OK_AND_ASSIGN(const auto adj_list_vec,
                       ConvertToAdjList({{0, 1, 0.5}}, node_sizes));
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  const DynamicHacNodeColorTest color({{0, kRed, 0}, {1, kBlue, 0}});
  absl::flat_hash_map<NodeId, NodeId> partition_map = {{1, 2}};

  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{1, 0.5}}};

  ASSERT_OK_AND_ASSIGN(const auto changes,
                       UpdatePartitions({adj_list_0}, neighbors_deleted, graph,
                                        color, partition_map));
  EXPECT_EQ(partition_map[0], 0);
  EXPECT_EQ(partition_map[1], 0);
}

TEST(GetDirtyPartitionsTest, DirtyPartitions) {
  DynamicClusteredGraph graph;
  ASSERT_OK_AND_ASSIGN(const auto adj_list_vec,
                       ConvertToAdjList({{0, 3}, {1, 3}, {2, 3}}));
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const std::vector<std::tuple<NodeId, NodeId, NodeId>> changes{
      {0, 4, 3}, {3, -1, 3}, {2, 2, 3}};
  const auto color = DynamicHacNodeColorTest(
      DynamicHacNodeColorTest({{2, NodeColor::kBlue, 1},
                               {3, NodeColor::kRed, 1},
                               {4, NodeColor::kRed, 1}}));
  ASSERT_OK_AND_ASSIGN(const auto dirty_partitions,
                       DirtyPartitions(changes, graph, color));

  EXPECT_THAT(dirty_partitions, UnorderedElementsAre(3));
}

TEST(GetDirtyPartitionsTest, SingletonBlue) {
  // Singleton blue dirty partitions should be returned, even though they are
  // not truly dirty.
  DynamicClusteredGraph graph;
  ASSERT_OK_AND_ASSIGN(const auto adj_list_vec, ConvertToAdjList({{0, 1}}));
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const std::vector<std::tuple<NodeId, NodeId, NodeId>> changes{{0, 0, 0}};
  const auto color = DynamicHacNodeColorTest(
      DynamicHacNodeColorTest({{0, NodeColor::kBlue, 1}}));
  ASSERT_OK_AND_ASSIGN(const auto dirty_partitions,
                       DirtyPartitions(changes, graph, color));

  EXPECT_THAT(dirty_partitions, UnorderedElementsAre(0));
}

TEST(GetDirtyPartitionsTest, BlueToRed) {
  // Singleton blue partition node changes partition. The old singleton blue
  // partition is not a partition anymore, and should not be returned.
  DynamicClusteredGraph graph;
  ASSERT_OK_AND_ASSIGN(const auto adj_list_vec, ConvertToAdjList({{0, 1}}));
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  const std::vector<std::tuple<NodeId, NodeId, NodeId>> changes{{0, 0, 1}};
  const auto color = DynamicHacNodeColorTest(DynamicHacNodeColorTest(
      {{0, NodeColor::kBlue, 1}, {1, NodeColor::kRed, 1}}));
  ASSERT_OK_AND_ASSIGN(const auto dirty_partitions,
                       DirtyPartitions(changes, graph, color));

  EXPECT_THAT(dirty_partitions, UnorderedElementsAre(1));
}

TEST(LeafToRootIdTest, Empty) {
  graph_mining::in_memory::Dendrogram dendrogram(0);
  const auto to_cluster_id = LeafToRootId(dendrogram);
  EXPECT_THAT(to_cluster_id, IsEmpty());
}

TEST(LeafToRootIdTest, MultipleTrees) {
  //           9
  //          / \
  //   7     8   \
  //  / \   / \   \
  // 0   1  2  3  4  5  6
  graph_mining::in_memory::Dendrogram dendrogram(7);

  EXPECT_OK(dendrogram.BinaryMerge(0, 1, 0.5));
  EXPECT_OK(dendrogram.BinaryMerge(2, 3, 0.5));
  EXPECT_OK(dendrogram.BinaryMerge(4, 8, 0.5));

  const auto to_cluster_id = LeafToRootId(dendrogram);
  EXPECT_THAT(to_cluster_id, ElementsAre(7, 7, 9, 9, 9, 5, 6));
}

TEST(LocalMinMergeSimilaritiesTest, MultipleTrees) {
  //           9
  //          / \
  //   7     8   \
  //  / \   / \   \
  // 0   1  2  3  4  5  6
  const size_t partition_num_nodes = 7;
  const auto inf = std::numeric_limits<double>::infinity();
  const std::vector<std::tuple<NodeId, NodeId, double>> merges{
      {0, 1, 0.5}, {2, 3, 0.5}, {4, 8, 0.4}};
  const std::vector<NodeId> node_map{10, 11, 12, 13, 14, 15, 16};
  const absl::flat_hash_map<NodeId, double> min_merge_similarities{
      {10, 0.3}, {11, 1},   {12, inf}, {13, inf},
      {14, inf}, {15, 0.6}, {16, 0.7}};

  auto min_merge_similarities_partition_map =
      SubgraphMinMergeSimilarity(node_map, min_merge_similarities);
  ASSERT_OK_AND_ASSIGN(
      const auto min_similarities,
      LocalMinMergeSimilarities(merges, min_merge_similarities_partition_map,
                                partition_num_nodes));
  EXPECT_THAT(min_similarities, ElementsAre(0.3, 1, inf, inf, inf, 0.6, 0.7,
                                            0.3, 0.5, 0.4, inf, inf, inf));
}

}  // namespace
}  // namespace graph_mining::in_memory
