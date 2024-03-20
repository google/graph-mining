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

#include "in_memory/clustering/dynamic/hac/dynamic_hac_updater.h"

#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"
#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"
#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {
namespace {

using graph_mining::in_memory::NodeId;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using graph_mining::in_memory::SimpleUndirectedGraph;

TEST(UpdateDendrogramTest, NoMerge) {
  std::vector<NodeId> node_map;
  const std::vector<std::tuple<NodeId, NodeId, double>> merges;
  DynamicDendrogram dynamic_dendrogram;

  auto next_unused_node_id = NextUnusedId(101);

  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));
  EXPECT_EQ(dynamic_dendrogram.NumNodes(), 0);
  EXPECT_THAT(node_map, IsEmpty());
}

TEST(UpdateDendrogramTest, TwoNodesMerge) {
  // A graph with edge 10 -- 11. The two nodes are both active.
  const std::vector<std::tuple<NodeId, NodeId, double>> merges{{0, 1, 0.5}};
  std::vector<NodeId> node_map{10, 11};
  DynamicDendrogram dynamic_dendrogram;
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(10));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(11));

  auto next_unused_node_id = NextUnusedId(101);

  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));

  EXPECT_EQ(node_map.size(), 3);
  EXPECT_EQ(node_map[2], 101);
  EXPECT_THAT(dynamic_dendrogram.Parent(10), IsOkAndHolds(FieldsAre(101, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(11), IsOkAndHolds(FieldsAre(101, 0.5)));
}

TEST(UpdateDendrogramTest, TwoMerges) {
  const std::vector<std::tuple<NodeId, NodeId, double>> merges{{0, 1, 0.5},
                                                               {3, 2, 0.2}};
  std::vector<NodeId> node_map{10, 11, 12};

  // no merge in dendrogram
  DynamicDendrogram dynamic_dendrogram;
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(10));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(11));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(12));

  auto next_unused_node_id = NextUnusedId(101);

  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));

  EXPECT_THAT(node_map, ElementsAre(10, 11, 12, 101, 102));
  EXPECT_THAT(dynamic_dendrogram.Parent(10), IsOkAndHolds(FieldsAre(101, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(11), IsOkAndHolds(FieldsAre(101, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(12),
              IsOkAndHolds(FieldsAre(Eq(102), FloatEq(0.2))));
  EXPECT_THAT(dynamic_dendrogram.Parent(101),
              IsOkAndHolds(FieldsAre(Eq(102), FloatEq(0.2))));

  // Both merges are in dendrogram.
  dynamic_dendrogram.Clear();
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(10));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(11));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(12));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(13, 10, 11, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(14, 13, 12, 0.2));
  next_unused_node_id = NextUnusedId(101);
  node_map = {10, 11, 12};
  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));
  // mapped to original internal nodes.
  EXPECT_THAT(node_map, ElementsAre(10, 11, 12, 13, 14));
  EXPECT_THAT(dynamic_dendrogram.Parent(10), IsOkAndHolds(FieldsAre(13, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(11), IsOkAndHolds(FieldsAre(13, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(12),
              IsOkAndHolds(FieldsAre(Eq(14), FloatEq(0.2))));
  EXPECT_THAT(dynamic_dendrogram.Parent(13),
              IsOkAndHolds(FieldsAre(Eq(14), FloatEq(0.2))));

  // One merge is in dendrogram.
  dynamic_dendrogram.Clear();
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(10));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(11));
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(12));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(13, 10, 11, 0.5));
  next_unused_node_id = NextUnusedId(101);
  node_map = {10, 11, 12};
  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));
  EXPECT_THAT(node_map, ElementsAre(10, 11, 12, 13, 101));
  EXPECT_THAT(dynamic_dendrogram.Parent(10), IsOkAndHolds(FieldsAre(13, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(11), IsOkAndHolds(FieldsAre(13, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(12),
              IsOkAndHolds(FieldsAre(Eq(101), FloatEq(0.2))));
  EXPECT_THAT(dynamic_dendrogram.Parent(13),
              IsOkAndHolds(FieldsAre(Eq(101), FloatEq(0.2))));
}

TEST(UpdateDendrogramTest, MultipleMerges) {
  //   Original dendrogram looks like below. X nodes are invalid after new
  //   merges are propagated. 83 is merged with 84, but not in this round.
  //           82X    (85)
  //          / \     / \
  //   80    81  \   83  (84)
  //  / \   / \   \  / \
  // 0   1  2  3  4  5  6  7
  // Dendrogram after
  //                  /
  //  80     81      83    101
  // /  \   / \      / \   / \
  // 0   1  2  3     5  6  7  4
  std::vector<NodeId> node_map{0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<std::tuple<NodeId, NodeId, double>> merges{
      {2, 3, 0.5}, {4, 7, 0.4}, {5, 6, 0.5}};
  DynamicDendrogram dynamic_dendrogram;

  for (auto i : node_map) {
    ASSERT_OK(dynamic_dendrogram.AddLeafNode(i));
  }
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(84));

  ASSERT_OK(dynamic_dendrogram.AddInternalNode(80, 0, 1, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(81, 2, 3, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(82, 81, 4, 0.3));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(83, 5, 6, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(85, 83, 84, 0.5));

  auto next_unused_node_id = NextUnusedId(101);
  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));
  EXPECT_THAT(node_map, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 81, 101, 83));

  EXPECT_EQ(dynamic_dendrogram.HasNode(82), false);
  EXPECT_THAT(dynamic_dendrogram.HasValidParent(81), IsOkAndHolds(false));
  EXPECT_THAT(dynamic_dendrogram.HasValidParent(101), IsOkAndHolds(false));

  EXPECT_THAT(dynamic_dendrogram.Parent(0), IsOkAndHolds(FieldsAre(80, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(1), IsOkAndHolds(FieldsAre(80, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(2), IsOkAndHolds(FieldsAre(81, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(3), IsOkAndHolds(FieldsAre(81, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(5),
              IsOkAndHolds(FieldsAre(Eq(83), FloatEq(0.5))));
  EXPECT_THAT(dynamic_dendrogram.Parent(6),
              IsOkAndHolds(FieldsAre(Eq(83), FloatEq(0.5))));
  EXPECT_THAT(dynamic_dendrogram.Parent(4), IsOkAndHolds(FieldsAre(101, 0.4)));
  EXPECT_THAT(dynamic_dendrogram.Parent(7), IsOkAndHolds(FieldsAre(101, 0.4)));
  EXPECT_THAT(dynamic_dendrogram.Parent(83), IsOkAndHolds(FieldsAre(85, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(84), IsOkAndHolds(FieldsAre(85, 0.5)));
}

TEST(UpdateDendrogramTest, AllDendrogramIsValid) {
  //   Original dendrogram looks like below. All nodes are valid after new
  //   merges are propagated. 83 is merged with 84, but not in this round.
  //           82    (85)
  //          / \     / \
  //   80   81   \   83  (84)
  //  / \   / \   \  / \
  // 0   1  2  3  4  5  6
  // Dendrogram after is the same.
  std::vector<NodeId> node_map{0, 1, 2, 3, 4, 5, 6};
  const std::vector<std::tuple<NodeId, NodeId, double>> merges{{2, 3, 0.5},
                                                               {5, 6, 0.5}};

  DynamicDendrogram dynamic_dendrogram;

  for (auto i : node_map) {
    ASSERT_OK(dynamic_dendrogram.AddLeafNode(i));
  }
  ASSERT_OK(dynamic_dendrogram.AddLeafNode(84));

  ASSERT_OK(dynamic_dendrogram.AddInternalNode(80, 0, 1, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(81, 2, 3, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(82, 81, 4, 0.3));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(83, 5, 6, 0.5));
  ASSERT_OK(dynamic_dendrogram.AddInternalNode(85, 83, 84, 0.5));

  auto next_unused_node_id = NextUnusedId(101);
  ASSERT_OK(UpdateDendrogram(merges, node_map, next_unused_node_id,
                             dynamic_dendrogram));
  EXPECT_THAT(node_map, ElementsAre(0, 1, 2, 3, 4, 5, 6, 81, 83));

  EXPECT_THAT(dynamic_dendrogram.Parent(0), IsOkAndHolds(FieldsAre(80, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(1), IsOkAndHolds(FieldsAre(80, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(2), IsOkAndHolds(FieldsAre(81, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(3), IsOkAndHolds(FieldsAre(81, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(4), IsOkAndHolds(FieldsAre(82, 0.3)));
  EXPECT_THAT(dynamic_dendrogram.Parent(81), IsOkAndHolds(FieldsAre(82, 0.3)));
  EXPECT_THAT(dynamic_dendrogram.Parent(5),
              IsOkAndHolds(FieldsAre(Eq(83), FloatEq(0.5))));
  EXPECT_THAT(dynamic_dendrogram.Parent(6),
              IsOkAndHolds(FieldsAre(Eq(83), FloatEq(0.5))));
  EXPECT_THAT(dynamic_dendrogram.Parent(83), IsOkAndHolds(FieldsAre(85, 0.5)));
  EXPECT_THAT(dynamic_dendrogram.Parent(84), IsOkAndHolds(FieldsAre(85, 0.5)));
}

absl::StatusOr<std::unique_ptr<ContractedGraph>> GetContractedGraph() {
  const std::vector<std::pair<NodeId, NodeId>> edges{
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
  ASSIGN_OR_RETURN(const auto adj_lists, ConvertToAdjList(edges));

  auto graph = std::make_unique<SimpleUndirectedGraph>();
  RETURN_IF_ERROR(graph->PrepareImport(6));
  for (const auto& adj_list : adj_lists) {
    RETURN_IF_ERROR(graph->Import(adj_list));
  }
  RETURN_IF_ERROR(graph->FinishImport());
  graph->SetNodeWeight(3, -1);
  RETURN_IF_ERROR(graph->SetEdgeWeight(0, 1, 2));
  RETURN_IF_ERROR(graph->SetEdgeWeight(4, 5, 2));
  std::vector<double> min_merge_similarities(
      graph->NumNodes(), std::numeric_limits<double>::infinity());
  ASSIGN_OR_RETURN(auto subgraph_hac_result,
                   ApproximateSubgraphHac(
                       std::move(graph), std::move(min_merge_similarities), 0));
  return std::move(subgraph_hac_result.contracted_graph);
}

TEST(AdjacencyListsOfNewNodesTest, AdjacencyListsOfNewNodesTest) {
  // 10 - 11 - 12 - 13 - 14 - 15.. All nodes have weight 1, except node 13 has
  // weight 3.
  DynamicClusteredGraph graph;
  const std::vector<std::tuple<NodeId, NodeId, double>> edges_global{
      {10, 11, 1}, {11, 12, 1}, {12, 13, 1}, {13, 14, 1}, {14, 15, 1}};
  const absl::flat_hash_map<NodeId, NodeId> node_weights_global{
      {10, 1}, {11, 1}, {12, 1}, {13, 3}, {14, 1}, {15, 1}};
  ASSERT_OK_AND_ASSIGN(const auto adj_lists_global,
                       ConvertToAdjList(edges_global, node_weights_global));
  ASSERT_OK(graph.AddNodes(adj_lists_global));

  // 1 - 2 - 3 - 5. 0, 3, and 4 are inactive.
  ASSERT_OK_AND_ASSIGN(const auto contracted_graph, GetContractedGraph());

  const std::vector<graph_mining::in_memory::NodeId> to_cluster_ids{6, 6, 2,
                                                                    3, 7, 7};
  const std::vector<graph_mining::in_memory::NodeId> node_map{10, 11, 12,  13,
                                                              14, 15, 101, 102};
  const absl::flat_hash_map<NodeId, NodeId> next_round_node_map{
      {10, 101}, {11, 101}, {12, 12}, {13, 13}, {14, 102}, {15, 102}};

  const std::vector<NodeId> new_node_ids{4};
  const absl::flat_hash_set<NodeId> delete_nodes;

  const auto subgraph_cluster_id = SubgraphClusterId(node_map, to_cluster_ids);
  ASSERT_OK_AND_ASSIGN(const auto new_nodes,
                       AdjacencyListsOfNewNodes(
                           new_node_ids, subgraph_cluster_id, contracted_graph,
                           next_round_node_map, delete_nodes, graph));
  // Edge weight is 3 because inactive node 13 has weight 3 in `graph`.
  EXPECT_THAT(new_nodes,
              UnorderedElementsAre(FieldsAre(
                  102, 2, UnorderedElementsAre(Pair(13, 3)), std::nullopt)));
}

absl::StatusOr<std::unique_ptr<ContractedGraph>> GetContractedGraphSmall() {
  const std::vector<std::pair<NodeId, NodeId>> edges{{0, 1}, {0, 2}};
  ASSIGN_OR_RETURN(const auto adj_lists, ConvertToAdjList(edges));

  auto graph = std::make_unique<SimpleUndirectedGraph>();
  RETURN_IF_ERROR(graph->PrepareImport(3));
  for (const auto& adj_list : adj_lists) {
    RETURN_IF_ERROR(graph->Import(adj_list));
  }
  RETURN_IF_ERROR(graph->FinishImport());
  graph->SetNodeWeight(2, -1);
  std::vector<double> min_merge_similarities(
      graph->NumNodes(), std::numeric_limits<double>::infinity());
  ASSIGN_OR_RETURN(auto subgraph_hac_result,
                   ApproximateSubgraphHac(
                       std::move(graph), std::move(min_merge_similarities), 0));
  return std::move(subgraph_hac_result.contracted_graph);
}

TEST(AdjacencyListsOfNewNodesTest, IgnoreNegativeOneNodeTest) {
  // 10 - 11
  //    \ 12. All nodes have weight 1. Node 12 maps to -1.
  DynamicClusteredGraph graph;
  const std::vector<std::tuple<NodeId, NodeId, double>> edges_global{
      {10, 11, 1}, {10, 12, 1}};
  const absl::flat_hash_map<NodeId, NodeId> node_weights_global{
      {10, 1}, {11, 1}, {12, 1}};
  ASSERT_OK_AND_ASSIGN(const auto adj_lists_global,
                       ConvertToAdjList(edges_global, node_weights_global));
  ASSERT_OK(graph.AddNodes(adj_lists_global));

  // 3 - 2. 2 is inactive.
  ASSERT_OK_AND_ASSIGN(const auto contracted_graph, GetContractedGraphSmall());

  const std::vector<graph_mining::in_memory::NodeId> to_cluster_ids{3, 3, 2, 3};
  const std::vector<graph_mining::in_memory::NodeId> node_map{10, 11, 12, 101};
  const absl::flat_hash_map<NodeId, NodeId> next_round_node_map{
      {10, 101}, {11, 101}, {12, -1}};

  const std::vector<NodeId> new_node_ids{0};
  const absl::flat_hash_set<NodeId> delete_nodes;

  const auto subgraph_cluster_id = SubgraphClusterId(node_map, to_cluster_ids);
  ASSERT_OK_AND_ASSIGN(const auto new_nodes,
                       AdjacencyListsOfNewNodes(
                           new_node_ids, subgraph_cluster_id, contracted_graph,
                           next_round_node_map, delete_nodes, graph));

  EXPECT_THAT(new_nodes,
              UnorderedElementsAre(FieldsAre(101, 2, IsEmpty(), std::nullopt)));
}

TEST(UpdateMappingLastRoundTest, UpdateMappingLastRoundTest) {
  std::vector<NodeId> node_map = {10, 11, 12, 13, 14, 15, 101, 102};
  const int num_active_nodes = 3;
  absl::Span<const NodeId> active_nodes =
      absl::Span<const NodeId>(node_map.data(), num_active_nodes);
  const std::vector<NodeId> to_cluster_id = {7, 7, 7, 3, 4, 5};
  const absl::flat_hash_map<NodeId, NodeId> root_map{
      {10, 102}, {11, 102}, {12, 102}, {13, 13}, {14, 14}, {15, 15}};

  ASSERT_OK_AND_ASSIGN(const auto next_round_node_map,
                       MappingLastRound(active_nodes, root_map));
  EXPECT_THAT(
      next_round_node_map,
      UnorderedElementsAre(Pair(10, 102), Pair(11, 102), Pair(12, 102)));
}

TEST(NodesToDeleteTest, NodesToDeleteTest) {
  const std::vector<std::tuple<NodeId, NodeId, NodeId>> current_mapping{
      {0, 10, 100}, {1, 11, 100}, {2, 12, 12},  {3, 13, 13},
      {4, 14, 14},  {5, 15, 102}, {-1, 16, 18}, {-1, 17, 19}};
  std::vector<NodeId> node_map = {10, 11, 12, 13, 14, 15, 101, 102};
  const std::vector<NodeId> to_cluster_id = {7, 7, 7, 3, 4, 5};
  const absl::flat_hash_set<NodeId> active_contracted_nodes{102, 13, 14, 15,
                                                            19};
  absl::flat_hash_map<NodeId, NodeId> next_round_node_map;
  const absl::flat_hash_map<NodeId, NodeId> root_map{
      {10, 102}, {11, 102}, {12, 102}, {13, 13}, {14, 14}, {15, 15}};

  ASSERT_OK_AND_ASSIGN(
      const auto nodes_to_delete,
      NodesToDelete(current_mapping, root_map, active_contracted_nodes,
                    next_round_node_map));

  // 100 and 12 are not mapped to anymore in `subgraph_cluster_id`. 18 was
  // mapped to from a deleted node 16.  19 was also mapped to from deleted node,
  // but it is not returned because it is in `active_contracted_nodes`.
  EXPECT_THAT(nodes_to_delete, UnorderedElementsAre(100, 12, 18));
}

TEST(CurrentMapNextRoundTest, CurrentMapNextRoundTest) {
  std::vector<NodeId> node_map = {10, 11, 12};
  const int num_active_nodes = 3;
  absl::Span<const NodeId> active_nodes =
      absl::Span<const NodeId>(node_map.data(), num_active_nodes);
  const absl::flat_hash_set<NodeId> deleted_nodes{13, 14};
  const absl::flat_hash_set<NodeId> new_nodes{10};
  const absl::flat_hash_map<NodeId, NodeId> next_round_node_map{
      {11, 101}, {12, 102}, {13, 103}, {14, 104}};

  ASSERT_OK_AND_ASSIGN(const auto current_map_next_round,
                       CurrentMapNextRound(active_nodes, deleted_nodes,
                                           new_nodes, next_round_node_map));
  EXPECT_THAT(
      current_map_next_round,
      UnorderedElementsAre(FieldsAre(-1, 13, 103), FieldsAre(-1, 14, 104),
                           FieldsAre(0, 10, -1), FieldsAre(1, 11, 101),
                           FieldsAre(2, 12, 102)));
}

}  // namespace
}  // namespace graph_mining::in_memory
