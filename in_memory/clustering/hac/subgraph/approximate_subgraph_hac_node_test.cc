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

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_node.h"

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining {
namespace in_memory {
namespace {

using NodeId = InMemoryClusterer::NodeId;

namespace {

ApproximateSubgraphHacNode CreateNodeWith100Neighbors() {
  absl::flat_hash_map<NodeId, double> neighbors;
  for (size_t i = 0; i < 100; ++i) {
    neighbors.insert({i, 42.0 * i});
  }
  ApproximateSubgraphHacNode node(1, 1.1);
  for (const auto& [neighbor, weight] : neighbors) {
    node.InsertEdge(neighbor, 1, weight);
  }
  return node;
}

ApproximateSubgraphHacNode CreateEmptyNode() {
  ApproximateSubgraphHacNode node(1, 1.1);
  return node;
}

}  // namespace

TEST(ApproximateSubgraphHacNodeTest, IsolatedNode) {
  ApproximateSubgraphHacNode node(1, 1.1);
  EXPECT_EQ(node.Neighbors().size(), 0);
  EXPECT_EQ(node.NumAssignedEdges(), 0);
}

TEST(ApproximateSubgraphHacNodeTest, NodeWithNeighbors) {
  auto node = CreateNodeWith100Neighbors();
  EXPECT_EQ(node.Neighbors().size(), 100);
}

TEST(ApproximateSubgraphHacNodeTest, TestEdgeAssignment) {
  auto node = CreateEmptyNode();
  node.InsertEdge(242, 5, 0.5);
  EXPECT_EQ(node.NumAssignedEdges(), 0);
  node.AssignEdge(242, 1.0);
  EXPECT_EQ(node.NumAssignedEdges(), 1);

  EXPECT_EQ(node.GetNeighborInfo(242).partial_weight, 0.5);
  EXPECT_EQ(node.GetNeighborInfo(242).cluster_size, 5);
  EXPECT_EQ(node.GetNeighborInfo(242).goodness, 1.0);
}

TEST(ApproximateSubgraphHacNodeTest, TestPartialWeight) {
  ApproximateSubgraphHacNode node(100, 1.1);

  node.InsertEdge(242, 2, 100);
  EXPECT_EQ(node.EdgeWeight(242, 2), 100);
  EXPECT_TRUE(node.IsNeighbor(242));

  EXPECT_EQ(node.GetNeighborInfo(242).partial_weight, 10000);  // 100*100
  EXPECT_EQ(node.GetNeighborInfo(242).cluster_size, 2);
  EXPECT_EQ(node.GetNeighborInfo(242).goodness, node.kDefaultGoodness);
  EXPECT_EQ(node.NumAssignedEdges(), 0);

  node.AssignEdge(242, 1.0);
  EXPECT_EQ(node.NumAssignedEdges(), 1);
  EXPECT_EQ(node.GetNeighborInfo(242).partial_weight, 10000);
  EXPECT_EQ(node.GetNeighborInfo(242).cluster_size, 2);
  EXPECT_EQ(node.GetNeighborInfo(242).goodness, 1.0);
}

TEST(ApproximateSubgraphHacNodeTest, TestMerge) {
  std::vector<ApproximateSubgraphHacNode> nodes;
  // Create four nodes, each with a cluster size of 1
  double one_plus_alpha = 1.1;
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  std::vector<bool> is_active = {true, true, true, true};

  // Edges from 0
  nodes[0].InsertEdge(1, 1, 1.0);
  nodes[0].InsertEdge(2, 1, 3.0);
  nodes[0].InsertEdge(3, 1, 3.0);
  // Edges from 1
  nodes[1].InsertEdge(0, 1, 1.0);
  nodes[1].InsertEdge(2, 1, 1.0);
  nodes[1].InsertEdge(3, 1, 1.0);
  // Reciprocal edges
  nodes[2].InsertEdge(0, 1, 3.0);
  nodes[2].InsertEdge(1, 1, 1.0);
  nodes[3].InsertEdge(0, 1, 3.0);
  nodes[3].InsertEdge(1, 1, 1.0);

  // Assign all edges to 0 and 1. The goodness values don't really matter here.
  nodes[0].AssignEdge(1, 1.0);
  nodes[0].AssignEdge(2, 1.0);
  nodes[0].AssignEdge(3, 1.0);
  nodes[1].AssignEdge(2, 1.0);
  nodes[1].AssignEdge(3, 1.0);

  EXPECT_EQ(nodes[0].Neighbors().size(), 3);
  EXPECT_EQ(nodes[1].Neighbors().size(), 3);
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 3);

  // Merge (0, 1).
  auto [edges_to_reassign, nodes_to_update] = nodes[0].Merge(
      0, 1, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active);
  EXPECT_EQ(nodes[0].Neighbors().size(), 0);
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[1].Neighbors().size(), 2);
  EXPECT_EQ(nodes[1].CurrentClusterSize(), 2);
  EXPECT_EQ(nodes[1].NumAssignedEdges(), 0);

  // Weights are now (3+1)/2.
  EXPECT_EQ(nodes[1].EdgeWeight(2, 1), 2);
  EXPECT_EQ(nodes[1].EdgeWeight(3, 1), 2);

  EXPECT_TRUE(nodes[1].MaybeBroadcastClusterSize(
      1, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active));
  absl::flat_hash_set<NodeId> nodes_to_update_in_pq;

  // The best edge for 1 goes up---we don't need to reassign edges.
  auto get_goodness = [&](NodeId node_u, NodeId node_v) { return 1.0; };
  EXPECT_FALSE(nodes[1].MaybeReassignEdges(
      1, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active, &nodes_to_update_in_pq, get_goodness));

  // Check the neighbors of 1.
  auto edges_for_one = nodes[1].Neighbors();
  EXPECT_FALSE(edges_for_one.contains(0));
  EXPECT_TRUE(edges_for_one.contains(2));
  EXPECT_TRUE(edges_for_one.contains(3));

  // No node should have any edges assigned to it.
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[1].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[2].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[3].NumAssignedEdges(), 0);

  std::vector<std::pair<NodeId, NodeId>> vec = {{1, 2}, {1, 3}};
  EXPECT_THAT(edges_to_reassign, testing::UnorderedElementsAreArray(vec));

  // No elements in the PQ since no edges are reassigned.
  EXPECT_THAT(nodes_to_update_in_pq, testing::UnorderedElementsAre());
}

TEST(ApproximateSubgraphHacNodeTest, TestMergeDecreaseBestWeight) {
  std::vector<ApproximateSubgraphHacNode> nodes;
  // Create four nodes, each with a cluster size of 1
  double one_plus_alpha = 1.1;
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  nodes.push_back(ApproximateSubgraphHacNode(1, one_plus_alpha));
  std::vector<bool> is_active = {true, true, true, true};

  // Edges from 0
  nodes[0].InsertEdge(1, 1, 3.0);
  nodes[0].InsertEdge(2, 1, 3.0);
  nodes[0].InsertEdge(3, 1, 3.0);
  // Edges from 1
  nodes[1].InsertEdge(0, 1, 3.0);
  nodes[1].InsertEdge(2, 1, 1.0);
  nodes[1].InsertEdge(3, 1, 1.0);
  // Reciprocal edges
  nodes[2].InsertEdge(0, 1, 3.0);
  nodes[2].InsertEdge(1, 1, 1.0);
  nodes[3].InsertEdge(0, 1, 3.0);
  nodes[3].InsertEdge(1, 1, 1.0);

  // Assign all edges to 0 and 1. The goodness values don't really matter here.
  nodes[0].AssignEdge(1, 1.0);
  nodes[0].AssignEdge(2, 1.0);
  nodes[0].AssignEdge(3, 1.0);
  nodes[1].AssignEdge(2, 1.0);
  nodes[1].AssignEdge(3, 1.0);

  EXPECT_EQ(nodes[0].Neighbors().size(), 3);
  EXPECT_EQ(nodes[1].Neighbors().size(), 3);
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 3);

  EXPECT_EQ(nodes[1].ApproximateBestWeightAndId().first, 3);
  EXPECT_EQ(nodes[1].ApproximateBestWeightAndId().second, 0);

  EXPECT_EQ(nodes[2].ApproximateBestWeightAndId().first, 3);
  EXPECT_EQ(nodes[2].ApproximateBestWeightAndId().second, 0);

  EXPECT_EQ(nodes[3].ApproximateBestWeightAndId().first, 3);
  EXPECT_EQ(nodes[3].ApproximateBestWeightAndId().second, 0);

  // Merge (1, 0).
  auto [edges_to_reassign, nodes_to_update] = nodes[1].Merge(
      1, 0, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active);
  EXPECT_EQ(nodes[0].Neighbors().size(), 2);
  EXPECT_EQ(nodes[0].CurrentClusterSize(), 2);
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[1].Neighbors().size(), 0);
  EXPECT_EQ(nodes[1].NumAssignedEdges(), 0);

  // Weights are now (3+1)/2.
  EXPECT_EQ(nodes[0].EdgeWeight(2, 1), 2);
  EXPECT_EQ(nodes[0].EdgeWeight(3, 1), 2);

  EXPECT_TRUE(nodes[0].MaybeBroadcastClusterSize(
      1, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active));
  absl::flat_hash_set<NodeId> nodes_to_update_in_pq;

  // The best edge for 0 goes down---we may need to reassign edges.
  auto get_goodness = [&](NodeId node_u, NodeId node_v) { return 1.0; };
  EXPECT_TRUE(nodes[0].MaybeReassignEdges(
      1, absl::Span<ApproximateSubgraphHacNode>(nodes.data(), nodes.size()),
      is_active, &nodes_to_update_in_pq, get_goodness));

  // Check the neighbors of 1.
  auto edges_for_one = nodes[0].Neighbors();
  EXPECT_FALSE(edges_for_one.contains(1));
  EXPECT_TRUE(edges_for_one.contains(2));
  EXPECT_TRUE(edges_for_one.contains(3));

  // No node should have any edges assigned to it.
  EXPECT_EQ(nodes[0].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[1].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[2].NumAssignedEdges(), 0);
  EXPECT_EQ(nodes[3].NumAssignedEdges(), 0);

  std::vector<std::pair<NodeId, NodeId>> vec = {{0, 2}, {0, 3}};
  EXPECT_THAT(edges_to_reassign, testing::UnorderedElementsAreArray(vec));

  // No elements in the PQ since no edges are reassigned.
  EXPECT_THAT(nodes_to_update_in_pq, testing::UnorderedElementsAre());
}

TEST(ApproximateSubgraphHacNodeTest, GetGoodEdge) {
  ApproximateSubgraphHacNode node(100, 1.1);

  node.InsertEdge(0, 2, 100);
  EXPECT_EQ(node.EdgeWeight(0, 2), 100);
  auto get_goodness = [&](NodeId node_u, NodeId node_v) { return 1.0; };
  node.AssignEdge(0, 1.0);

  EXPECT_EQ(node.NumAssignedEdges(), 1);
  EXPECT_EQ(node.GetGoodEdge(1, /*threshold=*/1, get_goodness).first, 1.0);
  EXPECT_EQ(node.GetGoodEdge(1, /*threshold=*/1, get_goodness).second, 0);

  auto get_goodness_2 = [&](NodeId node_u, NodeId node_v) {
    return node.kDefaultGoodness;
  };
  EXPECT_EQ(node.GetGoodEdge(1, /*threshold=*/1, get_goodness_2).first,
            node.kDefaultGoodness);
  EXPECT_EQ(node.GetGoodEdge(1, /*threshold=*/1, get_goodness_2).second,
            std::numeric_limits<NodeId>::max());
}

}  // namespace
}  // namespace in_memory
}  // namespace graph_mining
