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

#include "in_memory/clustering/dynamic/hac/dynamic_clustered_graph.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {

namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;


using AdjacencyList =
    graph_mining::in_memory::InMemoryClusterer::Graph::AdjacencyList;
using NodeId = graph_mining::in_memory::NodeId;

TEST(ConstructionTest, Empty) {
  DynamicClusteredGraph graph;
  EXPECT_EQ(graph.NumNodes(), 0);
  EXPECT_FALSE(graph.ContainsNode(0));
  EXPECT_THAT(graph.ImmutableNode(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in graph")));
  EXPECT_THAT(graph.RemoveNode(0), StatusIs(absl::StatusCode::kNotFound,
                                            HasSubstr("node not in graph")));
  EXPECT_EQ(graph.NumEdges(), 0);
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());

  EXPECT_THAT(graph.Nodes(), IsEmpty());
}

TEST(ConstructionTest, AddNegativeWeightNode) {
  DynamicClusteredGraph graph;
  AdjacencyList adj_list = {/*id=*/0, /*weight=*/-1, /*outgoing_edges=*/{}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list};
  EXPECT_THAT(graph.AddNodes(adj_list_vec),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node weight is non-positive")));
}

TEST(ConstructionTest, AddEdgeToNonExistingNode) {
  DynamicClusteredGraph graph;
  AdjacencyList adj_list = {/*id=*/0, /*weight=*/1,
                            /*outgoing_edges=*/{{1, 0}}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list};
  EXPECT_THAT(graph.AddNodes(adj_list_vec),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("edge to non-existing node")));
}

TEST(ConstructionTest, AddDuplicateNode) {
  DynamicClusteredGraph graph;
  AdjacencyList adj_list = {/*id=*/0, /*weight=*/1, /*outgoing_edges=*/{}};
  AdjacencyList adj_list_duplicate = {/*id=*/0, /*weight=*/1,
                                      /*outgoing_edges=*/{}};

  std::vector<AdjacencyList> adj_list_vec = {adj_list, adj_list_duplicate};
  EXPECT_THAT(graph.AddNodes(adj_list_vec),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("node already exists")));
}

TEST(ConstructionTest, AddSelfEdge) {
  DynamicClusteredGraph graph;
  AdjacencyList adj_list = {/*id=*/0, /*weight=*/1,
                            /*outgoing_edges=*/{{0, 0}}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list};
  EXPECT_THAT(graph.AddNodes(adj_list_vec),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("self edge should not exist")));
}

TEST(ConstructionTest, AddOneNode) {
  DynamicClusteredGraph graph;
  // Add a node with no neighbor and weight 3. Node id is 0.
  AdjacencyList adj_list = {/*id=*/0, /*weight=*/3, /*outgoing_edges=*/{}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_EQ(graph.NumNodes(), 1);
  EXPECT_EQ(graph.NumEdges(), 0);

  EXPECT_TRUE(graph.ContainsNode(0));
  ASSERT_OK_AND_ASSIGN(auto node, graph.ImmutableNode(0));
  EXPECT_EQ(node->CurrentNodeId(), 0);
  EXPECT_EQ(node->CurrentClusterId(), 0);
  EXPECT_EQ(node->ClusterSize(), 3);
  EXPECT_TRUE(node->IsActive());
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());

  EXPECT_THAT(graph.Nodes(), UnorderedElementsAre(0));

  EXPECT_OK(graph.RemoveNode(0));
  EXPECT_EQ(graph.NumNodes(), 0);
  EXPECT_EQ(graph.NumEdges(), 0);
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());
  EXPECT_THAT(graph.Nodes(), IsEmpty());
}

TEST(ConstructionTest, AddTwoNodes) {
  DynamicClusteredGraph graph;
  // Add two nodes 0 and 2 with an edge between them.

  AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                              /*outgoing_edges=*/{{2, 2}}};
  AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                              /*outgoing_edges=*/{{0, 2}}};

  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(graph.NumEdges(), 2);

  EXPECT_TRUE(graph.ContainsNode(0));
  EXPECT_TRUE(graph.ContainsNode(2));
  EXPECT_FALSE(graph.ContainsNode(1));
  EXPECT_THAT(graph.Nodes(), UnorderedElementsAre(0, 2));

  ASSERT_OK_AND_ASSIGN(auto node_2, graph.ImmutableNode(2));
  EXPECT_EQ(node_2->CurrentNodeId(), 2);
  EXPECT_EQ(node_2->CurrentClusterId(), 2);
  EXPECT_EQ(node_2->ClusterSize(), 2);
  EXPECT_TRUE(node_2->IsActive());

  std::vector<std::pair<NodeId, double>> neighbors;
  auto add_f = [&](const gbbs::uintE& v, const double wgh) {
    neighbors.push_back({v, wgh});
    return false;
  };
  node_2->IterateUntil(add_f);

  EXPECT_THAT(neighbors, UnorderedElementsAre(Pair(0, 1)));

  ASSERT_OK_AND_ASSIGN(auto node_0, graph.ImmutableNode(0));
  EXPECT_EQ(node_0->CurrentNodeId(), 0);
  EXPECT_EQ(node_0->CurrentClusterId(), 0);
  EXPECT_EQ(node_0->ClusterSize(), 1);
  EXPECT_TRUE(node_0->IsActive());

  neighbors.clear();
  node_0->IterateUntil(add_f);

  EXPECT_THAT(neighbors, UnorderedElementsAre(Pair(2, 1)));

  EXPECT_OK(graph.RemoveNode(0));
  EXPECT_EQ(graph.NumNodes(), 1);
  EXPECT_EQ(graph.NumEdges(), 0);
  EXPECT_THAT(graph.Nodes(), UnorderedElementsAre(2));

  neighbors.clear();
  ASSERT_OK_AND_ASSIGN(node_2, graph.ImmutableNode(2));
  node_2->IterateUntil(add_f);
  EXPECT_THAT(neighbors, IsEmpty());
}

TEST(ConstructionTest, AddTwice) {
  DynamicClusteredGraph graph(0.5);

  AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                              /*outgoing_edges=*/{{2, 2}}};
  AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                              /*outgoing_edges=*/{{0, 2}}};
  AdjacencyList adj_list_3 = {/*id=*/3, /*weight=*/2,
                              /*outgoing_edges=*/{{2, 12}}};

  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_EQ(graph.NumNodes(), 2);
  EXPECT_EQ(graph.NumEdges(), 2);
  EXPECT_THAT(graph.NumHeavyEdges(), 1);
  EXPECT_TRUE(graph.HasHeavyEdges());

  ASSERT_OK_AND_ASSIGN(auto node_2, graph.ImmutableNode(2));
  EXPECT_TRUE(node_2->IsActive());

  EXPECT_OK(graph.AddNodes({adj_list_3}));
  EXPECT_EQ(graph.NumNodes(), 3);
  EXPECT_EQ(graph.NumEdges(), 4);
  ASSERT_OK_AND_ASSIGN(node_2, graph.ImmutableNode(2));
  EXPECT_TRUE(node_2->IsActive());
  EXPECT_THAT(graph.NumHeavyEdges(), 2);
  EXPECT_TRUE(graph.HasHeavyEdges());
}

TEST(ConstructionTest, AddManyNodesStarGraph) {
  DynamicClusteredGraph graph(0);
  std::vector<std::pair<NodeId, double>> outgoing_edges;
  for (int i = 1; i < 100; ++i) {
    outgoing_edges.push_back({i, 1});
  }

  AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1, outgoing_edges};
  std::vector<AdjacencyList> adj_list_vec = {adj_list_0};

  for (int i = 1; i < 100; ++i) {
    adj_list_vec.push_back({/*id=*/i, /*weight=*/1, {{0, 1}}});
  }

  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_EQ(198, graph.NumEdges());
  EXPECT_THAT(graph.NumHeavyEdges(), 99);
  EXPECT_TRUE(graph.HasHeavyEdges());
}

TEST(ConstructionTest, AddManyNodesStarGraph2) {
  DynamicClusteredGraph graph(0);

  EXPECT_OK(graph.AddNodes({{/*id=*/0, /*weight=*/1, {}}}));

  for (int i = 1; i < 100; ++i) {
    EXPECT_OK(graph.AddNodes({{/*id=*/i, /*weight=*/1, {{0, 1}}}}));
  }
  EXPECT_EQ(198, graph.NumEdges());
  EXPECT_THAT(graph.NumHeavyEdges(), 99);
  EXPECT_TRUE(graph.HasHeavyEdges());
}

bool IsActive(double node_weight) { return node_weight >= 0; }
absl::flat_hash_map<NodeId, NodeId> GetReverseMap(
    absl::Span<const NodeId> mapping) {
  auto node_map_rev = absl::flat_hash_map<NodeId, NodeId>();
  for (NodeId j = 0; j < mapping.size(); ++j) {
    auto v = mapping[j];
    node_map_rev[v] = j;
  }
  return node_map_rev;
}

TEST(CreateSubgraphTest, OneNode) {
  DynamicClusteredGraph graph(0);
  // Add a node with no neighbor and weight 3. Node id is 0.
  const AdjacencyList adj_list = {/*id=*/0, /*weight=*/3,
                                  /*outgoing_edges=*/{}};
  const std::vector<AdjacencyList> adj_list_vec = {adj_list};
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  absl::flat_hash_map<NodeId, NodeId> partition_map;
  partition_map[0] = 0;

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph({0}, partition_map));

  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;

  // Node 0 is inactive because it has no edge, so also no heavy edge.
  EXPECT_THAT(mapping, ElementsAre(0));
  EXPECT_EQ(0, num_active_nodes);
  EXPECT_EQ(partition_graph->NumNodes(), 1);
  EXPECT_FALSE(IsActive(partition_graph->NodeWeight(0)));
  EXPECT_THAT(partition_graph->Neighbors(0), IsEmpty());
  EXPECT_THAT(ignored_nodes, ElementsAre(0));
}

TEST(CreateSubgraphTest, TwoNodesOnePartition) {
  // Create a graph with edge 0 -- 2.
  DynamicClusteredGraph graph(0);
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  absl::flat_hash_map<NodeId, NodeId> partition_map;
  partition_map[0] = 0;
  partition_map[2] = 0;

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph({0}, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  // Since 0 and 2 are both active, they can be in any order.
  EXPECT_THAT(mapping, UnorderedElementsAre(0, 2));
  EXPECT_EQ(num_active_nodes, 2);
  EXPECT_EQ(partition_graph->NumNodes(), 2);
  EXPECT_TRUE(IsActive(partition_graph->NodeWeight(0)));
  EXPECT_TRUE(IsActive(partition_graph->NodeWeight(1)));
  EXPECT_THAT(ignored_nodes, IsEmpty());

  EXPECT_THAT(partition_graph->Neighbors(0), ElementsAre(Pair(1, 1.0)));
}

TEST(CreateSubgraphTest, TwoNodesTwoPartitions) {
  // Create a graph with edge 0 - 2.
  DynamicClusteredGraph graph(0);
  const AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                                    /*outgoing_edges=*/{{2, 2}}};
  const AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                                    /*outgoing_edges=*/{{0, 2}}};

  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};
  EXPECT_OK(graph.AddNodes(adj_list_vec));

  absl::flat_hash_map<NodeId, NodeId> partition_map;
  partition_map[0] = 0;
  partition_map[2] = 2;

  // In the resulting `graph`. We have active node 0 (mapping to 2) and it has
  // an edge to node 1 (mapping to 0).
  // Node 0 maps to original node 2 because active nodes have smaller ids.
  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph({2}, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  EXPECT_THAT(mapping, ElementsAre(2, 0));
  EXPECT_EQ(1, num_active_nodes);
  EXPECT_EQ(partition_graph->NumNodes(), 2);
  EXPECT_TRUE(IsActive(partition_graph->NodeWeight(0)));
  EXPECT_FALSE(IsActive(partition_graph->NodeWeight(1)));
  EXPECT_THAT(ignored_nodes, IsEmpty());

  EXPECT_THAT(partition_graph->Neighbors(0), ElementsAre(Pair(1, 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(1), ElementsAre(Pair(0, 1.0)));
}

DynamicClusteredGraph GetLargeTestGraph(
    absl::flat_hash_map<NodeId, NodeId>& partition_map) {
  //            +----+
  //            | 10 |
  //            +----+
  //              |
  // +----+     +----+
  // | 12 | --- | 11 | -+
  // +----+     +----+  |
  //            +----+  |
  //            | 19 |  |
  //            +----+  |
  //              |     |
  //            +----+  |
  //            | 18 |  |
  //            +----+  |
  //              |     |
  // +----+     +----+  |
  // | 17 | --- | 16 |  |
  // +----+     +----+  |
  //              |     |
  //            +----+  |
  //            | 13 | -+
  //            +----+
  DynamicClusteredGraph graph(0);
  std::vector<AdjacencyList> adj_list_vec;

  // The partition we want to extract. We call this the "center" node.
  const AdjacencyList adj_list_13 = {/*id=*/13, /*weight=*/1,
                                     /*outgoing_edges=*/{{16, 1}, {11, 1}}};
  partition_map[13] = 13;
  adj_list_vec.push_back(adj_list_13);

  // Center node's neighbor out-of-partition.
  const AdjacencyList adj_list_11 = {
      /*id=*/11, /*weight=*/1,
      /*outgoing_edges=*/{{10, 1}, {12, 1}, {13, 1}}};
  partition_map[11] = 11;
  adj_list_vec.push_back(adj_list_11);

  // Center node's neighbor in the partition.
  const AdjacencyList adj_list_16 = {
      /*id=*/16, /*weight=*/1,
      /*outgoing_edges=*/{{13, 1}, {18, 1}, {17, 1}}};
  partition_map[16] = 13;
  adj_list_vec.push_back(adj_list_16);

  // Center node's two hop neighbors. They are neighboring an out-of-partition
  // node.
  const AdjacencyList adj_list_10 = {/*id=*/10, /*weight=*/1,
                                     /*outgoing_edges=*/{{11, 1}}};
  partition_map[10] = 11;
  adj_list_vec.push_back(adj_list_10);
  const AdjacencyList adj_list_12 = {/*id=*/12, /*weight=*/1,
                                     /*outgoing_edges=*/{{11, 1}}};
  partition_map[12] = 11;
  adj_list_vec.push_back(adj_list_12);

  // Center node's two hop neighbors. They are neighboring an in-partition
  // node.
  const AdjacencyList adj_list_17 = {/*id=*/17, /*weight=*/1,
                                     /*outgoing_edges=*/{{16, 1}}};
  partition_map[17] = 17;
  adj_list_vec.push_back(adj_list_17);
  const AdjacencyList adj_list_18 = {/*id=*/18, /*weight=*/1,
                                     /*outgoing_edges=*/{{16, 1}, {19, 1}}};
  partition_map[18] = 18;
  adj_list_vec.push_back(adj_list_18);

  // Center node's three hop neighbor.
  const AdjacencyList adj_list_19 = {/*id=*/19, /*weight=*/1,
                                     /*outgoing_edges=*/{{18, 1}}};
  partition_map[19] = 19;
  adj_list_vec.push_back(adj_list_19);

  EXPECT_OK(graph.AddNodes(adj_list_vec));
  return graph;
}

TEST(CreateSubgraphTest, MultiplePartitionsSingleReturn) {
  absl::flat_hash_map<NodeId, NodeId> partition_map;
  DynamicClusteredGraph graph = GetLargeTestGraph(partition_map);

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph({13}, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  const auto& partition_graph_num_nodes = partition_graph->NumNodes();
  auto active_nodes = std::vector<NodeId>();
  auto inactive_nodes = std::vector<NodeId>();
  EXPECT_EQ(2, num_active_nodes);
  EXPECT_EQ(partition_graph_num_nodes, 5);
  for (int i = 0; i < num_active_nodes; ++i) {
    EXPECT_TRUE(IsActive(partition_graph->NodeWeight(i)));
    active_nodes.push_back(mapping[i]);
  }
  for (int i = num_active_nodes; i < partition_graph_num_nodes; ++i) {
    EXPECT_FALSE(IsActive(partition_graph->NodeWeight(i)));
    inactive_nodes.push_back(mapping[i]);
  }

  EXPECT_THAT(active_nodes, UnorderedElementsAre(13, 16));
  EXPECT_THAT(inactive_nodes, UnorderedElementsAre(11, 17, 18));
  EXPECT_THAT(ignored_nodes, IsEmpty());

  // Check subgraph's edges. We need a reverse mapping from original graph
  // to the subgraph's node id.

  //            +----+
  //            | 17 |
  //            +----+
  //              |
  // +----+     +----+
  // | 18 | --- | 16 |
  // +----+     +----+
  //              |
  //            +----+
  //            | 13 |
  //            +----+
  //              |
  //            +----+
  //            | 11 |
  //            +----+

  auto node_map_rev = GetReverseMap(mapping);

  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[13]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0),
                                   Pair(node_map_rev[11], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[16]),
              UnorderedElementsAre(Pair(node_map_rev[13], 1.0),
                                   Pair(node_map_rev[17], 1.0),
                                   Pair(node_map_rev[18], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[17]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[18]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[11]),
              UnorderedElementsAre(Pair(node_map_rev[13], 1.0)));
}

TEST(CreateSubgraphTest, MultiplePartitionsMultipleReturn) {
  //            +----+
  //            | 10 |
  //            +----+
  //              |
  // +----+     +----+
  // | 12 | --- | 11 | -+
  // +----+     +----+  |
  //            +----+  |
  //            | 19 |  |
  //            +----+  |
  //              |     |
  //            +----+  |
  //            | 18 |  |
  //            +----+  |
  //              |     |
  // +----+     +----+  |
  // | 17 | --- | 16 |  |
  // +----+     +----+  |
  //              |     |
  //            +----+  |
  //            | 13 | -+
  //            +----+
  absl::flat_hash_map<NodeId, NodeId> partition_map;
  DynamicClusteredGraph graph = GetLargeTestGraph(partition_map);

  const absl::flat_hash_set<NodeId> node_ids{11, 13};
  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph(node_ids, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  const auto& partition_graph_num_nodes = partition_graph->NumNodes();
  auto active_nodes = std::vector<NodeId>();
  auto inactive_nodes = std::vector<NodeId>();
  EXPECT_EQ(5, num_active_nodes);
  EXPECT_EQ(partition_graph_num_nodes, 7);
  for (int i = 0; i < num_active_nodes; ++i) {
    EXPECT_TRUE(IsActive(partition_graph->NodeWeight(i)));
    active_nodes.push_back(mapping[i]);
  }
  for (int i = num_active_nodes; i < partition_graph_num_nodes; ++i) {
    EXPECT_FALSE(IsActive(partition_graph->NodeWeight(i)));
    inactive_nodes.push_back(mapping[i]);
  }

  EXPECT_THAT(active_nodes, UnorderedElementsAre(10, 11, 12, 13, 16));
  EXPECT_THAT(inactive_nodes, UnorderedElementsAre(17, 18));
  EXPECT_THAT(ignored_nodes, IsEmpty());

  // Check subgraph's edges. We need a reverse mapping from original graph
  // to the subgraph's node id.
  //            +----+
  //            | 10 |
  //            +----+
  //              |
  // +----+     +----+
  // | 12 | --- | 11 | -+
  // +----+     +----+  |
  //            +----+  |
  //            | 18 |  |
  //            +----+  |
  //              |     |
  // +----+     +----+  |
  // | 17 | --- | 16 |  |
  // +----+     +----+  |
  //              |     |
  //            +----+  |
  //            | 13 | -+
  //            +----+

  auto node_map_rev = GetReverseMap(mapping);

  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[13]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0),
                                   Pair(node_map_rev[11], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[16]),
              UnorderedElementsAre(Pair(node_map_rev[13], 1.0),
                                   Pair(node_map_rev[17], 1.0),
                                   Pair(node_map_rev[18], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[17]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[18]),
              UnorderedElementsAre(Pair(node_map_rev[16], 1.0)));
  EXPECT_THAT(partition_graph->Neighbors(node_map_rev[11]),
              UnorderedElementsAre(Pair(node_map_rev[13], 1.0),
                                   Pair(node_map_rev[10], 1.0),
                                   Pair(node_map_rev[12], 1.0)));
}

TEST(NeighborsTest, LargeGraph) {
  absl::flat_hash_map<NodeId, NodeId> partition_map;
  const DynamicClusteredGraph graph = GetLargeTestGraph(partition_map);

  ASSERT_OK_AND_ASSIGN(auto nodes, graph.Neighbors({11, 13}));

  EXPECT_THAT(nodes, UnorderedElementsAre(10, 12, 16, 11, 13));
  EXPECT_EQ(graph.NumNodes(), 8);
}

TEST(ConstructionTest, LargeGraphNumEdges) {
  absl::flat_hash_map<NodeId, NodeId> partition_map;
  DynamicClusteredGraph graph = GetLargeTestGraph(partition_map);
  int num_nodes = 8;
  EXPECT_EQ(graph.NumNodes(), num_nodes);
  EXPECT_EQ(graph.NumEdges(), 14);
  const auto edges = std::vector<int>{12, 8, 8, 6, 2, 2, 0, 0};
  int edge_num_id = 0;
  for (const NodeId i : {10, 11, 12, 13, 16, 17, 18, 19}) {
    num_nodes--;
    ASSERT_OK(graph.RemoveNode(i));
    EXPECT_EQ(graph.NumNodes(), num_nodes);
    EXPECT_EQ(graph.NumEdges(), edges[edge_num_id]);
    edge_num_id++;
  }
}

TEST(HeavyEdgeTest, AddTwoNodes) {
  // Add two nodes 0 and 2 with an edge between them.

  AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                              /*outgoing_edges=*/{{2, 2}}};
  AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                              /*outgoing_edges=*/{{0, 2}}};
  const absl::flat_hash_set<NodeId> node_ids{0};
  const absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 0}, {2, 2}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};

  // Weight threshold = 2.
  DynamicClusteredGraph graph(2);
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph(node_ids, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  EXPECT_EQ(num_active_nodes, 0);
  // Only 1 node ignored, because the other node is not in the partition.
  EXPECT_THAT(ignored_nodes, UnorderedElementsAre(0));

  EXPECT_OK(graph.RemoveNode(0));
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());

  // Weight threshold = 1.
  DynamicClusteredGraph graph2(1);
  EXPECT_OK(graph2.AddNodes(adj_list_vec));
  EXPECT_THAT(graph2.NumHeavyEdges(), 1);
  EXPECT_TRUE(graph2.HasHeavyEdges());
  EXPECT_THAT(graph2.HasHeavyEdges(0), IsOkAndHolds(true));
  EXPECT_THAT(graph2.HasHeavyEdges(2), IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(const auto& result2,
                       graph2.CreateSubgraph(node_ids, partition_map));
  const auto& [partition_graph2, mapping2, num_active_nodes2, ignored_nodes2] =
      *result2;
  EXPECT_EQ(num_active_nodes2, 1);
  EXPECT_THAT(ignored_nodes2, IsEmpty());

  EXPECT_OK(graph2.RemoveNode(0));
  EXPECT_THAT(graph2.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph2.HasHeavyEdges());
}

TEST(HeavyEdgeTest, IgnoreBlueNode) {
  // Add two nodes 0 and 2 with an edge between them.
  AdjacencyList adj_list_0 = {/*id=*/0, /*weight=*/1,
                              /*outgoing_edges=*/{{2, 2}}};
  AdjacencyList adj_list_2 = {/*id=*/2, /*weight=*/2,
                              /*outgoing_edges=*/{{0, 2}}};
  const absl::flat_hash_set<NodeId> node_ids{0};
  const absl::flat_hash_map<NodeId, NodeId> partition_map{{0, 0}, {2, 0}};
  std::vector<AdjacencyList> adj_list_vec = {adj_list_0, adj_list_2};

  // Weight threshold = 2.
  DynamicClusteredGraph graph(2);
  EXPECT_OK(graph.AddNodes(adj_list_vec));
  EXPECT_THAT(graph.NumHeavyEdges(), 0);
  EXPECT_FALSE(graph.HasHeavyEdges());

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       graph.CreateSubgraph(node_ids, partition_map));
  const auto& [partition_graph, mapping, num_active_nodes, ignored_nodes] =
      *result;
  EXPECT_EQ(num_active_nodes, 0);
  EXPECT_THAT(ignored_nodes, UnorderedElementsAre(0, 2));
}
}  // namespace
}  // namespace graph_mining::in_memory
