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

#include "in_memory/clustering/dynamic/hac/adjacency_lists.h"

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {
namespace {

using ::testing::FieldsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

using graph_mining::in_memory::NodeId;

using ConvertToAdjListTest = ::testing::Test;

TEST_F(ConvertToAdjListTest, TriangleGraph) {
  auto graph = SimpleUndirectedGraph();
  graph.SetNumNodes(3);
  ASSERT_OK(graph.AddEdge(0, 0, 1.0));
  ASSERT_OK(graph.AddEdge(0, 1, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 0.5));
  ASSERT_OK(graph.AddEdge(1, 2, 0.25));
  graph.SetNodeWeight(0, 1);
  graph.SetNodeWeight(1, 1);
  graph.SetNodeWeight(2, -1);
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list, ConvertToAdjList(graph));
  EXPECT_THAT(
      adjacency_list,
      UnorderedElementsAre(
          FieldsAre(0, 1, UnorderedElementsAre(Pair(1, 1), Pair(2, 0.5)),
                    std::nullopt),
          FieldsAre(1, 1, UnorderedElementsAre(Pair(0, 1), Pair(2, 0.25)),
                    std::nullopt),
          FieldsAre(2, -1, UnorderedElementsAre(Pair(1, 0.25), Pair(0, 0.5)),
                    std::nullopt)));
}

TEST_F(ConvertToAdjListTest, TriangleEdges) {
  const std::vector<std::pair<NodeId, NodeId>> edges{{0, 1}, {0, 2}, {1, 2}};
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list, ConvertToAdjList(edges));
  EXPECT_THAT(adjacency_list,
              UnorderedElementsAre(
                  FieldsAre(0, 1, UnorderedElementsAre(Pair(1, 1), Pair(2, 1)),
                            std::nullopt),
                  FieldsAre(1, 1, UnorderedElementsAre(Pair(0, 1), Pair(2, 1)),
                            std::nullopt),
                  FieldsAre(2, 1, UnorderedElementsAre(Pair(1, 1), Pair(0, 1)),
                            std::nullopt)));
}

TEST_F(ConvertToAdjListTest, TriangleWeightedEdges) {
  const std::vector<std::tuple<NodeId, NodeId, double>> edges_weighted{
      {0, 1, 1}, {0, 2., 0.5}, {1, 2, 0.25}};
  const absl::flat_hash_map<NodeId, NodeId> node_weights{
      {0, 5}, {1, 1}, {2, 1}};
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list,
                       ConvertToAdjList(edges_weighted, node_weights));
  EXPECT_THAT(
      adjacency_list,
      UnorderedElementsAre(
          FieldsAre(0, 5, UnorderedElementsAre(Pair(1, 1), Pair(2, 0.5)),
                    std::nullopt),
          FieldsAre(1, 1, UnorderedElementsAre(Pair(0, 1), Pair(2, 0.25)),
                    std::nullopt),
          FieldsAre(2, 1, UnorderedElementsAre(Pair(1, 0.25), Pair(0, 0.5)),
                    std::nullopt)));
}

TEST_F(ConvertToAdjListTest, WedgeGraph) {
  auto graph = SimpleUndirectedGraph();
  graph.SetNumNodes(3);
  ASSERT_OK(graph.AddEdge(0, 0, 1.0));
  ASSERT_OK(graph.AddEdge(0, 2, 0.5));
  ASSERT_OK(graph.AddEdge(1, 2, 0.25));
  graph.SetNodeWeight(0, 1);
  graph.SetNodeWeight(1, 1);
  graph.SetNodeWeight(2, -1);
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list, ConvertToAdjList(graph));
  EXPECT_THAT(
      adjacency_list,
      UnorderedElementsAre(
          FieldsAre(0, 1, UnorderedElementsAre(Pair(2, 0.5)), std::nullopt),
          FieldsAre(1, 1, UnorderedElementsAre(Pair(2, 0.25)), std::nullopt),
          FieldsAre(2, -1, UnorderedElementsAre(Pair(1, 0.25), Pair(0, 0.5)),
                    std::nullopt)));
}

TEST_F(ConvertToAdjListTest, WedgeEdges) {
  const std::vector<std::pair<NodeId, NodeId>> edges{{0, 2}, {1, 2}};
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list, ConvertToAdjList(edges));
  EXPECT_THAT(
      adjacency_list,
      UnorderedElementsAre(
          FieldsAre(0, 1, UnorderedElementsAre(Pair(2, 1)), std::nullopt),
          FieldsAre(1, 1, UnorderedElementsAre(Pair(2, 1)), std::nullopt),
          FieldsAre(2, 1, UnorderedElementsAre(Pair(1, 1), Pair(0, 1)),
                    std::nullopt)));
}

TEST_F(ConvertToAdjListTest, WedgeWeightedEdges) {
  const std::vector<std::tuple<NodeId, NodeId, double>> edges_weighted{
      {0, 2., 0.5}, {1, 2, 0.25}};
  const absl::flat_hash_map<NodeId, NodeId> node_weights{{0, 5}, {2, 1}};
  ASSERT_OK_AND_ASSIGN(const auto adjacency_list,
                       ConvertToAdjList(edges_weighted, node_weights));
  EXPECT_THAT(
      adjacency_list,
      UnorderedElementsAre(
          FieldsAre(0, 5, UnorderedElementsAre(Pair(2, 0.5)), std::nullopt),
          FieldsAre(1, 1, UnorderedElementsAre(Pair(2, 0.25)), std::nullopt),
          FieldsAre(2, 1, UnorderedElementsAre(Pair(1, 0.25), Pair(0, 0.5)),
                    std::nullopt)));
}

}  // namespace
}  // namespace graph_mining::in_memory
