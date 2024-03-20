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

#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/status_macros.h"
#include "utils/math.h"

namespace graph_mining {
namespace in_memory {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

using NodeId = InMemoryClusterer::NodeId;
using Cluster = std::initializer_list<NodeId>;
using Dendrogram = graph_mining::in_memory::Dendrogram;
using DendrogramNode = graph_mining::in_memory::DendrogramNode;
using Merge = std::tuple<NodeId, NodeId, double>;

MATCHER_P2(IsNode, parent_id, merge_similarity, "") {
  return parent_id == arg.parent_id &&
         AlmostEquals(merge_similarity, arg.merge_similarity);
}
const auto IsEmptyNode =
    IsNode(Dendrogram::kNoParentId, std::numeric_limits<double>::infinity());
const auto kInf = std::numeric_limits<double>::infinity();

absl::StatusOr<std::pair<Dendrogram, InMemoryClusterer::Clustering>>
ApproximateSubgraphHacInternal(std::unique_ptr<SimpleUndirectedGraph> graph,
                               double epsilon) {
  std::vector<double> min_merge_similarities(
      graph->NumNodes(), std::numeric_limits<double>::infinity());
  ASSIGN_OR_RETURN(
      auto subgraph_hac_result,
      ApproximateSubgraphHac(std::move(graph),
                             std::move(min_merge_similarities), epsilon));
  return std::make_pair(std::move(subgraph_hac_result.dendrogram),
                        std::move(subgraph_hac_result.clustering));
}

absl::StatusOr<std::pair<Dendrogram, InMemoryClusterer::Clustering>>
ApproximateSubgraphHacInternal(std::unique_ptr<SimpleUndirectedGraph> graph,
                               std::vector<double> min_merge_similarities,
                               double epsilon) {
  ASSIGN_OR_RETURN(
      auto subgraph_hac_result,
      ApproximateSubgraphHac(std::move(graph),
                             std::move(min_merge_similarities), epsilon));
  return std::make_pair(std::move(subgraph_hac_result.dendrogram),
                        std::move(subgraph_hac_result.clustering));
}

absl::StatusOr<SubgraphHacResults> ApproximateSubgraphHacReturnGraphInternal(
    std::unique_ptr<SimpleUndirectedGraph> graph, double epsilon) {
  std::vector<double> min_merge_similarities(
      graph->NumNodes(), std::numeric_limits<double>::infinity());
  return ApproximateSubgraphHac(std::move(graph),
                                std::move(min_merge_similarities), epsilon);
}

std::vector<std::int64_t> NodeWeights(const ContractedGraph* graph) {
  std::vector<std::int64_t> weights;
  for (NodeId i = 0; i < graph->NumNodes(); ++i) {
    weights.push_back(graph->NodeWeight(i));
  }
  return weights;
}

TEST(ApproximateSubgraphHacTest, ClustersIsolatedGraph) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  graph->SetNumNodes(3);
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, -1);
  auto graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  // Test the flat clustering.
  Dendrogram dendrogram(0);
  InMemoryClusterer::Clustering clustering;
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(std::move(graph), 0.1));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0}, {1}}));

  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 3);

  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      ApproximateSubgraphHacReturnGraphInternal(std::move(graph_copy), 0.1));
  const auto& [merges, clustering2, dendrogram2, contracted_graph] =
      subgraph_hac_result;
  EXPECT_THAT(clustering2, UnorderedElementsAreArray<Cluster>({{0}, {1}}));

  EXPECT_THAT(dendrogram2.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram2.NumClusteredNodes(), 3);
  EXPECT_THAT(merges, IsEmpty());
  EXPECT_EQ(contracted_graph->NumNodes(), 3);
  EXPECT_THAT(NodeWeights(contracted_graph.get()), ElementsAre(1, 1, -1));
  EXPECT_THAT(contracted_graph->Neighbors(0), IsEmpty());
  EXPECT_THAT(contracted_graph->Neighbors(1), IsEmpty());
  EXPECT_THAT(contracted_graph->Neighbors(2), IsEmpty());
}

TEST(ApproximateSubgraphHacTest, ClusterTriangleWithLowWeightSatellite) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  graph->SetNumNodes(3);
  ASSERT_OK(graph->AddEdge(0, 1, 1.0));
  ASSERT_OK(graph->AddEdge(0, 2, 0.5));
  ASSERT_OK(graph->AddEdge(1, 2, 0.25));
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, -1);

  auto graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  Dendrogram dendrogram(0);
  InMemoryClusterer::Clustering clustering;
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.1));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0, 1}}));

  EXPECT_THAT(dendrogram.Nodes(), ElementsAre(IsNode(3, 1.0), IsNode(3, 1.0),
                                              IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 3);

  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      ApproximateSubgraphHacReturnGraphInternal(std::move(graph_copy), 0.1));
  const auto& [merges, clustering2, dendrogram2, contracted_graph] =
      subgraph_hac_result;
  EXPECT_THAT(merges, ElementsAreArray<Merge>({{0, 1, 1}}));
  EXPECT_THAT(clustering2, UnorderedElementsAreArray<Cluster>({{0, 1}}));

  EXPECT_THAT(dendrogram2.Nodes(), ElementsAre(IsNode(3, 1.0), IsNode(3, 1.0),
                                               IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram2.NumClusteredNodes(), 3);
  EXPECT_EQ(contracted_graph->NumNodes(), 3);
  EXPECT_THAT(NodeWeights(contracted_graph.get()), ElementsAre(-1, 2, -1));
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(1),
              UnorderedElementsAre(Pair(0.75, 2)));
  EXPECT_THAT(contracted_graph->Neighbors(1),
              UnorderedElementsAre(Pair(0.75 / 2, 2)));
  EXPECT_THAT(contracted_graph->Neighbors(0), IsEmpty());
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(2), IsEmpty());

  // Supply an artificially low min-merge value and check that the algorithm
  // does not merge any edges.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  std::vector<double> min_merge_similarities = {
      0.5, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  // The error score for the first edge is (0, 1) -> 1.0 / 0.5 = 2
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 0.1));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0}, {1}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 3);

  // One merge becomes possible again for eps=1.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 1));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0, 1}}));
  EXPECT_THAT(dendrogram.Nodes(), ElementsAre(IsNode(3, 1.0), IsNode(3, 1.0),
                                              IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 3);
}

TEST(ApproximateSubgraphHacTest, ClusterTriangleWithHighWeightSatellite) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  graph->SetNumNodes(3);
  ASSERT_OK(graph->AddEdge(0, 1, 1.0));
  ASSERT_OK(graph->AddEdge(0, 2, 0.5));
  ASSERT_OK(graph->AddEdge(1, 2, 1.25));
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, -1);
  Dendrogram dendrogram(0);
  InMemoryClusterer::Clustering clustering;
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(std::move(graph), 0.1));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0}, {1}}));

  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 3);
}

TEST(ApproximateSubgraphHacTest, ClusterChainWithSatellite) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  graph->SetNumNodes(5);
  ASSERT_OK(graph->AddEdge(0, 1, 1.0));
  ASSERT_OK(graph->AddEdge(1, 2, 1.0));
  ASSERT_OK(graph->AddEdge(2, 3, 0.8));
  ASSERT_OK(graph->AddEdge(3, 4, 0.4));
  graph->SetNodeWeight(0, 1);
  graph->SetNodeWeight(1, -1);
  graph->SetNodeWeight(2, 1);
  graph->SetNodeWeight(3, 1);
  graph->SetNodeWeight(4, 1);

  auto graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  Dendrogram dendrogram(0);
  InMemoryClusterer::Clustering clustering;
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.1));

  // No merges.
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{0}, {2}, {3}, {4}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode, IsEmptyNode,
                          IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);

  // Only 2 and 3 are merged with eps=0.25
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.4));
  // Only 2 and 3 are merged.
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{0}, {2, 3}, {4}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsNode(5, 0.8),
                          IsNode(5, 0.8), IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);

  // Only 2 and 3 are not merged with the given min_merge_similarities which
  // are good according to the notion in Lemma 2 of go/terahac-paper. (we have
  // that BestEdge(u) / min_merge_similarity(u) <= 1+epsilon.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  std::vector<double> min_merge_similarities = {kInf, kInf, 0.8, 0.64, kInf};
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy),
                                     min_merge_similarities, 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{0}, {2}, {3}, {4}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsEmptyNode, IsEmptyNode,
                          IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);

  // If 3's min_merge_similarity is also infinity, {2, 3} can still
  // successfully merge.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, 0.8, kInf, kInf};
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy),
                                     min_merge_similarities, 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{0}, {2, 3}, {4}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsNode(5, 0.8),
                          IsNode(5, 0.8), IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);

  // {2,3} are merged into 4.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 1.5));
  EXPECT_THAT(clustering, UnorderedElementsAreArray<Cluster>({{0}, {2, 3, 4}}));
  EXPECT_THAT(
      dendrogram.Nodes(),
      ElementsAre(IsEmptyNode, IsEmptyNode, IsNode(5, 0.8), IsNode(5, 0.8),
                  IsNode(6, 0.2), IsNode(6, 0.2), IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);

  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      auto subgraph_hac_result,
      ApproximateSubgraphHacReturnGraphInternal(std::move(graph_copy), 1.5));
  const auto& [merges, clustering2, dendrogram2, contracted_graph] =
      subgraph_hac_result;
  EXPECT_THAT(merges, ElementsAreArray<Merge>({{2, 3, 0.8}, {4, 5, 0.2}}));
  EXPECT_THAT(clustering2,
              UnorderedElementsAreArray<Cluster>({{0}, {2, 3, 4}}));
  EXPECT_THAT(
      dendrogram2.Nodes(),
      ElementsAre(IsEmptyNode, IsEmptyNode, IsNode(5, 0.8), IsNode(5, 0.8),
                  IsNode(6, 0.2), IsNode(6, 0.2), IsEmptyNode));
  EXPECT_EQ(dendrogram2.NumClusteredNodes(), 5);
  EXPECT_EQ(contracted_graph->NumNodes(), 5);
  EXPECT_THAT(NodeWeights(contracted_graph.get()),
              ElementsAre(1, -1, -1, 3, -1));
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(0),
              UnorderedElementsAre(Pair(1, 1)));
  EXPECT_THAT(contracted_graph->Neighbors(0), UnorderedElementsAre(Pair(1, 1)));
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(1), IsEmpty());
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(2), IsEmpty());
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(4), IsEmpty());
  EXPECT_THAT(contracted_graph->UnnormalizedNeighborsSimilarity(3),
              UnorderedElementsAre(Pair(1.0, 1)));
  EXPECT_THAT(contracted_graph->Neighbors(3),
              UnorderedElementsAre(Pair(DoubleEq(1.0 / 3), 1)));

  // {2,3} not merged into 4 when 4's min_merge_similarity is too low.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, kInf, kInf, 0.1};
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 2));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{0}, {2, 3}, {4}}));
  EXPECT_THAT(dendrogram.Nodes(),
              ElementsAre(IsEmptyNode, IsEmptyNode, IsNode(5, 0.8),
                          IsNode(5, 0.8), IsEmptyNode, IsEmptyNode));
  EXPECT_EQ(dendrogram.NumClusteredNodes(), 5);
}

TEST(ApproximateSubgraphHacTest, ClusterTwoTrianglesAndSatellites) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  graph->SetNumNodes(8);
  ASSERT_OK(graph->AddEdge(0, 1, 1));
  ASSERT_OK(graph->AddEdge(1, 2, 4.3));
  ASSERT_OK(graph->AddEdge(2, 3, 4.2));
  ASSERT_OK(graph->AddEdge(3, 1, 4.1));
  ASSERT_OK(graph->AddEdge(2, 4, 3));
  ASSERT_OK(graph->AddEdge(4, 5, 8.3));
  ASSERT_OK(graph->AddEdge(5, 6, 8.2));
  ASSERT_OK(graph->AddEdge(6, 4, 8.1));
  ASSERT_OK(graph->AddEdge(6, 7, 10));
  graph->SetNodeWeight(0, -1);  // inactive
  graph->SetNodeWeight(1, 1);
  graph->SetNodeWeight(2, 1);
  graph->SetNodeWeight(3, 1);
  graph->SetNodeWeight(4, 1);
  graph->SetNodeWeight(5, 1);
  graph->SetNodeWeight(6, 1);
  graph->SetNodeWeight(7, -1);  // inactive

  // For this test, we only check the clusterings returned by the algorithm,
  // ignoring the dendrogram values, since the order in which nodes can be
  // merged depends on tie-breaking of equal values done by the max-weight
  // finder. The reason is that there are multiple reciprocal best edges, and
  // once we sort by the "1/error" scores, the order in which we process the
  // reciprocal best edges depends on the implementation of the priority
  // queue, which may change and make the parnet_ids in the dendrogram
  // brittle. Note that this issue does not arise in the previous tests, since
  // there is only one reciprocal best edge available at any point in the
  // algorithm.
  auto graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  Dendrogram dendrogram(0);
  InMemoryClusterer::Clustering clustering;
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.1));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5}, {6}}));

  // Setting 3's min_merge_similarity to BestEdge(3)/(1+eps) prevents it from
  // merging.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  std::vector<double> min_merge_similarities = {kInf, kInf, kInf, 4.1 / 1.1,
                                                kInf, kInf, kInf, kInf};
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 0.1));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2}, {3}, {4, 5}, {6}}));

  // Setting 3's min_merge_similarity to 4.15/(1+eps) allows it to merge (4.15
  // is its edge weight to {1, 2}).
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  min_merge_similarities = {kInf, kInf, kInf, 4.15 / 1.1,
                            kInf, kInf, kInf, kInf};
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 0.1));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5}, {6}}));

  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.2));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5}, {6}}));

  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5, 6}}));

  // {4, 5}, {6} do not merge if 6's min_merge value is too low.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, kInf,       kInf,
                            kInf, kInf, 8.1 / 1.25, kInf};
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy),
                                     min_merge_similarities, 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5}, {6}}));

  // {4, 5}, {6} still cannot merge if the min_merge_similarity is w({4,5},
  // {6}) / 1+eps.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, kInf,        kInf,
                            kInf, kInf, 8.15 / 1.25, kInf};
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy),
                                     min_merge_similarities, 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5}, {6}}));

  // {4, 5}, {6} can merge if the min_merge_similarity is BestEdge(6) / 1 +
  // eps.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, kInf,      kInf,
                            kInf, kInf, 10 / 1.25, kInf};
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy),
                                     min_merge_similarities, 0.25));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3}, {4, 5, 6}}));

  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  ASSERT_OK_AND_ASSIGN(
      std::tie(dendrogram, clustering),
      ApproximateSubgraphHacInternal(std::move(graph_copy), 9.1));
  // Middle (bridge) edge has weight 1/3, and edge to right triangle's
  // inactive neighbor has weight 10/3 (ratio = 10 = 1+eps). So for eps > 9
  // will cluster the triangles together.
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3, 4, 5, 6}}));

  // min_merge_similarities that previously prevented merging {1,2} and {3}
  // have no effect for large epsilon.
  graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ASSERT_OK(CopyGraph(*graph, graph_copy.get()));
  min_merge_similarities = {kInf, kInf, kInf, 4.10 / 1.1,
                            kInf, kInf, kInf, kInf};
  ASSERT_OK_AND_ASSIGN(std::tie(dendrogram, clustering),
                       ApproximateSubgraphHacInternal(
                           std::move(graph_copy), min_merge_similarities, 9.1));
  EXPECT_THAT(clustering,
              UnorderedElementsAreArray<Cluster>({{1, 2, 3, 4, 5, 6}}));
}

}  // namespace
}  // namespace in_memory
}  // namespace graph_mining
