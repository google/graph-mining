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

#include "in_memory/clustering/affinity/affinity.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/random/random.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "utils/parse_proto/parse_text_proto.h"
#include "src/farmhash.h"

namespace graph_mining::in_memory {
namespace {

using ::graph_mining::in_memory::CanonicalizeClustering;
using ::graph_mining::in_memory::ClustererConfig;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

using Cluster = std::initializer_list<InMemoryClusterer::NodeId>;

TEST(AffinityTest, NoEdges) {
  SimpleUndirectedGraph graph;
  graph.AddNode();
  graph.AddNode();
  graph.AddNode();
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK_AND_ASSIGN(auto clusters, clusterer->Cluster(ClustererConfig()));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0}, {1}, {2}}));
}

TEST(AffinityTest, NumIterations) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK_AND_ASSIGN(auto clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 0 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0}, {1}, {2}, {3}}));
  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 1 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(AffinityTest, Hierarchy) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));

  ASSERT_OK_AND_ASSIGN(auto clustering_hierarchy,
                       clusterer->HierarchicalFlatCluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  ASSERT_EQ(clustering_hierarchy.size(), 2);
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[0]),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
  EXPECT_THAT(CanonicalizeClustering(clustering_hierarchy[1]),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(AffinityTest, WeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK_AND_ASSIGN(
      auto clusters,
      clusterer->Cluster(PARSE_TEXT_PROTO(
          "affinity_clusterer_config { weight_threshold: 2.0 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}}));
  ASSERT_OK_AND_ASSIGN(
      clusters, clusterer->Cluster(PARSE_TEXT_PROTO(
                    "affinity_clusterer_config { weight_threshold: 2.1 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0}, {1}}));
}

TEST(AffinityTest, MultiWeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 10.0));
  ASSERT_OK(graph.AddEdge(2, 3, 10.0));
  ASSERT_OK(graph.AddEdge(0, 2, 4.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
  ASSERT_OK_AND_ASSIGN(auto clusters, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      edge_aggregation_function: MAX
      per_iteration_weight_thresholds: { thresholds: 10.0 thresholds: 4.0 }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));

  // Prevent a complete clustering in the second round with a too-high second
  // weight threshold.
  ASSERT_OK_AND_ASSIGN(clusters, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      edge_aggregation_function: MAX
      per_iteration_weight_thresholds: { thresholds: 10.0 thresholds: 4.1 }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));
}

TEST(AffinityTest, EdgeAggregationFunction) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 10.0));
  ASSERT_OK(graph.AddEdge(2, 3, 10.0));
  ASSERT_OK(graph.AddEdge(0, 2, 4.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));

  ASSERT_OK_AND_ASSIGN(auto clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 1.1 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: DEFAULT_AVERAGE "
                           "weight_threshold: 1.0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));

  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: MAX weight_threshold: "
                           "4.1 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { "
                           "edge_aggregation_function: MAX weight_threshold: "
                           "4.0 num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(AffinityTest, OutputClusterEarly) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  ASSERT_OK(graph.AddEdge(2, 3, 2.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));

  ASSERT_OK_AND_ASSIGN(auto clusters, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      active_cluster_conditions { min_density: 3.0 }
      num_iterations: 2
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2, 3}}));

  ASSERT_OK_AND_ASSIGN(clusters,
                       clusterer->Cluster(PARSE_TEXT_PROTO(
                           "affinity_clusterer_config { num_iterations: 2 }")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2, 3}}));
}

TEST(AffinityTest, DynamicWeightThreshold) {
  SimpleUndirectedGraph graph;
  ASSERT_OK(graph.AddEdge(0, 1, 2.0));
  ASSERT_OK(graph.AddEdge(1, 2, 1.0));
  auto clusterer = std::make_unique<AffinityClusterer>();
  ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));

  ASSERT_OK_AND_ASSIGN(auto clusters, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      dynamic_weight_threshold_config {
        weight_decay_function: EXPONENTIAL_DECAY
        upper_bound: 2.0
        lower_bound: 0.51
      }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1}, {2}}));

  ASSERT_OK_AND_ASSIGN(clusters, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
    affinity_clusterer_config {
      num_iterations: 2
      dynamic_weight_threshold_config {
        weight_decay_function: EXPONENTIAL_DECAY
        upper_bound: 2.0
        lower_bound: 0.5
      }
    })pb")));
  EXPECT_THAT(CanonicalizeClustering(clusters),
              ElementsAreArray<Cluster>({{0, 1, 2}}));
}

// Check that single-level tiebreaking is consistent with distributed affinity
// clustering
TEST(AffinityTest, Tiebreaking) {
  absl::BitGen bitgen;

  for (int i = 0; i < 100; ++i) {
    std::vector<int> node_id_permutation(5);
    std::iota(node_id_permutation.begin(), node_id_permutation.end(), 0);
    absl::c_shuffle(node_id_permutation, bitgen);

    std::vector<std::string> node_ids(5);
    std::vector<std::string> permuted_node_ids(5);
    for (int j = 0; j < 5; ++j) {
      node_ids[j] = std::to_string(absl::Uniform<uint64_t>(bitgen));
      permuted_node_ids[node_id_permutation[j]] = node_ids[j];
    }

    SimpleUndirectedGraph graph;
    ASSERT_OK(graph.AddEdge(node_id_permutation[0], node_id_permutation[1], 5));
    ASSERT_OK(graph.AddEdge(node_id_permutation[1], node_id_permutation[2], 1));
    ASSERT_OK(graph.AddEdge(node_id_permutation[2], node_id_permutation[3], 1));
    ASSERT_OK(graph.AddEdge(node_id_permutation[3], node_id_permutation[4], 5));

    auto clusterer = std::make_unique<AffinityClusterer>();
    ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
    clusterer->set_node_id_map(&permuted_node_ids);

    // In every case {0, 1} and {3, 4} are clustered together. Whether 2 joins
    // the first or the second cluster depends on the tiebreaking.
    std::vector<int> expected_cluster_0 = {node_id_permutation[0],
                                           node_id_permutation[1]};
    std::vector<int> expected_cluster_1 = {node_id_permutation[3],
                                           node_id_permutation[4]};
          if (util::Hash64(node_ids[3]) >
      util::Hash64(node_ids[1])) {
      expected_cluster_1.push_back(node_id_permutation[2]);
    } else {
      expected_cluster_0.push_back(node_id_permutation[2]);
    }

    ASSERT_OK_AND_ASSIGN(auto clusters,
                         clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
                           affinity_clusterer_config {})pb")));
    EXPECT_THAT(CanonicalizeClustering(clusters),
                UnorderedElementsAreArray(
                    {UnorderedElementsAreArray(expected_cluster_0),
                     UnorderedElementsAreArray(expected_cluster_1)}));
  }
}

void VerifyDeterminism(int max_num_nodes, int max_num_edges,
                       int edge_weight_denominator,
                       int edge_weight_precision_bits,
                       int num_random_permutations) {
  absl::BitGen bitgen;
  int n = absl::Uniform<int>(bitgen, 3, max_num_nodes);
  int m = absl::Uniform<int>(bitgen, 2, max_num_edges);

  // Pick random edges
  std::vector<std::vector<double>> edges(n, std::vector<double>(n, 0));
  for (int j = 0; j < m; ++j) {
    edges[absl::Uniform<int>(bitgen, 0, n)][absl::Uniform<int>(bitgen, 0, n)] =
        absl::Uniform<int>(bitgen, 1, edge_weight_denominator) /
        static_cast<double>(edge_weight_denominator);
  }

  // Symmetrize and round edge weights
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      double shift = 1LL << edge_weight_precision_bits;
      edges[i][j] = edges[j][i] =
          std::round(std::max(edges[i][j], edges[j][i]) * shift) / shift;
    }
  }

  // Pick random node ids & initialize permutation
  std::vector<int> permutation(n);
  std::vector<std::string> node_ids(n);
  for (int i = 0; i < n; ++i) {
    permutation[i] = i;
    node_ids[i] = std::to_string(absl::Uniform<unsigned int>(bitgen));
  }

  std::vector<std::vector<std::string>> expected_string_clustering;
  for (int t = 0; t < num_random_permutations; ++t) {
    absl::c_shuffle(permutation, bitgen);
    std::vector<std::string> permuted_node_ids(n);
    for (int j = 0; j < n; ++j) {
      permuted_node_ids[permutation[j]] = node_ids[j];
    }

    // Build graph
    SimpleUndirectedGraph graph;
    graph.SetNumNodes(n);
    for (int x = 0; x < n; ++x) {
      for (int y = 0; y < x; ++y) {
        if (edges[x][y] == 0) continue;
        EXPECT_OK(graph.AddEdge(permutation[x], permutation[y], edges[x][y]));
      }
    }

    // Cluster
    auto clusterer = std::make_unique<AffinityClusterer>();
    ASSERT_OK(CopyGraph(graph, clusterer->MutableGraph()));
    clusterer->set_node_id_map(&permuted_node_ids);

    ASSERT_OK_AND_ASSIGN(
        auto clustering, clusterer->Cluster(PARSE_TEXT_PROTO(R"pb(
          affinity_clusterer_config { num_iterations: 2 })pb")));
    // Canonicalize the clustering
    std::vector<std::vector<std::string>> string_clustering;
    for (const auto& cluster : clustering) {
      string_clustering.emplace_back();
      for (const auto element : cluster) {
        string_clustering.back().push_back(permuted_node_ids[element]);
      }
      absl::c_sort(string_clustering.back());
    }
    absl::c_sort(string_clustering);
    if (expected_string_clustering.empty()) {
      // First permutation - just save the clustering
      expected_string_clustering = std::move(string_clustering);
    } else {
      EXPECT_EQ(string_clustering, expected_string_clustering);
    }
  }
}
TEST(AffinityTest, AlgorithmIsDeterministicUnderPermutationsOfNodeIds) {
  for (int repetition = 0; repetition < 200; ++repetition) {
    VerifyDeterminism(/*max_num_nodes=*/30, /*max_num_edges=*/500,
                      /*edge_weight_denominator=*/17,
                      /*edge_weight_precision_bits=*/30,
                      /*num_random_permutations=*/10);
  }
}

}  // namespace
}  // namespace graph_mining::in_memory
