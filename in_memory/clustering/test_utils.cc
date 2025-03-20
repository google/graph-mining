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

#include "in_memory/clustering/test_utils.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/graph_utils.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

using ::graph_mining::in_memory::OutputIndicesById;
using ::testing::AllOf;
using ::testing::Each;
using ::testing::Ge;
using ::testing::Lt;

void AddUndirectedCliqueToGraph(int32_t size, int32_t initial_n,
                                SimpleUndirectedGraph* graph) {
  ABSL_CHECK_GE(size, 0);
  ABSL_CHECK_GE(initial_n, 0);
  int32_t final_n = initial_n + size;
  for (NodeId id1 = initial_n; id1 < final_n - 1; id1++) {
    for (NodeId id2 = id1 + 1; id2 < final_n; id2++) {
      ABSL_CHECK_OK(graph->AddEdge(id1, id2, 1.0));
    }
  }
}

void AddUndirectedCliqueBarbellGraph(int32_t size1, int32_t size2,
                                     SimpleUndirectedGraph* graph) {
  AddUndirectedCliqueToGraph(size1, /*initial_n=*/0, graph);
  AddUndirectedCliqueToGraph(size2, /*initial_n=*/size1, graph);
  ABSL_CHECK_OK(graph->AddEdge(0, size1, 1.0));
}

std::unique_ptr<SimpleUndirectedGraph> MakeUndirectedCliqueBarbellGraph(
    int32_t size1, int32_t size2) {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  AddUndirectedCliqueBarbellGraph(size1, size2, graph.get());
  return graph;
}

std::unique_ptr<SimpleUndirectedGraph> MakeSmallTestGraph() {
  auto graph = MakeUndirectedCliqueBarbellGraph(2, 4);
  ABSL_CHECK_OK(graph->AddEdge(0, 3, 2.0));
  ABSL_CHECK_OK(graph->SetEdgeWeight(0, 1, 3.0));
  return graph;
}

void AddClusteringWithOverlapNode(SimpleUndirectedGraph* graph) {
  AddUndirectedCliqueToGraph(5, 0, graph);
  AddUndirectedCliqueToGraph(5, 5, graph);
  AddUndirectedCliqueToGraph(10, 10, graph);
  InMemoryClusterer::NodeId overlap_node = graph->AddNode();
  for (InMemoryClusterer::NodeId id = 0; id < 10; id++) {
    ABSL_CHECK_OK(graph->AddEdge(overlap_node, id, 1.0));
    ABSL_CHECK_OK(graph->AddEdge(id, overlap_node, 1.0));
  }
}

double ComputeClusterModularity(const absl::flat_hash_set<NodeId>& cluster,
                                const SimpleUndirectedGraph& graph,
                                double resolution, double scale_power) {
  std::vector<double> weighted_degrees = WeightedDegrees(graph);
  double graph_weight =
      std::accumulate(weighted_degrees.begin(), weighted_degrees.end(), 0.0);
  double cluster_modularity = 0.0;
  for (NodeId node1 : cluster) {
    for (NodeId node2 : cluster) {
      cluster_modularity += graph.EdgeWeight(node1, node2).value_or(0.0) -
                            resolution * weighted_degrees[node1] *
                                weighted_degrees[node2] / graph_weight;
    }
  }
  return cluster_modularity / pow(cluster.size(), scale_power);
}

absl::StatusOr<Clustering> Cluster(
    const InMemoryClusterer& clusterer,
    const graph_mining::in_memory::ClustererConfig& config,
    InMemoryClustererMethod clusterer_method, const int num_nodes) {
  switch (clusterer_method) {
    case InMemoryClustererMethod::kVectorOfClusters: {
      return clusterer.Cluster(config);
    }
    case InMemoryClustererMethod::kVectorOfClusterIds: {
      ASSIGN_OR_RETURN(std::vector<NodeId> cluster_ids,
                       clusterer.ClusterAndReturnClusterIds(config));
      EXPECT_THAT(cluster_ids, Each(AllOf(Ge(0), Lt(num_nodes))));
      return OutputIndicesById<NodeId, NodeId>(cluster_ids);
    }
  }
  ABSL_LOG(FATAL) << "Unsupported clusterer method: "
                  << absl::StrCat(clusterer_method);
}

}  // namespace graph_mining::in_memory
