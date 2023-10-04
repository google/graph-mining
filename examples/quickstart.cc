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

#include <iostream>
#include <vector>

#include "absl/status/status.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/parhac.h"
#include "in_memory/status_macros.h"

using graph_mining::in_memory::ParHacClusterer;
using graph_mining::in_memory::SimpleUndirectedGraph;
using research_graph::in_memory::ClustererConfig;

absl::Status ClusteringExample() {
  // Initialize a graph, where higher weight = more similar.
  SimpleUndirectedGraph graph;
  RETURN_IF_ERROR(
      graph.AddEdge(/*from_node=*/0, /*to_node=*/1,
                    /*weight=*/0.9));
  RETURN_IF_ERROR(graph.AddEdge(0, 2, 0.9));
  RETURN_IF_ERROR(graph.AddEdge(1, 2, 0.8));
  RETURN_IF_ERROR(graph.AddEdge(2, 3, 0.1));
  RETURN_IF_ERROR(graph.AddEdge(3, 4, 0.9));

  // Initialize the clusterer.
  ParHacClusterer clusterer;
  RETURN_IF_ERROR(CopyGraph(graph, clusterer.MutableGraph()));

  // Run parhac clustering.
  ClustererConfig config;
  config.mutable_parhac_clusterer_config()->set_weight_threshold(0.3);
  ASSIGN_OR_RETURN(auto clustering, clusterer.Cluster(config));

  // Print the cluster assignments.
  for (const std::vector<int>& cluster : clustering) {
    for (auto id : cluster) {
      std::cout << id << ",";
    }
    std::cout << std::endl;
  }

  return absl::OkStatus();
}

int main(void) {
  auto status = ClusteringExample();
  if (!status.ok()) {
    std::cout << "Error: " << status.ToString() << std::endl;
  }
  return 0;
}
