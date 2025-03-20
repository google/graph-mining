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

#include "in_memory/clustering/clique_container/clique_container.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/clique_container/clique_container.pb.h"
#include "in_memory/clustering/clique_container/internal.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining {
namespace in_memory {
namespace {

// Appends vector2 to vector1, destroying vector2 in the process.
void ConcatenateVectors(std::vector<ClusterContents>& vector1,
                        std::vector<ClusterContents>&& vector2) {
  vector1.insert(vector1.end(), std::make_move_iterator(vector2.begin()),
                 std::make_move_iterator(vector2.end()));
}

double NChoose2(double n) { return n * (n - 1.0) / 2.0; }

double Density(int64_t num_nodes, int64_t num_edges) {
  if (num_nodes <= 1) return 1.0;

  return num_edges / NChoose2(num_nodes);
}

// Computes the density of a graph obtained as follows. First, start with a
// graph with num_nodes nodes and num_edges undirected edges. Then, add a clique
// on num_clique_nodes nodes and add all possible edges between the newly added
// clique nodes and the existing num_nodes.
double CombinedDensity(int num_nodes, int num_edges, int num_clique_nodes) {
  if (num_nodes + num_clique_nodes <= 1) return 1.0;

  return Density(
      num_nodes + num_clique_nodes,
      num_edges + num_clique_nodes * num_nodes + NChoose2(num_clique_nodes));
}

// Builds a graph with consecutive node ids.
class ConsecutiveIndicesGraphBuilder {
 public:
  ConsecutiveIndicesGraphBuilder(NodeId num_nodes) : node_id_map_(num_nodes) {
    graph_ = std::make_unique<SimpleUndirectedGraph>();
    graph_->SetNumNodes(num_nodes);
  }

  void AddNode(NodeId node_id) { GetIndex(node_id); }

  void AddEdge(NodeId node_id1, NodeId node_id2) {
    ABSL_CHECK_OK(graph_->AddEdge(GetIndex(node_id1), GetIndex(node_id2), 1.0));
  }

  // Requires that exactly `num_nodes` different node ids have been provided
  // through AddNode and AddEdge calls.
  std::pair<std::unique_ptr<SimpleUndirectedGraph>, std::vector<NodeId>>
  Build() && {
    ABSL_CHECK_EQ(node_id_map_.size(), graph_->NumNodes());
    std::vector<NodeId> node_id_map(node_id_map_.size());
    for (auto [node_id, index] : node_id_map_) {
      node_id_map[index] = node_id;
    }
    return {std::move(graph_), std::move(node_id_map)};
  }

 private:
  // Maps node ids to consecutive integers. The return value of GetIndex(x) is
  // computed as follows:
  //  * if this is the first time x is provided as the argument, GetIndex
  //  returns
  //    the number of distinct node ids provided so far.
  //  * otherwise, GetIndex returns the value returned the first time x was
  //    provided as the argument.
  int32_t GetIndex(NodeId node_id) {
    auto [iterator, inserted] =
        node_id_map_.insert({node_id, node_id_map_.size()});
    return iterator->second;
  }

  absl::flat_hash_map<NodeId, int32_t> node_id_map_;

  std::unique_ptr<SimpleUndirectedGraph> graph_;
};

// Recursive function for computing clique containers.
//  * graph is the input graph.
//  * node_id_map maps node ids of `graph` to the "global" node ids that should
//  be used to return the result.
//  * partial_container is the set of nodes that should be added to the returned
//  clique containers.
//  * min_density is the minimum density of the clique containers to be
//    returned.
std::vector<std::vector<NodeId>> CliqueContainers(
    std::unique_ptr<SimpleUndirectedGraph> graph,
    const std::vector<NodeId>& node_id_map,
    const std::vector<NodeId>& partial_container, double min_density) {
  int num_nodes = graph->NumNodes();

  // Number of *undirected* edges.
  int64_t num_edges = 0;
  for (int i = 0; i < graph->NumNodes(); ++i)
    num_edges += graph->Neighbors(i).size();
  num_edges /= 2;

  if (CombinedDensity(num_nodes, num_edges, partial_container.size()) >=
      min_density) {
    if (partial_container.size() + graph->NumNodes() <= 1) {
      // Don't return containers of size 1.
      return {};
    } else {
      auto result = partial_container;
      result.insert(result.end(), node_id_map.begin(), node_id_map.end());
      return {result};
    }
  }

  auto graph_copy = std::make_unique<SimpleUndirectedGraph>();
  ABSL_CHECK_OK(CopyGraph(*graph, graph_copy.get()));
  auto degeneracy_ordering = DegeneracyOrdering(std::move(graph_copy));
  auto directed_graph = DirectGraph(*graph, degeneracy_ordering);
  std::vector<std::vector<NodeId>> result;

  for (int i = 0; i < degeneracy_ordering.size(); ++i) {
    auto node_id = degeneracy_ordering[i];
    ConsecutiveIndicesGraphBuilder recursive_graph_builder(
        directed_graph->Neighbors(node_id).size());
    for (auto [neighbor_id, _] : directed_graph->Neighbors(node_id)) {
      recursive_graph_builder.AddNode(neighbor_id);
      for (auto [neighbor_neighbor_id, __] :
           directed_graph->Neighbors(neighbor_id)) {
        if (directed_graph->Neighbors(node_id).contains(neighbor_neighbor_id)) {
          recursive_graph_builder.AddEdge(neighbor_id, neighbor_neighbor_id);
        }
      }
    }
    auto [recursive_graph, recursive_node_id_map] =
        std::move(recursive_graph_builder).Build();

    auto recursive_partial_container = partial_container;
    recursive_partial_container.push_back(node_id_map[node_id]);

    ConcatenateVectors(
        result,
        CliqueContainers(std::move(recursive_graph), recursive_node_id_map,
                         recursive_partial_container, min_density));

    // Now, delete the node and exit early if the density is high enough.
    --num_nodes;
    num_edges -= directed_graph->Neighbors(node_id).size();
    if (CombinedDensity(num_nodes, num_edges, partial_container.size()) >=
        min_density) {
      if (partial_container.size() + num_nodes <= 1) return result;

      std::vector<NodeId> new_container = partial_container;
      for (int j = i + 1; j < graph->NumNodes(); ++j) {
        new_container.push_back(node_id_map[degeneracy_ordering[j]]);
      }

      result.push_back(new_container);
      return result;
    }
  }

  return result;
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering> CliqueContainerClusterer::Cluster(
    const graph_mining::in_memory::ClustererConfig& config) const {
  auto graph = std::make_unique<SimpleUndirectedGraph>();
  RETURN_IF_ERROR(CopyGraph(graph_, graph.get()));

  std::vector<NodeId> node_id_map(graph->NumNodes());
  std::iota(node_id_map.begin(), node_id_map.end(), 0);

  std::vector<std::vector<NodeId>> containers =
      CliqueContainers(std::move(graph), node_id_map, {},
                       config.clique_container_config().min_density());

  return SortClustersAndRemoveContained(containers);
}

}  // namespace in_memory
}  // namespace graph_mining
