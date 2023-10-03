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

#include "in_memory/clustering/coconductance/coconductance_internal.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining {
namespace in_memory {

using NodeId = InMemoryClusterer::NodeId;

// Initially, each node is in its own cluster.
ClusteringState InitialState(const SimpleUndirectedGraph& graph) {
  ClusteringState result;
  result.cluster_ids.resize(graph.NumNodes());
  result.cluster_weight.resize(graph.NumNodes());
  result.cluster_edges.resize(graph.NumNodes());
  for (int i = 0; i < graph.NumNodes(); ++i) {
    result.cluster_ids[i] = i;
    result.cluster_weight[i] = graph.NodeWeight(i);
    result.cluster_edges[i] = graph.EdgeWeight(i, i).value_or(0);
  }
  return result;
}

double ObjectiveChangeAfterMove(NodeId node, NodeId new_cluster,
                                const SimpleUndirectedGraph& graph,
                                const ClusteringState& state,
                                double edges_to_current_cluster,
                                double edges_to_new_cluster, double exponent) {
  NodeId current_cluster = state.cluster_ids[node];
  if (new_cluster == current_cluster) return 0.0;

  auto objective = [exponent](double edge_weight, double node_weight) {
    return ClusterObjective(edge_weight, node_weight, exponent);
  };

  double new_state =
      objective(
          state.cluster_edges[current_cluster] - edges_to_current_cluster,
          state.cluster_weight[current_cluster] - graph.NodeWeight(node)) +
      objective(state.cluster_edges[new_cluster] + edges_to_new_cluster +
                    graph.EdgeWeight(node, node).value_or(0.0),
                state.cluster_weight[new_cluster] + graph.NodeWeight(node));

  double old_state = objective(state.cluster_edges[current_cluster],
                               state.cluster_weight[current_cluster]) +
                     objective(state.cluster_edges[new_cluster],
                               state.cluster_weight[new_cluster]);

  return new_state - old_state;
}

void MoveNodeAndUpdateState(ClusteringState& state, NodeId node,
                            NodeId new_cluster,
                            const SimpleUndirectedGraph& graph,
                            double edges_to_current_cluster,
                            double edges_to_new_cluster) {
  NodeId current_cluster = state.cluster_ids[node];
  if (current_cluster == new_cluster) return;

  state.cluster_ids[node] = new_cluster;
  state.cluster_weight[new_cluster] += graph.NodeWeight(node);
  state.cluster_weight[current_cluster] -= graph.NodeWeight(node);
  state.cluster_edges[new_cluster] +=
      edges_to_new_cluster + graph.EdgeWeight(node, node).value_or(0.0);
  state.cluster_edges[current_cluster] -= edges_to_current_cluster;
}

namespace {

// Colors a graph uniformly into two colors and then returns edges (s, t) of the
// graph, which satisfy the following:
//  * the color of s is 0 and the color of t is 1 (hence, the edges define a
//    bipartite graph)
//  * weight(s) >= weight(t)
std::vector<std::pair<NodeId, NodeId>> RandomColorInducedBipartiteSubgraph(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    absl::BitGenRef rng) {
  std::vector<bool> t_side(graph.NumNodes());
  for (NodeId node = 0; node < graph.NumNodes(); ++node) {
    t_side[node] = absl::Uniform<char>(rng, 0, 2);
  }

  std::vector<std::pair<NodeId, NodeId>> result;
  for (NodeId node = 0; node < graph.NumNodes(); ++node) {
    if (t_side[node]) continue;
    for (const auto& [neighbor, _] : graph.Neighbors(node)) {
      if (t_side[neighbor] &&
          graph.NodeWeight(node) >= graph.NodeWeight(neighbor)) {
        result.emplace_back(node, neighbor);
      }
    }
  }
  return result;
}

void ValidateWeights(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph) {
  for (NodeId node = 0; node < graph.NumNodes(); ++node) {
    ABSL_CHECK_LT(std::abs(graph.NodeWeight(node) - graph.WeightedDegree(node)),
                  1e-5);
  }
}

}  // namespace

std::vector<NodeId> ConstantApproximateCoconductance(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    absl::BitGenRef rng) {
  ValidateWeights(graph);
  auto bipartite_edges = RandomColorInducedBipartiteSubgraph(graph, rng);
  std::sort(bipartite_edges.begin(), bipartite_edges.end(),
            [&graph](const std::pair<NodeId, NodeId>& x,
                     const std ::pair<NodeId, NodeId>& y) {
              return graph.NodeWeight(x.first) + graph.NodeWeight(x.second) <
                     graph.NodeWeight(y.first) + graph.NodeWeight(y.second);
            });

  std::vector<NodeId> cluster_ids(graph.NumNodes());
  std::iota(cluster_ids.begin(), cluster_ids.end(), 0);

  // We assume that each edge in bipartite_edges has their first endpoint on the
  // S-side and their second endpoint on the T-side.

  // For each node on the S-side, the weight of the nodes on the T-side matched
  // to it.
  std::vector<double> t_weight(graph.NumNodes());
  std::vector<bool> t_node_matched(graph.NumNodes());

  for (const auto& [s, t] : bipartite_edges) {
    if (!t_node_matched[t] &&
        t_weight[s] + graph.NodeWeight(t) <= 2 * graph.NodeWeight(s)) {
      t_node_matched[t] = true;
      t_weight[s] += graph.NodeWeight(t);
      cluster_ids[t] = s;
    }
  }
  return cluster_ids;
}

double CoconductanceObjective(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    const std::vector<NodeId>& cluster_ids) {
  ABSL_CHECK_EQ(graph.NumNodes(), cluster_ids.size());
  ValidateWeights(graph);
  std::vector<double> total_volume(graph.NumNodes());
  std::vector<double> total_edges(graph.NumNodes());
  std::vector<bool> nonempty_cluster(graph.NumNodes());

  for (NodeId node = 0; node < graph.NumNodes(); ++node) {
    ABSL_CHECK_GE(cluster_ids[node], 0);
    ABSL_CHECK_LT(cluster_ids[node], graph.NumNodes());
    nonempty_cluster[cluster_ids[node]] = true;
    total_volume[cluster_ids[node]] += graph.NodeWeight(node);
    for (const auto& [neighbor, weight] : graph.Neighbors(node)) {
      if (cluster_ids[neighbor] == cluster_ids[node]) {
        total_edges[cluster_ids[node]] += weight;
      }
    }
  }
  double result = 0.0;
  for (NodeId cluster = 0; cluster < graph.NumNodes(); ++cluster) {
    if (!nonempty_cluster[cluster]) continue;

    if (total_volume[cluster] > 0) {
      // No need to multiply by 2, since we counted each undirected edge twice.
      result += total_edges[cluster] / total_volume[cluster];
    } else {
      ++result;
    }
  }
  return result;
}

std::vector<NodeId> ConstantApproximateCoconductance(
    const graph_mining::in_memory::SimpleUndirectedGraph& graph,
    int num_repetitions) {
  ABSL_CHECK_GT(num_repetitions, 0);
  ValidateWeights(graph);

  std::vector<NodeId> best_cluster_ids;
  double best_objective = -1;
  absl::BitGen bg;
  for (int i = 0; i < num_repetitions; ++i) {
    auto cluster_ids = ConstantApproximateCoconductance(graph, bg);
    double objective = CoconductanceObjective(graph, cluster_ids);
    if (objective > best_objective) {
      best_cluster_ids = std::move(cluster_ids);
      best_objective = objective;
    }
  }
  ABSL_CHECK_GT(best_objective, 0.0);
  return best_cluster_ids;
}
}  // namespace in_memory
}  // namespace graph_mining
