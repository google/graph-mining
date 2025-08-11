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

#include "in_memory/clustering/clique_aggregator/clique_aggregator.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "in_memory/clustering/clique_aggregator/bitset.h"
#include "in_memory/clustering/clique_aggregator/clique_aggregator.pb.h"
#include "in_memory/clustering/clique_aggregator/degeneracy_orientation.h"
#include "in_memory/clustering/clique_aggregator/graphs.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"

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

// Maps node IDs to consecutive integers.
class Indexer {
 public:
  explicit Indexer(NodeId num_nodes) : num_nodes_(num_nodes) {
    node_id_map_.reserve(num_nodes_);
  }

  // Returns the remapped node ID of `node_id`.
  NodeId GetExistingIndex(NodeId node_id) const {
    auto it = node_id_map_.find(node_id);
    ABSL_CHECK(it != node_id_map_.end());
    return it->second;
  }

  // Maps node IDs to consecutive integers. The return value of
  // `GetOrCreateIndex(x)` is computed as follows:
  //  * if this is the first time `x` is provided as the argument,
  //    `GetOrCreateIndex` returns the number of distinct node IDs provided so
  //    far.
  //  * otherwise, `GetOrCreateIndex` returns the value returned the first time
  //    `x` was provided as the argument.
  int32_t GetOrCreateIndex(NodeId node_id) {
    auto [iterator, inserted] =
        node_id_map_.insert({node_id, node_id_map_.size()});
    ABSL_CHECK_LE(node_id_map_.size(), num_nodes_);
    return iterator->second;
  }

  // Returns a vector mapping the remapped indices back to the original node
  // IDs.
  std::vector<NodeId> InverseNodeIdMap() const {
    std::vector<NodeId> inverse_map(node_id_map_.size());
    for (const auto& [node_id, index] : node_id_map_) {
      inverse_map[index] = node_id;
    }
    return inverse_map;
  }

 private:
  // Maps node IDs from input ID space to (remapped) consecutive integers.
  absl::flat_hash_map<NodeId, NodeId> node_id_map_;
  NodeId num_nodes_;
};

// A template class for building a graph for the recursive calls to
// `CliqueAggregator`.
template <typename GraphT>
class ConsecutiveIndicesGraphBuilder {};

// Given a graph using arbitrary integer node IDs, build a new graph where the
// ID space is remapped to [0, num_nodes). Usage:
//  1. Specify the final number of nodes of the remapped graph in the
//     constructor.
//  2. Add all the edges using `AddEdge` calls.
//  3. If there are isolated nodes, add them using `AddNode`. It's fine to add
//     the same node multiple times or add a node that was implicitly added by
//     an `AddEdge` call.
//  4. Call `Build()` to build the remapped graph. The instance is no longer
//     usable after calling `Build()`.
template <>
class ConsecutiveIndicesGraphBuilder<BitSetGraph> {
 public:
  explicit ConsecutiveIndicesGraphBuilder(NodeId num_nodes)
      : indexer_(num_nodes) {
    graph_ = std::make_unique<BitSetGraph>(num_nodes);
  }

  // Adds a node with the given ID. If the node already exists, does nothing.
  void AddNode(NodeId node_id) { indexer_.GetOrCreateIndex(node_id); }

  // Adds an undirected edge between the two nodes, implicitly adding the edge
  // endpoints if necessary. If the edge already exists, does nothing.
  void AddEdge(NodeId node_id1, NodeId node_id2) {
    auto index1 = indexer_.GetOrCreateIndex(node_id1);
    auto index2 = indexer_.GetOrCreateIndex(node_id2);
    graph_->AddEdge(index1, index2);
    graph_->AddEdge(index2, index1);
  }

  // Requires that exactly `num_nodes` different node IDs have been provided
  // through `AddNode` and `AddEdge` calls.
  // Returns the graph and a vector that maps the remapped node IDs to the
  // original node IDs.
  std::pair<absl_nonnull std::unique_ptr<BitSetGraph>, std::vector<NodeId>>
  Build() && {
    return {std::move(graph_), indexer_.InverseNodeIdMap()};
  }

  NodeId GetExistingIndex(NodeId node_id) const {
    return indexer_.GetExistingIndex(node_id);
  }

 private:
  Indexer indexer_;
  absl_nonnull std::unique_ptr<BitSetGraph> graph_;
};

template <>
class ConsecutiveIndicesGraphBuilder<GbbsGraphWrapper<UnweightedGbbsGraph>> {
 public:
  explicit ConsecutiveIndicesGraphBuilder(NodeId num_nodes)
      : indexer_(num_nodes), adjacency_lists_(num_nodes) {}

  // Adds a node with the given ID. If the node already exists, does nothing.
  void AddNode(NodeId node_id) { indexer_.GetOrCreateIndex(node_id); }

  // Adds an undirected edge between the two nodes, implicitly adding the edge
  // endpoints if necessary. If the edge already exists, does nothing.
  void AddEdge(NodeId node_id1, NodeId node_id2) {
    auto index1 = indexer_.GetOrCreateIndex(node_id1);
    auto index2 = indexer_.GetOrCreateIndex(node_id2);
    adjacency_lists_[index1].push_back(index2);
    adjacency_lists_[index2].push_back(index1);
  }

  // Requires that exactly `num_nodes` different node IDs have been provided
  // through `AddNode` and `AddEdge` calls.
  // Returns the graph and a vector that maps the remapped node IDs to the
  // original node IDs.
  std::pair<absl_nonnull std::unique_ptr<GbbsGraphWrapper<UnweightedGbbsGraph>>,
            std::vector<NodeId>>
  Build() && {
    std::vector<NodeId> node_id_map = indexer_.InverseNodeIdMap();
    ABSL_CHECK_EQ(node_id_map.size(), adjacency_lists_.size());

    auto graph = std::make_unique<UnweightedGbbsGraph>();
    ABSL_CHECK_OK(graph->PrepareImport(adjacency_lists_.size()));

    for (NodeId i = 0; i < adjacency_lists_.size(); ++i) {
      absl::c_sort(adjacency_lists_[i]);
      CliqueAggregatorClusterer::AdjacencyList adj;
      adj.id = i;
      adj.outgoing_edges.reserve(adjacency_lists_[i].size());
      NodeId previous_id = -1;
      for (NodeId j : adjacency_lists_[i]) {
        ABSL_CHECK_GT(j, previous_id);
        previous_id = j;
        adj.outgoing_edges.emplace_back(j, 1.0);
      }
      ABSL_CHECK_OK(graph->Import(std::move(adj)));
    }

    ABSL_CHECK_OK(graph->FinishImport());

    return {std::make_unique<GbbsGraphWrapper<UnweightedGbbsGraph>>(
                std::move(graph)),
            std::move(node_id_map)};
  }

  NodeId GetExistingIndex(NodeId node_id) const {
    return indexer_.GetExistingIndex(node_id);
  }

 private:
  Indexer indexer_;
  std::vector<std::vector<NodeId>> adjacency_lists_;
};

// Returns true iff there exists a node that
// * comes in `degeneracy_ordering` no later than `last_node_index`, and
// * has edges to all nodes in `degeneracy_ordering` that are after
//   `last_node_index`.
//
// Crashes if any of the following is true:
//  * `degeneracy_ordering` is not a permutation of [0,
//    `degeneracy_ordering.size()`).
//  * `last_node_index` is not in [0, `degeneracy_ordering.size()`).
template <typename GraphT>
bool ExistsNodeWithEdgesToAllLaterNodes(
    const GraphT& graph, absl::Span<const NodeId> degeneracy_ordering,
    int32_t last_node_index) {
  ABSL_CHECK_LT(last_node_index, degeneracy_ordering.size());
  ABSL_CHECK_GE(last_node_index, 0);
  ABSL_CHECK_EQ(degeneracy_ordering.size(), graph.NumNodes());
  std::vector<bool> is_past_last_node_index(degeneracy_ordering.size(), false);
  std::vector<int> degeneracy_index(degeneracy_ordering.size(), -1);
  for (int i = 0; i < degeneracy_ordering.size(); ++i) {
    ABSL_CHECK_GE(degeneracy_ordering[i], 0);
    ABSL_CHECK_LT(degeneracy_ordering[i], degeneracy_ordering.size());
    is_past_last_node_index[degeneracy_ordering[i]] = i > last_node_index;
  }
  for (int i = 0; i <= last_node_index; ++i) {
    int later_nodes_seen = 0;
    graph.MapNeighbors(degeneracy_ordering[i], [&](NodeId ngh) {
      if (is_past_last_node_index[ngh]) {
        ++later_nodes_seen;
      }
    });

    if (later_nodes_seen + last_node_index + 1 == degeneracy_ordering.size()) {
      return true;
    }
  }
  return false;
}

// This function is called when we are about to add a new cluster to the result,
// which will be obtained by adding all nodes in `degeneracy_ordering` that are
// after `last_node_index`. This function returns true iff we have already added
// a cluster that covers this set of nodes.
template <typename GraphT>
bool RemainingNodesAreAlreadyCovered(
    const GraphT& directed_graph, absl::Span<const BitSet> covered_sets,
    absl::Span<const NodeId> degeneracy_ordering, int32_t last_node_index) {
  if (!covered_sets.empty()) {
    // If `covered_sets` is non-empty, we are *not* being called from the
    // outermost recursive call. Hence, we work with a set of nodes of size at
    // most the graph degeneracy, and can use a `BitSet` to represent it.
    BitSet remaining_nodes(directed_graph.NumNodes());
    for (int j = last_node_index + 1; j < directed_graph.NumNodes(); ++j) {
      ABSL_DCHECK_GE(degeneracy_ordering[j], 0);
      ABSL_DCHECK_LT(degeneracy_ordering[j], directed_graph.NumNodes());
      remaining_nodes.Insert(degeneracy_ordering[j]);
    }

    for (const auto& covered_set : covered_sets) {
      if (covered_set.IsSupersetOf(remaining_nodes)) return true;
    }
  }

  return ExistsNodeWithEdgesToAllLaterNodes(directed_graph, degeneracy_ordering,
                                            last_node_index);
}

// Computes the set of covered sets for a recursive call to `CliqueAggregator`.
//  * `covered_sets` is the set of covered sets.
//  * `node_id` is the node on whose neighborhood we recurse.
//  * `directed_graph` is a directed graph we use for the recursive call.
//  * `transposed_graph` is the transposed graph corresponding to
//    `directed_graph`.
//  * `map_to_recursive_node_id` maps node IDs to node IDs in the recursive
//    graph.
// `GraphT` is a const reference to either `GbbsGraphWrapper` or `BitSetGraph`.
template <typename GraphT>
std::vector<BitSet> BuildRecursiveCoveredSets(
    absl::Span<const BitSet> covered_sets, NodeId node_id,
    const GraphT& directed_graph, const GraphT& transposed_graph,
    absl::AnyInvocable<NodeId(NodeId)> map_to_recursive_node_id) {
  std::vector<BitSet> recursive_covered;

  // Add a covered set to the result, which is obtained by intersecting the
  // neighborhood of `node_id` with `other`. Here `other` is either a `BitSet`
  // or a node ID, in which case we intersect with the neighborhood of the
  // corresponding node in `directed_graph`.
  auto add_covered_set = [&]<typename T>(const T& other) {
    BitSet recursive_covered_set(directed_graph.Degree(node_id));
    directed_graph.MapCommonNeighbors(node_id, other, [&](NodeId covered_node) {
      recursive_covered_set.Insert(map_to_recursive_node_id(covered_node));
    });

    // Don't add empty sets, unless the graph in the recursive call is empty,
    // in which case an empty set would prune the recursive call entirely.
    if (directed_graph.Degree(node_id) == 0 || recursive_covered_set.Size() > 0)
      recursive_covered.push_back(recursive_covered_set);
  };

  // Case 1: for all covered sets that contain `node_id`, intersect them with
  // the set of nodes in the recursive call and remap node IDs.
  for (const auto& covered_set : covered_sets) {
    if (!covered_set.Contains(node_id)) continue;
    add_covered_set(covered_set);
  }

  // Case 2: All nodes that come in the degeneracy ordering before `node_id`
  // define new covered sets. Specifically, for all such nodes we have covered
  // all cliques in their directed out-neighborhood in `directed_graph`. Hence,
  // we look at all nodes that have an edge to `node_id` and for each such node
  // intersect its neighborhood with the neighborhood of `node_id`.
  transposed_graph.MapNeighbors(
      node_id, [&](NodeId in_neighbor_id) { add_covered_set(in_neighbor_id); });
  return recursive_covered;
}

// Recursive function for computing a clique aggregator.
// * `graph` is the input graph.
// * `node_id_map` maps node ids of `graph` to the "global" node IDs that should
//   be used to return the result.
// * `partial_cluster` is the set of nodes that should be added to the
//   returned clique aggregator.
// * `min_density` is the minimum density of the clusters to be
//   returned and should be in [0, 1].
// * `covered_sets` is a family of sets of nodes. Each set is a subset of
//   `graph` nodes and represents a set of nodes, within which all cliques have
//   already been covered by other recursive calls. The function ensures that no
//   returned cluster is a subset of any of the sets in `covered_sets`.
//   This parameter is analogous to the `X` parameter in
//   https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm.
//   An empty optional disables the pruning entirely. Note that the is a subtle
//   difference between an empty optional and an empty Span. In the former case
//   pruning is not performed in all recursive calls, while in the latter case
//   recursive calls may be passed a non-empty `covered_sets`.
// Template parameters:
// * `GraphT` is the type of the graph used for graph parameter.
// * `RecursiveGraphT` is the type of the graph used for the recursive calls.
template <typename GraphT, typename RecursiveGraphT>
std::vector<std::vector<NodeId>> CliqueAggregator(
    const GraphT& graph, const std::vector<NodeId>& node_id_map,
    const absl::Span<const NodeId> partial_cluster, double min_density,
    std::optional<absl::Span<const BitSet>> covered_sets) {
  int num_nodes = graph.NumNodes();
  const bool bron_kerbosch_pruning = covered_sets.has_value();

  if (bron_kerbosch_pruning) {
    for (const auto& covered_set : *covered_sets) {
      if (covered_set.Size() == num_nodes) return {};
    }
  }

  // Number of *undirected* edges.
  int64_t num_edges = NumEdges(graph) / 2;

  if (CombinedDensity(num_nodes, num_edges, partial_cluster.size()) >=
      min_density) {
    if (partial_cluster.size() + graph.NumNodes() <= 1) {
      // Don't return clusters of size 1.
      return {};
    } else {
      std::vector<NodeId> result(partial_cluster.begin(),
                                 partial_cluster.end());
      result.insert(result.end(), node_id_map.begin(), node_id_map.end());
      return {result};
    }
  }

  auto degeneracy_ordering = DegeneracyOrdering(graph);
  auto [directed_graph, transposed_graph] =
      DirectGraph(graph, degeneracy_ordering);
  std::vector<std::vector<NodeId>> result;

  for (int i = 0; i < degeneracy_ordering.size(); ++i) {
    auto node_id = degeneracy_ordering[i];
    ConsecutiveIndicesGraphBuilder<RecursiveGraphT> recursive_graph_builder(
        directed_graph->Degree(node_id));

    directed_graph->MapNeighbors(node_id, [&](NodeId neighbor_id) {
      recursive_graph_builder.AddNode(neighbor_id);
      directed_graph->MapCommonNeighbors(
          node_id, neighbor_id, [&](NodeId neighbor_neighbor_id) {
            recursive_graph_builder.AddEdge(neighbor_id, neighbor_neighbor_id);
          });
    });

    std::vector<BitSet> recursive_covered;
    if (bron_kerbosch_pruning) {
      recursive_covered = BuildRecursiveCoveredSets(
          *covered_sets, node_id, *directed_graph, *transposed_graph,
          [&recursive_graph_builder](NodeId node_id) {
            return recursive_graph_builder.GetExistingIndex(node_id);
          });
    }

    auto [recursive_graph, recursive_node_id_map] =
        std::move(recursive_graph_builder).Build();

    for (int i = 0; i < recursive_node_id_map.size(); ++i) {
      recursive_node_id_map[i] = node_id_map[recursive_node_id_map[i]];
    }

    std::vector<NodeId> recursive_partial_cluster(partial_cluster.begin(),
                                                  partial_cluster.end());
    recursive_partial_cluster.push_back(node_id_map[node_id]);

    ConcatenateVectors(
        result, CliqueAggregator<RecursiveGraphT, RecursiveGraphT>(
                    *recursive_graph, recursive_node_id_map,
                    recursive_partial_cluster, min_density,
                    bron_kerbosch_pruning
                        ? std::make_optional(absl::MakeSpan(recursive_covered))
                        : std::nullopt));

    // Now, delete the node and exit early if the density is high enough.
    --num_nodes;
    num_edges -= directed_graph->Degree(node_id);
    if (CombinedDensity(num_nodes, num_edges, partial_cluster.size()) >=
        min_density) {
      if (partial_cluster.size() + num_nodes <= 1) return result;

      if (bron_kerbosch_pruning &&
          RemainingNodesAreAlreadyCovered(*directed_graph, *covered_sets,
                                          degeneracy_ordering, i)) {
        return result;
      }

      std::vector<NodeId> new_cluster(partial_cluster.begin(),
                                      partial_cluster.end());
      for (int j = i + 1; j < graph.NumNodes(); ++j) {
        new_cluster.push_back(node_id_map[degeneracy_ordering[j]]);
      }

      result.push_back(new_cluster);
      return result;
    }
  }
  // We should never reach this point, since when we have only one node left,
  // we exit the for loop (as a single node cluster has density 1.0).
  ABSL_LOG(FATAL) << "This should never happen.";
}

}  // namespace

absl::StatusOr<InMemoryClusterer::Clustering>
CliqueAggregatorClusterer::Cluster(
    const graph_mining::in_memory::ClustererConfig& config) const {
  GbbsGraphWrapper<UnweightedGbbsGraph> graph(graph_);

  std::vector<NodeId> node_id_map(graph.NumNodes());
  std::iota(node_id_map.begin(), node_id_map.end(), 0);

  auto aggregator_function =
      config.clique_aggregator_config().use_gbbs_graph_recursively()
          ? CliqueAggregator<GbbsGraphWrapper<UnweightedGbbsGraph>,
                             GbbsGraphWrapper<UnweightedGbbsGraph>>
          : CliqueAggregator<GbbsGraphWrapper<UnweightedGbbsGraph>,
                             BitSetGraph>;

  std::vector<std::vector<NodeId>> aggregator = aggregator_function(
      graph, node_id_map, /*partial_cluster=*/{},
      config.clique_aggregator_config().min_density(),
      config.clique_aggregator_config().bron_kerbosch_pruning()
          ? std::make_optional(absl::Span<const BitSet>())
          : std::nullopt);

  for (auto& cluster : aggregator) {
    absl::c_sort(cluster);
  }

  ABSL_LOG(INFO) << "CliqueAggregatorClusterer output number of clusters: "
                 << aggregator.size();
  return aggregator;
}

}  // namespace in_memory
}  // namespace graph_mining
