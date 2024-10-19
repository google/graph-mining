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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_TERAHAC_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_TERAHAC_INTERNAL_H_

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/status_macros.h"
#include "parlay/sequence.h"

namespace graph_mining::in_memory {

// A wrapper for calling ApproximateSubgraphHac in the in-memory setting. See
// the comments in approximate_subgraph_hac.h for more details.
template <class ClusteredGraph, class NodeIds, class NodeToPartition>
absl::StatusOr<parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, float>>>
ApproximateSubgraphHacWrapper(
    ClusteredGraph& graph,      // handle to the full clustered graph
    const NodeIds& node_ids,  // the nodes mapping to this partition
    const NodeToPartition& node_to_partition,  // V -> partition
    gbbs::uintE partition,                       // this partition's id.
    parlay::sequence<float>&
        all_min_merge_similarities,  // all min_merge_similarities
    float epsilon, float weight_threshold, size_t round) {
  size_t num_in_partition = node_ids.size();
  using uintE = gbbs::uintE;

  absl::flat_hash_map<uintE, uintE> subgraph_id;
  for (size_t i = 0; i < num_in_partition; ++i) {
    uintE orig_u = node_ids[i];
    subgraph_id[orig_u] = i;
  }

  // The total number of nodes in this partition (incremented below to include
  // inactive neighbors).
  size_t num_nodes = num_in_partition;

  for (size_t i = 0; i < num_in_partition; ++i) {
    uintE orig_u = node_ids[i];
    ABSL_CHECK_EQ(node_to_partition[orig_u], partition);
    if (graph.MutableNode(orig_u)->IsActive()) {
      auto map_f = [&](uintE orig_v, AverageLinkageWeight& weight) {
        if (node_to_partition[orig_v] != partition) {
          if (!subgraph_id.contains(orig_v)) {
            subgraph_id[orig_v] = num_nodes;
            ++num_nodes;
          }
        }
      };
      graph.MutableNode(orig_u)->GetNeighbors()->MapSequential(map_f);
    }
  }
  auto min_merge_similarities = std::vector<double>(num_in_partition);

  auto CurrentClusterSize = [&](uintE orig_id) {
    return graph.MutableNode(orig_id)->ClusterSize();
  };

  // Map from subraph_id -> original ids
  auto original_id = std::vector<uintE>(num_nodes);

  // Fill in the min_merge_similarities and cluster_sizes.
  for (const auto& [v, index] : subgraph_id) {
    original_id[index] = v;
    if (index < num_in_partition) {
      min_merge_similarities[index] = all_min_merge_similarities[v];
    }
  }

  auto subgraph = absl::make_unique<SimpleUndirectedGraph>();
  subgraph->SetNumNodes(num_nodes);
  for (size_t i = 0; i < num_in_partition; ++i) {
    uintE orig_u = node_ids[i];
    uintE cluster_size_u = CurrentClusterSize(orig_u);
    auto map_f = [&](uintE orig_v, AverageLinkageWeight& weight) {
      uintE v = subgraph_id[orig_v];
      double similarity = weight.Similarity(cluster_size_u);
      auto status = subgraph->AddEdge(i, v, similarity);
      ABSL_CHECK(status.ok());
    };
    graph.MutableNode(orig_u)->GetNeighbors()->MapSequential(map_f);
  }

  for (const auto& [v, index] : subgraph_id) {
    if (index < num_in_partition) {
      auto size_v = CurrentClusterSize(v);
      subgraph->SetNodeWeight(index, size_v);
    } else {
      subgraph->SetNodeWeight(index, -1);
    }
  }

  ASSIGN_OR_RETURN(auto results,
                   ApproximateSubgraphHac(std::move(subgraph),
                                          min_merge_similarities, epsilon));

  auto& merges = results.merges;

  uintE last_id = num_nodes;
  // Maps nodes in the range [num_nodes, ...] to their original ids.
  // Preserve the smallest original node in the cluster as the representative.
  absl::flat_hash_map<uintE, uintE> internal_nodes_to_orig;
  parlay::sequence<std::tuple<uintE, uintE, float>> remapped_merges;
  for (auto [u, v, sim] : merges) {
    // Map back to the original node. If the id is >= num_nodes, then it is
    // an internal node, i.e., the result of a merge, and it will be present in
    // internal_nodes_to_orig.
    uintE orig_u = (u < num_nodes) ? original_id[u] : internal_nodes_to_orig[u];
    uintE orig_v = (v < num_nodes) ? original_id[v] : internal_nodes_to_orig[v];

    // Store the new internal node in internal_nodes_to_orig.
    uintE next_id = last_id;
    internal_nodes_to_orig[next_id] = std::min(orig_u, orig_v);
    ++last_id;

    // Update min_merge_similarities for this merge.
    auto merge_to = std::min(orig_u, orig_v);
    auto merge_from = std::max(orig_u, orig_v);
    all_min_merge_similarities[merge_to] =
        std::min(all_min_merge_similarities[merge_from],
                 std::min(all_min_merge_similarities[merge_to], (float)sim));

    ABSL_CHECK_NE(orig_u, orig_v);  // No self-loops, or self-merges.
    remapped_merges.push_back({merge_to, merge_from, sim});
  }

  return remapped_merges;
}

// Returns a sequence of pairs where each pair is the (cluster_id,
// node_id) of an active node.
template <class ClusteredGraph>
inline parlay::sequence<std::pair<gbbs::uintE, gbbs::uintE>>
SizeConstrainedAffinity(ClusteredGraph& CG,
                        const parlay::sequence<gbbs::uintE>& active,
                        size_t max_size_constraint, size_t round) {
  size_t n = CG.NumNodes();
  auto UF = AsynchronousUnionFind<gbbs::uintE>(n);

  auto best_nghs = parlay::tabulate(active.size(), [&](gbbs::uintE i) {
    return std::make_pair(i, float{0});
  });
  parlay::parallel_for(0, active.size(), [&](size_t i) {
    gbbs::uintE v = active[i];
    auto [best_ngh, best_wgh] = CG.MutableNode(v)->BestEdge();
    best_nghs[i] = std::make_pair(best_ngh, best_wgh);
    UF.Unite(v, best_ngh);
  });

  auto cluster_wgh_node =
      parlay::sequence<std::tuple<gbbs::uintE, float, gbbs::uintE>>::
          from_function(active.size(), [&](size_t i) {
            gbbs::uintE v = active[i];
            gbbs::uintE l_v = UF.Find(v);
            auto best_wgh = best_nghs[i].second;
            return std::make_tuple(l_v, -best_wgh, v);
          });
  parlay::sort_inplace(cluster_wgh_node);

  auto cluster_starts =
      parlay::pack_index(parlay::delayed_tabulate(active.size(), [&](size_t i) {
        return (i == 0) || (std::get<0>(cluster_wgh_node[i]) !=
                            std::get<0>(cluster_wgh_node[i - 1]));
      }));

  auto SizeConstrainedUF = AsynchronousUnionFind<gbbs::uintE>(n);
  auto sizes = parlay::sequence<gbbs::uintE>(n, 1);

  auto num_clusters = cluster_starts.size();

  parlay::parallel_for(
      0, num_clusters,
      [&](size_t i) {
        size_t begin = cluster_starts[i];
        size_t end =
            (i == num_clusters - 1) ? active.size() : cluster_starts[i + 1];
        auto cluster_size = end - begin;

        // Note that we are processing all merges trying to merge into this
        // component sequentially, so there are no races. Parallelizing this
        // would be interesting.
        for (size_t j = 0; j < cluster_size; ++j) {
          auto [l_v, b_w, v] = cluster_wgh_node[begin + j];
          auto [w, wgh] = CG.MutableNode(v)->BestEdge();

          auto cw = SizeConstrainedUF.Find(w);
          auto cv = SizeConstrainedUF.Find(v);

          size_t cv_size = sizes[cv];
          size_t cw_size = sizes[cw];

          if (cv_size + cw_size <= max_size_constraint) {
            SizeConstrainedUF.Unite(cv, cw);
          }
          auto min_id = SizeConstrainedUF.Find(cv);
          sizes[min_id] = cv_size + cw_size;
        }
      },
      1);

  auto cluster_and_active = parlay::tabulate(active.size(), [&](size_t i) {
    gbbs::uintE v = active[i];
    return std::make_pair(SizeConstrainedUF.Find(v), v);
  });

  return cluster_and_active;
}

}  // namespace graph_mining::in_memory
#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_HAC_TERAHAC_INTERNAL_H_
