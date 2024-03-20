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

#include <algorithm>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/hac/subgraph/approximate_subgraph_hac_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {
namespace {

using graph_mining::in_memory::Dendrogram;
using Clustering = InMemoryClusterer::Clustering;
using NodeId = InMemoryClusterer::NodeId;

bool IsActive(double node_weight) { return node_weight >= 0; }

}  // namespace

// Nodes are either active or inactive.
// - Active nodes are nodes assigned to this subgraph can be potentially
//   clustered in this call by merging a (1+epsilon)-good edge.
// - Inactive nodes are nodes not in this subgraph, and are thus not mergeable.
//
// Weights in the input graph are the average-linkage weight at the start of the
// round.
// The min_merge_similarities vector stores the smallest similarity over all
// merges that occurred to create it.
absl::StatusOr<SubgraphHacResults> ApproximateSubgraphHac(
    std::unique_ptr<SimpleUndirectedGraph> graph,
    std::vector<double> min_merge_similarities, double epsilon) {
  NodeId num_nodes = graph->NumNodes();

  // A vector of bools storing the active status of nodes.
  std::vector<bool> is_active(graph->NumNodes());
  for (NodeId id = 0; id < num_nodes; id++) {
    is_active[id] = IsActive(graph->NodeWeight(id));
  }
  // Save initial active status so that we don't emit dendrogram values for
  // the inactive nodes (nodes not in this subgraph).
  auto is_initially_active = is_active;

  for (NodeId i = 0; i < num_nodes; ++i) {
    // Set the node weight (the cluster size) of an inactive node to 1. This is
    // safe because (1) weights on the input graph are the average-linkage
    // weights, i.e., they are already normalized by the product of the two
    // endpoint's sizes; (2) an inactive node never participates in a merge, and
    // therefore never has its cluster size change.
    //
    // So if an inactive node C has two neighbors A and B which merge, the new
    // weight of the edge will be [w(A,C)*|A| + w(B,C)*|B|] / [|A| + |B|] and
    // since w(A,C), w(B, C) were already normalized by a factor of 1/|I|, the
    // resulting weight will be correct.
    graph->SetNodeWeight(i, is_active[i] ? graph->NodeWeight(i) : 1);
  }

  // Build a Hac graph that supports efficiently extracting (1+eps)-good edges
  // from the subgraph and performing merges.
  auto subgraph = std::make_unique<ApproximateSubgraphHacGraph>(
      *graph, num_nodes, epsilon, epsilon / 2, std::move(is_active),
      min_merge_similarities);

  // Structure required to emit a dendrogram:
  // Note that we may need to emit up to 2*num_nodes - 1 nodes in the dendrogram
  // for this subgraph.
  Dendrogram dendrogram(num_nodes);

  // Map from an active node id to its current cluster id. Initially just the
  // identity mapping.
  std::vector<NodeId> to_cluster_id(num_nodes);
  absl::c_iota(to_cluster_id, 0);

  std::vector<std::tuple<NodeId, NodeId, double>> merges;
  size_t num_merges = 0;
  while (true) {
    NodeId node_a, node_b;
    double goodness_ab;
    // Extract an edge with goodness <= (1+epsilon).
    std::tie(node_a, node_b, goodness_ab) = subgraph->GetGoodEdge();

    // If this edge is invalid, quit.
    if (goodness_ab == ApproximateSubgraphHacGraph::kDefaultGoodness) {
      break;
    }

    // The good edge returned by the graph can only go between active nodes.
    ABSL_CHECK(subgraph->IsActive(node_a));
    ABSL_CHECK(subgraph->IsActive(node_b));

    ABSL_VLOG(1) << "Merging: node_a = " << node_a << " node_b = " << node_b
                 << " goodness = " << goodness_ab << " epsilon = " << epsilon;

    // Since the goodness values we use are upper-bounds on the true goodness
    // values, we're guaranteed that this is a good edge to merge.
    double merge_similarity = subgraph->EdgeWeight(node_a, node_b);
    NodeId node_to;
    ASSIGN_OR_RETURN(node_to,
                     subgraph->Merge(&dendrogram, &to_cluster_id,
                                     &min_merge_similarities, node_a, node_b));
    NodeId node_from = (node_to == node_a) ? node_b : node_a;

    // Update the dendrogram.
    NodeId cluster_a = to_cluster_id[node_from];
    NodeId cluster_b = to_cluster_id[node_to];
    auto new_id =
        dendrogram.BinaryMerge(cluster_a, cluster_b, merge_similarity);
    if (new_id.ok()) {
      to_cluster_id[node_to] = new_id.value();
    } else {
      return new_id.status();
    }

    // Update the min-merge similarities
    min_merge_similarities[node_to] =
        std::min(min_merge_similarities[node_to],
                 std::min(min_merge_similarities[node_from], merge_similarity));

    num_merges++;
    // The smaller cluster id is the first element. This makes testing easier.
    merges.push_back(std::make_tuple(std::min(cluster_a, cluster_b),
                                     std::max(cluster_a, cluster_b),
                                     merge_similarity));
  }

  // Generate a clustering of all the nodes from the dendrogram. We use
  // FlattenSubtreeClustering since the dendrogram that we return may be
  // non-monotone.
  Clustering clustering;
  ASSIGN_OR_RETURN(clustering, dendrogram.FlattenSubtreeClustering(0));

  // Remove originally inactive nodes from clustering.
  for (auto it = clustering.begin(); it != clustering.end(); ++it) {
    it->erase(std::remove_if(it->begin(), it->end(),
                             [&](auto id) { return !is_initially_active[id]; }),
              it->end());
  }
  clustering.erase(
      std::remove_if(clustering.begin(), clustering.end(),
                     [&](const auto& cluster) { return cluster.empty(); }),
      clustering.end());

  return SubgraphHacResults(std::move(merges), std::move(dendrogram),
                            std::move(subgraph), std::move(clustering));
}

}  // namespace graph_mining::in_memory
