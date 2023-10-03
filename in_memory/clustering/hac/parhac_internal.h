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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_INTERNAL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_INTERNAL_H_

#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gbbs/macros.h"
#include "utils/math.h"

namespace graph_mining::in_memory {

// This file provides internal functions that are used by the ParHac algorithm
// (see parhac.cc). This text provides some high-level notes about the
// the internal functions and algorithm.
// - In each round of the algorithm, it operates on a subset of *alive* nodes,
//   which are nodes that have an incident edge with weight at least equal to
//   the current lower_threshold. Note that this is distinct from the IsActive
//   function for a ClusteredGraph node, which indicates whether the node has
//   already been clustered, or not.
// - We store a sequence containing the node sizes in the clustered graph so
//   that we can perform fetch-and-adds on this sequence. Otherwise, we would
//   have to store these values in the clustered graph representation, which
//   would leak some implementation details about ParHac into the clustered
//   graph object.

namespace internal {

constexpr gbbs::uintE kMaxUintE = std::numeric_limits<gbbs::uintE>::max();

enum class NodeColor {
  kNoColor,
  kBlue,
  kRed,
};

// Check whether x >= y, also returning true if x and y are approximately the
// same floating point value.
inline bool GreaterThanEq(double x, double y) {
  return x >= y || AlmostEquals(x, y);
}

// Identify all nodes that are alive in this bucket, i.e., have an incident
// edge with similarity at least the lower_threshold and are active using the
// GreaterThanEq function defined above. We then pack out nodes that are still
// alive into an array of ids. The return value is a std::pair representing the
// set of alive nodes in the (1) sparse format (a sequence of node identifiers)
// and (2) the dense format (a sequence of boolean values indicating whether the
// i-th node is alive).
template <class ClusteredGraph>
std::pair<parlay::sequence<gbbs::uintE>, parlay::sequence<bool>> GetAliveNodes(
    const ClusteredGraph& clustered_graph, double lower_threshold) {
  gbbs::uintE num_nodes = clustered_graph.NumNodes();
  parlay::sequence<bool> alive_dense_seq(num_nodes, false);
  parlay::parallel_for(0, num_nodes, [&](size_t i) {
    if (clustered_graph.ImmutableNode(i).IsActive()) {
      auto [id, similarity] = clustered_graph.ImmutableNode(i).BestEdge();
      if (id != kMaxUintE && GreaterThanEq(similarity, lower_threshold)) {
        alive_dense_seq[i] = true;
      }
    }
  });
  return std::make_pair(
      parlay::pack_index<gbbs::uintE>(parlay::make_slice(alive_dense_seq)),
      std::move(alive_dense_seq));
}

// Go over the nodes in alive_nodes. For each of these nodes, check whether the
// node is still alive by testing if it still has an incident edge with
// similarity at least lower_threshold. This function side-effects
// alive_nodes_seq to set it to contain only the nodes that are still alive
// based on this predicate. The function also side affects sequences used by the
// algorithm to reset data for each node (specifically, the nodes that are
// initially alive). For each alive node, this involves (1) setting the node's
// color to kNodeColor, (2) setting the merge target to the nullary value
// indicating that the node has no merge target, and (3) resetting the current
// cluster size to the node's current cluster size. Lastly, the function
// side-effects alive_dense_seq, which is a dense representation of the alive
// nodes (i.e., it is a boolean sequence of length n with one entry per node
// indicating whether the i-th node is alive), so that the invariant
// [PackIndex(alive_dense_seq) = alive_nodes_seq] holds. This dense sequence is
// used to save whether a node is alive to avoid computing BestEdge on a node
// twice.
template <class ClusteredGraph>
void UpdateAliveNodesAndResetSequences(
    const ClusteredGraph& clustered_graph, double lower_threshold,
    parlay::sequence<gbbs::uintE>& alive_nodes_seq,
    parlay::sequence<bool>& alive_dense_seq,
    parlay::sequence<NodeColor>& colors_seq,
    parlay::sequence<gbbs::uintE>& cluster_sizes_seq,
    parlay::sequence<std::pair<gbbs::uintE, float>>& merge_target_seq) {
  // Reset colors.
  // Recompute nodes that are alive and update alive_nodes.
  parlay::parallel_for(0, alive_nodes_seq.size(), [&](size_t i) {
    gbbs::uintE node_index = alive_nodes_seq[i];
    ABSL_CHECK(alive_dense_seq[node_index]);
    bool is_alive = false;
    if (clustered_graph.ImmutableNode(node_index).IsActive()) {
      auto [id, similarity] =
          clustered_graph.ImmutableNode(node_index).BestEdge();
      if (id != internal::kMaxUintE &&
          GreaterThanEq(similarity, lower_threshold)) {
        is_alive = true;
      }
      // Reset current cluster size.
      cluster_sizes_seq[node_index] =
          clustered_graph.ImmutableNode(node_index).ClusterSize();
    }
    // Update alive.
    alive_dense_seq[node_index] = is_alive;
    // Reset color.
    colors_seq[node_index] = NodeColor::kNoColor;
    // Reset merge target.
    merge_target_seq[node_index] = std::make_pair(internal::kMaxUintE, float());
  });

  // Update alive_nodes nodes.
  alive_nodes_seq =
      parlay::filter(make_slice(alive_nodes_seq),
                     [&](gbbs::uintE u) { return alive_dense_seq[u] > 0; });
}

// Given a node node_index and the current {clustered_graph, lower_threshold,
// epsilon} and associated sequences tracking the colors of vertices
// (colors_seq) and cluster sizes (cluster_sizes_seq), this function tries to
// non-deterministically merge the blue node to one of its red neighbors. The
// function side-effects cluster_sizes_seq. Please see the comments below for
// more details on how the merge decision is made for each neighbor of
// node_index. The function returns an optional value that holds the id of the
// neighbor to merge to, and the merge similarity, if it successfully found a
// merge.
template <class ClusteredGraph>
std::optional<std::pair<gbbs::uintE, float>> TryToMergeNode(
    gbbs::uintE node_index, const ClusteredGraph& clustered_graph,
    double lower_threshold, double epsilon,
    const parlay::sequence<NodeColor>& colors_seq,
    parlay::sequence<gbbs::uintE>& cluster_sizes_seq) {
  using uintE = gbbs::uintE;
  std::optional<std::pair<uintE, float>> merge_target;
  if (colors_seq[node_index] == NodeColor::kBlue) {
    // The function try_merge_neighbor_f is applied to each incident
    // edge of the alive node node_index. For each (neighbor_index, similarity)
    // incident edge (call the neighbor "v"), the function merges into v if (1)
    // v is a red neighbor and (2) the size of v's cluster has not yet grown by
    // more than a (1+epsilon) factor since the start of this round. This
    // iterator function returns true if it has successfully merged (and thus
    // does not need to iterate further) and false otherwise.
    auto try_merge_neighbor_f = [&](uintE neighbor_index, double similarity) {
      if (internal::GreaterThanEq(similarity, lower_threshold) &&
          colors_seq[neighbor_index] == NodeColor::kRed) {
        // Note that it's important to use the ClusterSize method from the
        // clustered_graph here, and not cluster_sizes_seq[neighbor_index].
        // The reason is that ClusterSize() represents the size at the beginning
        // of hte round, whereas the value stored in cluster_sizes_seq can
        // include earlier merges made in the same round. In more detail,
        // reading cluster_sizes_seq[neighbor_index] can cause a read/write race
        // due to other neighbors of neighbor_index calling fetch-and-add on
        // this value concurrently. This could cause us to end up overshooting
        // the actual capacity constraint on neighbor_index's size in this
        // round.
        uintE neighbor_cur_size =
            clustered_graph.ImmutableNode(neighbor_index).ClusterSize();
        uintE upper_bound = (1 + epsilon) * neighbor_cur_size;
        uintE our_size =
            clustered_graph.ImmutableNode(node_index).ClusterSize();
        auto opt = gbbs::fetch_and_add_threshold(
            &(cluster_sizes_seq[neighbor_index]), our_size, upper_bound);
        if (opt.has_value()) {  // Success in the fetch-and-add.
          merge_target = {neighbor_index, similarity};
          return true;  // Done iterating.
        }
      }
      return false;  // Keep iterating.
    };
    // Note that if the order of the underlying container storing the
    // node neighbors is sorted, then iterating over the neighbors in a
    // sorted order using try_merge_neighbor_f (above) could cause
    // issues with contention, since high-degree nodes with low vertex
    // ids would receive a large amount of compare-and-swaps.
    clustered_graph.ImmutableNode(node_index)
        .IterateUntil(try_merge_neighbor_f);
  }
  return merge_target;
}

// Extract the merges that will be performed in this round. Some merges stored
// in merge_target_seq are not valid (their first component is kMaxUintE), so we
// filter these out in the returned sequence of merge triples with format
// (node_id, node_id, merge_similarity)
inline parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, float>> GetMerges(
    parlay::sequence<gbbs::uintE>& alive_nodes_seq,
    parlay::sequence<std::pair<gbbs::uintE, float>>& merge_target_seq) {
  auto all_merges =
      parlay::delayed_seq<std::tuple<gbbs::uintE, gbbs::uintE, float>>(
          alive_nodes_seq.size(), [&](size_t i) {
            gbbs::uintE node_index = alive_nodes_seq[i];
            auto [center_id, merge_similarity] = merge_target_seq[node_index];
            return std::make_tuple(center_id, node_index, merge_similarity);
          });
  return parlay::filter(all_merges, [&](const auto& tup) {
    return std::get<0>(tup) != internal::kMaxUintE;
  });
}

}  // namespace internal

// Given a clustered graph H and a similarity threshold, lower_threshold, this
// function processes a "bucket" in the HAC algorithm (conceptually just a set
// of edges with similarity at least the lower_threshold) using a randomized
// merge algorithm. The algorithm first identifies alive nodes, which have and
// incident edge with similarity at least lower_threshold. It then processes the
// alive nodes over a sequence of rounds. In each round, it colors the alive
// nodes red or blue randomly, and atomically assigns blue nodes to their red
// neighbors if the red neighbor can accept this merge without violating the
// (1+epsilon) approximation guarantee for the average-linkage function.
//
// In more detail, suppose a node v with cluster size |v| is alive and colored
// red in the current round. Then we can imagine sequentially merging v with its
// blue neighbors until the first merge that causes v's cluster size to grow
// larger than (1+epsilon)*|v|. At this point, v becomes ineligible for further
// merges in this round. It may become eligible again in the next round if after
// performing these merges, it still has an incident edge with weight at least
// lower_threshold.
//
// The function guarantees that all merges performed are (1+epsilon)^2 close.
// approximate. We also ensure upon exit that H has no edges with weight >=
// lower_threshold by running the merge procedure in a loop over multiple
// rounds.
//
// The last argument is epsilon, the accuracy parameter.
//
// The function returns the number of clusters merged by the routine, and the
// total number of rounds taken to process the bucket.
template <class ClusteredGraph>
std::pair<size_t, size_t> ProcessHacBucketRandomized(
    ClusteredGraph& clustered_graph, double lower_threshold, double epsilon) {
  using uintE = gbbs::uintE;
  using internal::NodeColor;

  // TODO: store these sequences as private class members to avoid
  // potentially expensive reinitialization for each bucket call.
  size_t num_nodes = clustered_graph.NumNodes();
  auto colors_seq = parlay::sequence<NodeColor>(num_nodes, NodeColor::kNoColor);
  auto merge_target_seq = parlay::sequence<std::pair<uintE, float>>(
      num_nodes, std::make_pair(internal::kMaxUintE, float()));
  auto cluster_sizes_seq = parlay::sequence<uintE>::from_function(
      num_nodes,
      [&](size_t i) { return clustered_graph.ImmutableNode(i).ClusterSize(); });
  parlay::sequence<uintE> alive_nodes_seq;
  parlay::sequence<bool> alive_dense_seq;
  std::tie(alive_nodes_seq, alive_dense_seq) =
      internal::GetAliveNodes(clustered_graph, lower_threshold);

  size_t num_merged = 0;
  size_t inner_rounds = 0;

  // Process this bucket while at least one node is still alive in it.
  while (alive_nodes_seq.size() > 1) {
    // Generate colors for all active nodes.
    constexpr size_t kBlockSize = 2048;
    size_t num_blocks =
        parlay::internal::num_blocks(alive_nodes_seq.size(), kBlockSize);
    parlay::parallel_for(
        0, num_blocks,
        [&](size_t block_idx) {
          size_t start = block_idx * kBlockSize;
          size_t end = std::min(start + kBlockSize, alive_nodes_seq.size());
          // Note: absl::BitGen should be replaced with a more scalable 
          // (shared) random bit-generation mechanism for good performance.
          absl::BitGen gen;
          for (size_t i = start; i < end; ++i) {
            uintE node_index = alive_nodes_seq[i];
            colors_seq[node_index] = (absl::Uniform(gen, 0, 2) == 1)
                                         ? NodeColor::kRed
                                         : NodeColor::kBlue;
          }
        },
        /*granularity=*/1);

    parlay::parallel_for(
        0, alive_nodes_seq.size(),
        [&](size_t i) {
          uintE node_index = alive_nodes_seq[i];
          std::optional<std::pair<uintE, float>> merge_target =
              TryToMergeNode(node_index, clustered_graph, lower_threshold,
                             epsilon, colors_seq, cluster_sizes_seq);
          if (merge_target.has_value()) {
            merge_target_seq[node_index] = merge_target.value();
          }
        },
        1);

    auto merges = internal::GetMerges(alive_nodes_seq, merge_target_seq);

    num_merged += merges.size();

    // Perform the merges in H.
    clustered_graph.StarMerge(merges);

    // Reset colors.
    // Recompute the nodes that are alive and update alive_nodes.
    UpdateAliveNodesAndResetSequences(
        clustered_graph, lower_threshold, alive_nodes_seq, alive_dense_seq,
        colors_seq, cluster_sizes_seq, merge_target_seq);

    inner_rounds++;
  }

  return {num_merged, inner_rounds};
}

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARHAC_INTERNAL_H_
