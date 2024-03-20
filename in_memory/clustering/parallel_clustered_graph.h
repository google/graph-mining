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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_H_

#include <functional>
#include <limits>
#include <optional>

#include "absl/log/absl_check.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/parallel_clustered_graph_internal.h"
#include "in_memory/clustering/parallel_dendrogram.h"
#include "in_memory/clustering/types.h"

namespace graph_mining {
namespace in_memory {

// Weight is a type specifying the linkage function (please see
// parallel-clustered-graph-internal.h for more details). This class represents
// a cluster of nodes in a contracted graph and the neighborhood information of
// the cluster of nodes.
template <class Weight>
class ClusteredNode {
 public:
  using uintE = gbbs::uintE;

  ClusteredNode()
      : current_node_id_(kInvalidNodeId),
        current_cluster_id_(kInvalidNodeId),
        num_in_cluster_(0),
        active_(false) {}

  explicit ClusteredNode(uintE id)
      : current_node_id_(id),
        current_cluster_id_(id),
        num_in_cluster_(1),
        active_(true) {}

  // We distinguish between active and inactive clusters. All clusters are
  // initially active, and clusters become inactive when they are merged to
  // another cluster.
  bool IsActive() const { return active_; }

  // Returns pointer to the underlying neighbors (provides primitives for
  // mapping over, updating edges, etc.).
  HashBasedNeighborhood<Weight>* GetNeighbors() { return &neighbors_; }
  // Returns an immutable handle to the underlying neighbors.
  const HashBasedNeighborhood<Weight>& GetImmutableNeighbors() const {
    return neighbors_;
  }

  // Number of nodes contained in this cluster.
  uintE ClusterSize() const { return num_in_cluster_; }

  // Returns the neighbor-id and similarity of the "best" incident
  // edge. If the node has no neighbors, the returned id is
  // std::numeric_limits<uintE>::max().
  std::pair<uintE, double> BestEdge() const {
    using EdgePair = std::pair<uintE, Weight>;
    auto map_f = [&](uintE neighbor, Weight weight) -> EdgePair {
      ABSL_CHECK_NE(neighbor,
                    current_node_id_);  // Consistency check: no self-loops.
      return {neighbor, weight};
    };
    auto reduce_f = [&](const EdgePair& l, const EdgePair& r) -> EdgePair {
      return Weight::Smaller(l.second, r.second) ? r : l;
    };
    EdgePair id = {std::numeric_limits<uintE>::max(), Weight()};
    auto reduce = parlay::make_monoid(reduce_f, id);
    auto [neighbor_id, weight] = neighbors_.MapReduce(map_f, reduce);
    double similarity = weight.Similarity(ClusterSize());
    return {neighbor_id, similarity};
  }

  // Apply the predicate function pred on each of the underlying neighbors until
  // it returns true. The pred function takes as arguments the neighbor id and
  // similarity of the neighbor edge, and returns a bool.
  void IterateUntil(std::function<bool(uintE, double)> pred) const {
    uintE our_size = ClusterSize();
    auto fn = [&](uintE v, Weight weight) {
      return pred(v, weight.Similarity(our_size));
    };
    GetImmutableNeighbors().IterateUntil(fn);
  }

  // The current node id of this cluster. This id is equal to its initial id
  // (set in the constructor above) until the node is merged to another node in
  // the graph using the LogicallyMergeWithParent function below (at which point
  // the node id is set to the merged node's id). Note that these ids refer to
  // the *node* ids which range between [0, n), and not the *cluster* ids, which
  // range between [0, 2*n-1). Note that the new clusters we create (clusters
  // with size larger than 1) will have ids in [n, 2*n-1).
  uintE CurrentNodeId() const { return current_node_id_; }

  // The current cluster id of this node. This value ranges between [0, 2*n-1)
  // and is updated every time the cluster participates in a merge and stays
  // active.
  uintE CurrentClusterId() const { return current_cluster_id_; }

  void SetClusterId(uintE new_cluster_id) {
    current_cluster_id_ = new_cluster_id;
  }

  // This function deactivates this node and *logically* merges the node with
  // the parent node. Specifically, we update the parent's cluster size based on
  // this node's cluster size and set this node's current node id to the
  // parent's node id. Note that this function does not handle relabling and
  // merging this node's neighbors with the parent's neighbors.
  void LogicallyMergeWithParent(ClusteredNode<Weight>* parent) {
    ABSL_CHECK(IsActive());
    // Update parent's size based on child's.
    parent->num_in_cluster_ += ClusterSize();
    // Deactivate child and update its current id.
    active_ = false;
    current_node_id_ = parent->CurrentNodeId();
  }

  void Clear() { neighbors_.Clear(); }

  void SetClusterSize(uintE new_size) { num_in_cluster_ = new_size; }

  // Return the similarity of the edge (this node, `neighbor_id`). Returns an
  // empty optional if `neighbor_id` is not in the neighborhood.
  std::optional<typename Weight::StoredWeightType> EdgeSimilarity(
      uintE neighbor_id) const {
    const auto similarity = neighbors_.FindValue(neighbor_id);
    if (!similarity.has_value()) {
      return std::nullopt;
    }
    return similarity.value().Similarity(ClusterSize());
  }

 private:
  // The current node id of this cluster, updated upon a merge that keeps this
  // cluster active. This field remains unchanged until the node is deactivated,
  // at which point it points to the node id of the node it merges with.
  uintE current_node_id_;
  // The cluster id of this node (a value between [0, 2*n-1)). If a cluster
  // remains active after a merge, this field is updated with a new cluster id
  // representing the newly merged cluster.
  uintE current_cluster_id_;
  // Number of nodes contained in this cluster.
  uintE num_in_cluster_;

  // active = false iff this cluster is no longer active.
  bool active_;
  // Simple hash-based representation of a node's neighborhood
  HashBasedNeighborhood<Weight> neighbors_;
  static constexpr gbbs::uintE kInvalidNodeId =
      std::numeric_limits<gbbs::uintE>::max();
};

// Interface for representing an undirected similarity graph over a sequence
// of contractions. Weight is an object representing a linkage function (a way
// to "merge" edge weights). More detail about the Weight object can be found
// in parallel-clustered-graph-internal.h SimilarityGraph is a graph object
// supporting the GBBS interface (e.g., symmetric_ptr_graph).
template <class Weight, class SimilarityGraph>
class ClusteredGraph {
 public:
  using uintE = gbbs::uintE;
  using ClusterNode = ClusteredNode<Weight>;
  using StoredWeightType = typename Weight::StoredWeightType;
  using Edge = std::tuple<uintE, uintE, StoredWeightType>;
  using NodeId = graph_mining::in_memory::NodeId;

  size_t NumNodes() const { return num_nodes_; }

  // Construct a ClusteredGraph given the underlying weighted
  // similarity graph G.
  explicit ClusteredGraph(SimilarityGraph* base_graph)
      : base_graph_(base_graph),
        num_nodes_(base_graph_->n),
        dendrogram_(ParallelDendrogram(num_nodes_)) {
    using W = typename SimilarityGraph::weight_type;
    clusters_ = parlay::sequence<ClusterNode>::from_function(
        NumNodes(), [&](size_t i) { return ClusterNode(i); });
    // Allocate sufficient space for each node initially
    parlay::parallel_for(0, NumNodes(), [&](size_t i) {
      auto neighbors = clusters_[i].GetNeighbors();
      uintE out_degree = base_graph_->get_vertex(i).out_degree();
      if (out_degree > 0) {
        neighbors->AdjustSizeForIncoming(out_degree);
      }
      auto update_f = [&](Weight* weight) {};
      auto map_f = [&](uintE u, uintE v, W cut_weight) {
        Weight weight(cut_weight);
        ABSL_CHECK_NE(u, v);
        neighbors->InsertOrUpdate(v, weight, update_f);
      };
      base_graph_->get_vertex(i).out_neighbors().map(map_f);
    });
    // The next available cluster id is num_nodes_.
    last_unused_id_ = num_nodes_;
  }

  ClusterNode* MutableNode(uintE i) { return &(clusters_[i]); }
  const ClusterNode& ImmutableNode(uintE i) const { return clusters_[i]; }

  // Given merge_seq, a sequence of (u,v) tuples representing the merge of
  // satellites v with a center u, perform all merges and update the internal
  // clustered graph representation. Note the following requirements:
  // - All node ids in merge_seq are pairwise distinct.
  // - There can potentially be multiple v's (satellites) that are merging with
  //   a single u (center).
  void StarMerge(parlay::sequence<std::tuple<uintE, uintE, float>> merge_seq);

  // Returns the dendrogram object built over the course of merges.
  // Please see the comment on the GetClustering function in
  // parallel-dendrogram.h for more details about methods that the dendrogram
  // supports.
  const ParallelDendrogram* GetDendrogram() const { return &dendrogram_; }

 private:
  // The input is a sequence of sorted directed edge triples that correspond to
  // new edges to insert. The i-th edge is a triple (u,v,weight) indicating that
  // u should receive a new edge to v with the given weight. This function
  // performs all insertions provided in inserts. Note that the edges in the
  // input are *directed*, so a (u,v,weight) triple will only update u's
  // neighbor list.
  void InsertTriples(parlay::sequence<Edge> inserts) {
    auto all_starts =
        parlay::delayed_seq<size_t>(inserts.size(), [&](size_t i) {
          if ((i == 0) ||
              std::get<0>(inserts[i]) != std::get<0>(inserts[i - 1])) {
            return i;
          }
          return std::numeric_limits<size_t>::max();
        });
    // Index (start) of the edges being inserted to an active node.
    auto starts = parlay::filter(all_starts, [&](size_t v) {
      return v != std::numeric_limits<size_t>::max();
    });

    // In parallel, for each active node receiving edges:
    //   1. Resize the neighbor-list if necessary
    //   2. Insert all incoming edges
    parlay::parallel_for(0, starts.size(), [&](size_t i) {
      size_t start = starts[i];
      size_t end = (i == starts.size() - 1) ? inserts.size() : starts[i + 1];
      size_t num_edges = end - start;

      uintE source_id = std::get<0>(inserts[start]);
      auto& cluster = clusters_[source_id];
      ABSL_CHECK(cluster.IsActive());
      // Resize the node's neighbors if necessary before inserting. The capacity
      // (n) stored in each neighbor-list may be incorrect after performing
      // deletions, so we also recompute the size.
      cluster.GetNeighbors()->AdjustSizeForIncoming(num_edges);

      // Insert all edges for this node into our hash-table
      for (size_t j = start; j < end; j++) {
        auto neighbor_id = std::get<1>(inserts[j]);
        StoredWeightType merged_weight_data = std::get<2>(inserts[j]);
        ABSL_CHECK_NE(neighbor_id, source_id);
        // Note that for average-link and cut-sparsity, the cluster size
        // information for the stored weight is (temporarily) incorrect, but
        // will be fixed in the next step of the merge algorithm.
        Weight weight(merged_weight_data);
        auto update_f = [&](Weight* weight) {
          weight->UpdateWeight(merged_weight_data);
        };
        cluster.GetNeighbors()->InsertOrUpdate(neighbor_id, weight, update_f);
      }
    });
  }

  // The input to this function is a sorted sequence of merges, merge_seq (see
  // the comment for StarMerge above for more details), and a sequence of
  // starts specifying the index of the start of merges for the i-th center.
  // The function maps over all centers in the merge_seq. For each center c, we
  // map over its neighbors, n(c), and perform an update on this edge on both
  // endpoints. This function is only called for Weight parameters that require
  // either Weight::kRequiresNeighborUpdate or Weight::kRequiresEndpointsUpdate,
  // which means that after inserting new edges, the edge requires state from
  // either the neighbor endpoint's cluster size, or both endpoints' cluster
  // sizes, respectively.
  void UpdateNeighbors(
      parlay::sequence<std::tuple<uintE, uintE, float>>& merge_seq,
      parlay::sequence<size_t>& starts) {
    parlay::parallel_for(0, starts.size(), [&](size_t i) {
      ABSL_CHECK(Weight::kRequiresNeighborUpdate ||
                 Weight::kRequiresEndpointsUpdate);
      size_t start = starts[i];
      uintE center_id = std::get<0>(merge_seq[start]);
      uintE center_size = clusters_[center_id].ClusterSize();

      // Map over all of this center's neighbors and update the size
      // of center in their neighborhood.
      auto map_f = [&](uintE neighbor_id, Weight& weight) {
        auto& neighbor_cluster = clusters_[neighbor_id];
        ABSL_CHECK(neighbor_cluster.IsActive());

        if constexpr (Weight::kRequiresEndpointsUpdate) {
          uintE neighbor_size = neighbor_cluster.ClusterSize();
          // Update our weight.
          weight.UpdateEndpointsSize(center_size, neighbor_size);
          // Update the neighbor's weight.
          auto update_f = [&](Weight* weight) {
            weight->UpdateEndpointsSize(center_size, neighbor_size);
          };
          neighbor_cluster.GetNeighbors()->Update(center_id, update_f);
        }
        if constexpr (Weight::kRequiresNeighborUpdate) {
          // Update our weight.
          weight.UpdateNeighborSize(clusters_[neighbor_id].ClusterSize());
          // Update the neighbor's weight.
          auto update_f = [&](Weight* weight) {
            weight->UpdateNeighborSize(center_size);
          };
          neighbor_cluster.GetNeighbors()->Update(center_id, update_f);
        }
      };
      clusters_[center_id].GetNeighbors()->Map(map_f);
    });
  }

  SimilarityGraph* base_graph_;
  size_t num_nodes_;
  size_t last_unused_id_;

  parlay::sequence<ClusterNode> clusters_;
  ParallelDendrogram dendrogram_;
};

// Used to infer the second template argument.
template <class Weight, class SimilarityGraph>
auto BuildClusteredGraph(SimilarityGraph* G) {
  return ClusteredGraph<Weight, SimilarityGraph>(G);
}

template <class Weight, class SimilarityGraph>
void ClusteredGraph<Weight, SimilarityGraph>::StarMerge(
    parlay::sequence<std::tuple<uintE, uintE, float>> merge_seq) {
  // Sort merges based on the centers (note that a semi-sort is sufficient).
  auto get_key = [](const std::tuple<uintE, uintE, float>& x) -> uintE {
    return std::get<0>(x);
  };
  parlay::integer_sort_inplace(make_slice(merge_seq), get_key);

  // Identify the start of each center's merges.
  auto all_starts =
      parlay::delayed_seq<size_t>(merge_seq.size(), [&](size_t i) {
        if ((i == 0) ||
            std::get<0>(merge_seq[i]) != std::get<0>(merge_seq[i - 1])) {
          return i;
        }
        return std::numeric_limits<size_t>::max();
      });
  // The start of every center's list of satellites
  auto starts = parlay::filter(all_starts, [&](size_t v) {
    return v != std::numeric_limits<size_t>::max();
  });
  auto edge_sizes = parlay::sequence<size_t>(merge_seq.size());

  // In parallel over every component:
  // - Logically perform all of the merges: updates the clusters_ for all
  //   satellites, setting their parent to be the center, and then deactivate
  //   the satellite cluster.
  // - Update the cluster-id of all clusters that remain active.
  // TODO: inject the dendrogram maintenance code here (following CL).
  parlay::parallel_for(0, starts.size(), [&](size_t i) {
    size_t start = starts[i];
    size_t end = (i == starts.size() - 1) ? merge_seq.size() : starts[i + 1];
    ABSL_CHECK_GT(end,
                  start);  // Ensure non-zero number of merges for this cluster.

    uintE center_id = std::get<0>(merge_seq[start]);
    float max_similarity = 0;
    uintE new_cluster_id = last_unused_id_ + i;

    // Write the neighbor size before scan.
    // - Update the CurrentNodeId for each (being merged) neighbor.
    // - Set the active flag for each such neighbor to false ( Deactivate ).
    // These two operations are handled by the LogicallyMergeWithParent call
    // below.
    for (size_t j = start; j < end; ++j) {
      uintE neighbor_id = std::get<1>(merge_seq[j]);
      // We will emit edges for both endpoints of the edge which is
      // why we emit 2*neighbors.size() edges
      edge_sizes[j] = 2 * clusters_[neighbor_id].GetNeighbors()->Size();

      // Merge the neighbor to the center. The underlying memory for the
      // neighbor is not yet cleared.
      clusters_[neighbor_id].LogicallyMergeWithParent(&clusters_[center_id]);

      uintE neighbor_cluster = clusters_[neighbor_id].CurrentClusterId();
      float merge_sim = std::get<2>(merge_seq[j]);
      dendrogram_.MergeToParent(neighbor_cluster, new_cluster_id, merge_sim);
      if (merge_sim > max_similarity) max_similarity = merge_sim;
    }

    uintE center_cluster = clusters_[center_id].CurrentClusterId();
    dendrogram_.MergeToParent(center_cluster, new_cluster_id, max_similarity);

    // Update the current cluster id of the center which remains active.
    clusters_[center_id].SetClusterId(new_cluster_id);
  });

  // Update last_unused_id_ (we created starts.size() many new clusters as a
  // result of these merges).
  last_unused_id_ += starts.size();

  // Next, map over deleted nodes and perform all edge deletions.
  parlay::parallel_for(0, merge_seq.size(), [&](size_t i) {
    uintE sat_id = std::get<1>(merge_seq[i]);
    // Map over all edges incident to the satellite, sat_id. For
    // each neighbor, neighbor_id, if neighbor_id is still active, remove
    // sat_id from neighbor_id's neighbors.
    auto map_f = [&](uintE neighbor_id, Weight& weight) {
      if (clusters_[neighbor_id].IsActive()) {
        bool deleted = clusters_[neighbor_id].GetNeighbors()->Remove(sat_id);
        // Edge must be present reciprocally (note that we are not deleting
        // this satellite's neighbors yet, so this condition should hold).
        // Later we will clear the satellite's neighbors.
        ABSL_CHECK(deleted);
      }
    };
    clusters_[sat_id].GetNeighbors()->Map(map_f);
  });

  // No references to deactivated nodes in any neighborlist of an
  // active node at this point. Next, we simply perform all
  // insertions (concurrently) in parallel. An alternate approach
  // that we used with PAM is to perform some semisorts + scans to
  // merge "same" edges, and then perform the insertions as a bulk
  // step. TODO: benchmark which approach is faster.

  // Scan to compute #edges we need to merge.
  size_t total_edges = parlay::scan_inplace(make_slice(edge_sizes));
  auto edges = parlay::sequence<Edge>::uninitialized(total_edges);
  constexpr uintE MaxNodeId = std::numeric_limits<uintE>::max();

  // Copy edges from each deleted vertex's neighbor-lists to edges.
  parlay::parallel_for(0, merge_seq.size(), [&](size_t i) {
    uintE center_id = std::get<0>(merge_seq[i]);
    uintE satellite_id = std::get<1>(merge_seq[i]);
    size_t offset = edge_sizes[i];
    // Map over every edge to a neighbor v, incident to the satellite.
    auto map_f = [&](uintE v, Weight& weight, size_t k) {
      bool v_active = clusters_[v].IsActive();
      uintE merged_id = clusters_[v].CurrentNodeId();
      // (1) v is active (itself a center) and it is not this center
      // (2) symmetry break to prevent sending a (u,v) edge between
      // two deactivated nodes twice to the activated targets.
      if (merged_id != center_id && (v_active || (satellite_id < v))) {
        StoredWeightType merge_data = weight.GetMergeData();
        edges[offset + 2 * k] = {center_id, merged_id, merge_data};
        edges[offset + 2 * k + 1] = {merged_id, center_id, merge_data};
      } else {
        edges[offset + 2 * k] = {MaxNodeId, MaxNodeId, StoredWeightType()};
        edges[offset + 2 * k + 1] = {MaxNodeId, MaxNodeId, StoredWeightType()};
      }
    };
    clusters_[satellite_id].GetNeighbors()->MapIndex(map_f);
    // No longer need this memory.
    clusters_[satellite_id].Clear();
  });

  // Filter out empty edge triples.
  auto pred = [&](const Edge& e) {
    return std::get<0>(e) != std::numeric_limits<uintE>::max();
  };
  auto filtered_edges = parlay::filter(parlay::make_slice(edges), pred);

  // Sort triples lexicographically.
  parlay::sort_inplace(parlay::make_slice(filtered_edges));

  // Scan over the triples, merging edges going to the same neighbor
  // with Weight::MergeWeights (conceptually just "+" on the underlying
  // weight).
  auto scan_f = [&](const Edge& l, const Edge& r) -> Edge {
    auto [l_u, l_v, l_weight] = l;
    auto [r_u, r_v, r_weight] = r;
    if (l_u != r_u || l_v != r_v) return {r_u, r_v, r_weight};
    return {r_u, r_v, Weight::MergeWeights(l_weight, r_weight)};
  };
  Edge id = {MaxNodeId, MaxNodeId, StoredWeightType()};
  auto scan_mon = parlay::make_monoid(scan_f, id);
  // After the scan, the last occurrence of each ngh has the
  // aggregated weight.
  parlay::scan_inclusive_inplace(parlay::make_slice(filtered_edges), scan_mon);

  size_t filtered_edges_size = filtered_edges.size();
  // Apply filter index to extract the last occurrence of each edge.
  auto idx_f = [&](const Edge& e, size_t idx) {
    const auto& [u, v, weight] = e;
    if (u == MaxNodeId) return false;
    if (idx < (filtered_edges_size - 1)) {
      const auto& [next_u, next_v, next_weight] = filtered_edges[idx + 1];
      // Next edge is not the same as this one
      return u != next_u || v != next_v;
    }
    return true;
  };
  auto inserts =
      parlay::filter_index(parlay::make_slice(filtered_edges), idx_f);

  // Inserts consists of a set of de-duplicated, sorted edge-insertions. Next
  // InsertTriples applies them to the underlying neighborhoods, resizing if
  // necessary.
  InsertTriples(std::move(inserts));

  if constexpr (Weight::kRequiresNeighborUpdate ||
                Weight::kRequiresEndpointsUpdate) {
    UpdateNeighbors(merge_seq, starts);
  }
}

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_H_
