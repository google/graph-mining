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

#include "in_memory/clustering/affinity/parallel_affinity_internal.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/parallel/parallel_graph_utils.h"
#include "in_memory/parallel/parallel_sequence_ops.h"
#include "parlay/sequence.h"

using ::graph_mining::in_memory::AffinityClustererConfig;

namespace {

struct PerVertexClusterStats {
  gbbs::uintE cluster_id;
  float volume;
  float intra_cluster_weight;
  float inter_cluster_weight;
};

inline double GetNodeWeight(const std::vector<double>& node_weights,
                            const gbbs::uintE id) {
  return node_weights.empty() ? 1 : node_weights[id];
}

}  // namespace

namespace graph_mining::in_memory {

namespace internal {

std::vector<ClusterStats> ComputeFinishedClusterStats(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    const std::vector<gbbs::uintE>& cluster_ids,
    gbbs::uintE num_compressed_vertices) {
  std::size_t n = G.n;
  std::vector<ClusterStats> aggregate_cluster_stats(num_compressed_vertices,
                                                    {0, 0});
  std::vector<PerVertexClusterStats> cluster_stats(n);

  // Compute cluster statistics contributions of each vertex
  auto sum_map_f = [&](gbbs::uintE u, gbbs::uintE v, float weight) -> float {
    return weight;
  };
  auto add_m = parlay::addm<float>();
  parlay::parallel_for(0, n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids[i];
    auto volume = G.get_vertex(i).out_neighbors().reduce(sum_map_f, add_m);
    if (cluster_id_i == UINT_E_MAX) {
      cluster_stats[i] =
          PerVertexClusterStats{cluster_id_i, volume, float{0}, float{0}};
    } else {
      auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                         float weight) -> float {
        if (cluster_id_i == cluster_ids[v] && v <= i) return weight;
        return 0;
      };
      auto inter_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                         float weight) -> float {
        if (cluster_id_i != cluster_ids[v]) return weight;
        return 0;
      };
      auto intra_cluster_weight = G.get_vertex(i).out_neighbors().reduce(
          intra_cluster_sum_map_f, add_m);
      auto inter_cluster_weight = G.get_vertex(i).out_neighbors().reduce(
          inter_cluster_sum_map_f, add_m);
      cluster_stats[i] = PerVertexClusterStats{
          cluster_id_i, volume, intra_cluster_weight, inter_cluster_weight};
    }
  });

  // Compute total graph volume
  auto graph_volume_stats =
      graph_mining::in_memory::Reduce<PerVertexClusterStats>(
          absl::Span<PerVertexClusterStats>(cluster_stats.data(),
                                            cluster_stats.size()),
          [&](PerVertexClusterStats a,
              PerVertexClusterStats b) -> PerVertexClusterStats {
            return PerVertexClusterStats{0, a.volume + b.volume, 0, 0};
          },
          PerVertexClusterStats{0, 0, 0, 0});
  float graph_volume = graph_volume_stats.volume;

  // Cluster statistics must now be aggregated per cluster id
  // Sort cluster statistics by cluster id
  auto cluster_stats_sort =
      graph_mining::in_memory::ParallelSampleSort<PerVertexClusterStats>(
          absl::Span<PerVertexClusterStats>(cluster_stats.data(), n),
          [&](PerVertexClusterStats a, PerVertexClusterStats b) {
            return a.cluster_id < b.cluster_id;
          });

  // Obtain the boundary indices where statistics differ by cluster id
  // These indices are stored in filtered_mark_ids
  std::vector<gbbs::uintE> filtered_mark_ids =
      graph_mining::in_memory::GetBoundaryIndices<gbbs::uintE>(
          n, [&cluster_stats_sort](std::size_t i, std::size_t j) {
            return cluster_stats_sort[i].cluster_id ==
                   cluster_stats_sort[j].cluster_id;
          });
  std::size_t num_filtered_mark_ids = filtered_mark_ids.size() - 1;

  // Compute aggregate statistics by cluster id
  parlay::parallel_for(0, num_filtered_mark_ids, [&](size_t i) {
    // Combine cluster statistics from start_id_index to end_id_index
    gbbs::uintE start_id_index = filtered_mark_ids[i];
    gbbs::uintE end_id_index = filtered_mark_ids[i + 1];
    auto cluster_id = cluster_stats_sort[start_id_index].cluster_id;
    if (cluster_id != UINT_E_MAX) {
      gbbs::uintE cluster_size = end_id_index - start_id_index;
      auto stats_sum = graph_mining::in_memory::Reduce<PerVertexClusterStats>(
          absl::Span<const PerVertexClusterStats>(
              cluster_stats_sort.begin() + start_id_index,
              end_id_index - start_id_index),
          [&](PerVertexClusterStats a, PerVertexClusterStats b) {
            return PerVertexClusterStats{
                0, a.volume + b.volume,
                a.intra_cluster_weight + b.intra_cluster_weight,
                a.inter_cluster_weight + b.inter_cluster_weight};
          },
          PerVertexClusterStats{0, 0, 0, 0});
      float density = (cluster_size >= 2)
                          ? stats_sum.intra_cluster_weight /
                                (static_cast<float>(cluster_size) *
                                 (cluster_size - 1) / 2.0)
                          : 0.0;
      float volume = stats_sum.volume;
      float denominator = std::min(volume, graph_volume - volume);
      float inter_cluster_weight =
          (denominator < 1e-6) ? 1.0
                               : stats_sum.inter_cluster_weight / denominator;
      aggregate_cluster_stats[cluster_id] =
          ClusterStats(density, inter_cluster_weight);
    }
  });

  return aggregate_cluster_stats;
}

}  // namespace internal

absl::StatusOr<std::vector<gbbs::uintE>> NearestNeighborLinkage(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    float weight_threshold,
    std::optional<internal::SizeConstraintConfig> size_constraint_config) {
  std::size_t n = graph.n;
  AsynchronousUnionFind<gbbs::uintE> labels(n);

  auto best_neighbors =
      parlay::sequence<internal::Edge>::from_function(n, [](size_t i) {
        return internal::Edge{static_cast<gbbs::uintE>(i),
                              std::numeric_limits<float>::infinity()};
      });
  const gbbs::uintE undefined_neighbor = n;

  parlay::parallel_for(0, n, [&](gbbs::uintE i) {
    auto vertex = graph.get_vertex(i);
    // Size-constraint filtering for the center node.
    if (size_constraint_config.has_value() &&
        size_constraint_config->size_constraint.has_min_cluster_size() &&
        GetNodeWeight(size_constraint_config->node_weights, i) >
            size_constraint_config->size_constraint.min_cluster_size()) {
      return;
    }

    float max_weight = weight_threshold;
    gbbs::uintE max_neighbor = undefined_neighbor;
    auto find_max_neighbor_func = [&](gbbs::uintE u, gbbs::uintE v,
                                      float weight) {
      // Size-constraint-based edge filtering.
      if (size_constraint_config.has_value() &&
          size_constraint_config->size_constraint.has_max_cluster_size() &&
          GetNodeWeight(size_constraint_config->node_weights, u) +
                  GetNodeWeight(size_constraint_config->node_weights, v) >
              size_constraint_config->size_constraint.max_cluster_size()) {
        return;
      }

      // TODO: Make the tie-breaking consistent with
      // AffinityClusterer and distributed affinity clustering (which are the
      // same).
      if (std::tie(weight, v) > std::tie(max_weight, max_neighbor) ||
          (weight == weight_threshold && max_neighbor == undefined_neighbor)) {
        max_weight = weight;
        max_neighbor = v;
      }
    };
    vertex.out_neighbors().map(find_max_neighbor_func, false);
    if (max_neighbor != undefined_neighbor) {
      labels.Unite(i, max_neighbor);
      best_neighbors[i] = {max_neighbor, max_weight};
    }
  });

  parlay::parallel_for(0, n, [&](gbbs::uintE i) { labels.Find(i); });

  if (size_constraint_config.has_value()) {
    const auto& size_constraint =
        size_constraint_config.value().size_constraint;
    if (size_constraint.has_max_cluster_size() ||
        size_constraint.has_min_cluster_size()) {
      auto label_seq =
          EnforceMaxClusterSize(*size_constraint_config, labels.ComponentIds(),
                                std::move(best_neighbors));
      labels = AsynchronousUnionFind<gbbs::uintE>(std::move(label_seq));
    }
  }

  auto component_ids = labels.ComponentIds();
  return std::vector<gbbs::uintE>(component_ids.begin(), component_ids.end());
}

absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<double>& original_node_weights,
    const std::vector<gbbs::uintE>& cluster_ids,
    const AffinityClustererConfig& affinity_config) {
  const auto edge_aggregation = affinity_config.edge_aggregation_function();
  // TODO: Support percentile edge aggregation
  ABSL_CHECK_NE(edge_aggregation, AffinityClustererConfig::PERCENTILE)
      << "Aggregation not supported: " << edge_aggregation;
  ABSL_CHECK_NE(edge_aggregation, AffinityClustererConfig::EXPLICIT_AVERAGE)
      << "Aggregation not supported: " << edge_aggregation;
  std::size_t n = original_graph.n;
  // Obtain the number of vertices in the new graph
  gbbs::uintE num_compressed_vertices =
      1 +
      graph_mining::in_memory::Reduce<gbbs::uintE>(
          absl::Span<const gbbs::uintE>(cluster_ids.data(), cluster_ids.size()),
          [&](gbbs::uintE reduce, gbbs::uintE a) {
            return (reduce == UINT_E_MAX)
                       ? a
                       : (a == UINT_E_MAX ? reduce : std::max(reduce, a));
          },
          UINT_E_MAX);

  // Retrieve node weights
  std::vector<double> node_weights(num_compressed_vertices, double{0});
  // TODO: This could be made parallel, although perhaps it would be
  // too much overhead.
  for (std::size_t i = 0; i < n; ++i) {
    if (cluster_ids[i] != UINT_E_MAX)
      node_weights[cluster_ids[i]] += GetNodeWeight(original_node_weights, i);
  }

  // Compute new inter cluster edges using sorting
  // TODO: Allow optionality to choose between aggregation methods
  std::function<float(float, float)> edge_aggregation_func;
  // scale_func is applied to each edge weight prior to aggregation. This
  // is used in the average and cut sparsity aggregation methods, in which
  // aggregating new edges with previously compressed edges is not
  // straightforward. In more detail, these methods are functions f that do
  // not satisfy the property that f(x, y, z) = f(f(x, y), z) for edge weights
  // x, y, and z. Thus, in order to properly aggregate new edges with previously
  // compressed edges, we must first scale the edge weight by the node weights
  // of its endpoints, sum the resulting edge weights, and finally reapply
  // a scaling factor.
  std::function<float(std::tuple<gbbs::uintE, gbbs::uintE, float>)> scale_func =
      [](std::tuple<gbbs::uintE, gbbs::uintE, float> v) {
        return std::get<2>(v);
      };
  switch (edge_aggregation) {
    case AffinityClustererConfig::MAX:
      edge_aggregation_func = [](float w1, float w2) {
        return std::max(w1, w2);
      };
      break;
    case AffinityClustererConfig::SUM:
      edge_aggregation_func = [](float w1, float w2) { return w1 + w2; };
      break;
    case AffinityClustererConfig::DEFAULT_AVERAGE:
      edge_aggregation_func = [](float w1, float w2) { return w1 + w2; };
      if (!original_node_weights.empty()) {
        scale_func = [&original_node_weights](
                         std::tuple<gbbs::uintE, gbbs::uintE, float> v) {
          float scaling_factor = original_node_weights[std::get<0>(v)] *
                                 original_node_weights[std::get<1>(v)];
          return std::get<2>(v) * scaling_factor;
        };
      }
      break;
    case AffinityClustererConfig::CUT_SPARSITY:
      edge_aggregation_func = [](float w1, float w2) { return w1 + w2; };
      if (!original_node_weights.empty()) {
        scale_func = [&original_node_weights](
                         std::tuple<gbbs::uintE, gbbs::uintE, float> v) {
          float scaling_factor =
              std::min(original_node_weights[std::get<0>(v)],
                       original_node_weights[std::get<1>(v)]);
          return std::get<2>(v) * scaling_factor;
        };
      }
      break;
    default:
      ABSL_LOG(FATAL) << "Unknown edge aggregation method: "
                      << edge_aggregation;
  }

  OffsetsEdges offsets_edges = ComputeInterClusterEdgesSort(
      original_graph, cluster_ids, num_compressed_vertices,
      edge_aggregation_func, std::not_equal_to<gbbs::uintE>(), scale_func);

  std::vector<std::size_t> offsets = offsets_edges.offsets;
  std::size_t num_edges = offsets_edges.num_edges;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges =
      std::move(offsets_edges.edges);

  if (edge_aggregation == AffinityClustererConfig::SUM ||
      edge_aggregation == AffinityClustererConfig::MAX) {
    return GraphWithWeights(
        MakeGbbsGraph<float>(offsets, num_compressed_vertices, std::move(edges),
                             num_edges),
        node_weights);
  }

  ABSL_CHECK(edge_aggregation == AffinityClustererConfig::DEFAULT_AVERAGE ||
             edge_aggregation == AffinityClustererConfig::CUT_SPARSITY)
      << "Invalid aggregation: " << edge_aggregation;

  // Scale edge weights
  parlay::parallel_for(0, num_compressed_vertices, [&](std::size_t i) {
    auto offset = offsets[i];
    auto degree = offsets[i + 1] - offset;
    for (std::size_t j = 0; j < degree; j++) {
      const auto& edge = edges[offset + j];
      float scaling_factor = 0;
      if (edge_aggregation == AffinityClustererConfig::DEFAULT_AVERAGE) {
        scaling_factor = node_weights[i] * node_weights[std::get<0>(edge)];
      } else if (edge_aggregation == AffinityClustererConfig::CUT_SPARSITY) {
        scaling_factor =
            std::min(node_weights[i], node_weights[std::get<0>(edge)]);
      }
      edges[offset + j] = std::make_tuple(std::get<0>(edge),
                                          std::get<1>(edge) / scaling_factor);
    }
  });

  return GraphWithWeights(MakeGbbsGraph<float>(offsets, num_compressed_vertices,
                                               std::move(edges), num_edges),
                          node_weights);
}

InMemoryClusterer::Clustering ComputeClusters(
    const std::vector<gbbs::uintE>& cluster_ids,
    std::function<bool(gbbs::uintE)> is_finished) {
  // Pack out finished vertices from the boolean array
  auto finished_vertex_pack =
      graph_mining::in_memory::PackIndex<InMemoryClusterer::NodeId>(
          is_finished, cluster_ids.size());

  auto get_clusters =
      [&](InMemoryClusterer::NodeId i) -> InMemoryClusterer::NodeId {
    return finished_vertex_pack[i];
  };
  return graph_mining::in_memory::OutputIndicesById<gbbs::uintE,
                                                    InMemoryClusterer::NodeId>(
      cluster_ids, get_clusters, finished_vertex_pack.size());
}

InMemoryClusterer::Clustering FindFinishedClusters(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    const AffinityClustererConfig& affinity_config,
    std::vector<gbbs::uintE>& cluster_ids,
    std::vector<gbbs::uintE>& compressed_cluster_ids) {
  if (affinity_config.active_cluster_conditions().empty())
    return InMemoryClusterer::Clustering();
  std::size_t n = G.n;

  gbbs::uintE num_compressed_vertices =
      1 + graph_mining::in_memory::Reduce<gbbs::uintE>(
              absl::Span<gbbs::uintE>(cluster_ids.data(), cluster_ids.size()),
              [&](gbbs::uintE reduce, gbbs::uintE a) {
                return (reduce == UINT_E_MAX)
                           ? a
                           : (a == UINT_E_MAX ? reduce : std::max(reduce, a));
              },
              UINT_E_MAX);

  auto finished = parlay::sequence<bool>(num_compressed_vertices, true);

  if (!affinity_config.active_cluster_conditions().empty()) {
    std::vector<internal::ClusterStats> aggregate_cluster_stats =
        internal::ComputeFinishedClusterStats(G, cluster_ids,
                                              num_compressed_vertices);

    // Check for finished clusters
    // TODO: Use a unique ptr here
    parlay::parallel_for(0, num_compressed_vertices, [&](std::size_t i) {
      for (std::size_t j = 0;
           j < affinity_config.active_cluster_conditions().size(); j++) {
        bool satisfied = true;
        auto condition = affinity_config.active_cluster_conditions().Get(j);
        if (condition.has_min_density() &&
            aggregate_cluster_stats[i].density < condition.min_density())
          satisfied = false;
        if (condition.has_min_conductance() &&
            aggregate_cluster_stats[i].conductance <
                condition.min_conductance())
          satisfied = false;
        if (satisfied) {
          finished[i] = false;
          break;
        }
      }
    });
  }

  // Compute finished clusters
  auto is_finished = [&](gbbs::uintE i) {
    return (cluster_ids[i] == UINT_E_MAX) ? false : finished[cluster_ids[i]];
  };
  auto finished_clusters = ComputeClusters(cluster_ids, is_finished);

  // Update the cluster ids for vertices belonging to finished clusters
  parlay::parallel_for(0, n, [&](size_t i) {
    if (cluster_ids[i] != UINT_E_MAX && finished[cluster_ids[i]])
      cluster_ids[i] = UINT_E_MAX;
  });

  parlay::parallel_for(0, compressed_cluster_ids.size(), [&](size_t i) {
    if (compressed_cluster_ids[i] != UINT_E_MAX &&
        finished[compressed_cluster_ids[i]]) {
      compressed_cluster_ids[i] = UINT_E_MAX;
    }
  });

  return finished_clusters;
}

namespace internal {

// The implementation below is based on the Flume counterpart
// `EnforceMaxClusterSizeFn` (http://shortn/_KDc9bZPfpP).
//
// EnforceMaxClusterSize implements the following workflow
// - Group nodes by their cluster ids (i.e., connected component ids)
// - For each group, optionally break it down to honor max cluster size
// --- Sort edges
// --- Process sorted edges sequentially, unite nodes if min/max cluster size
//     constraints are satisfied
parlay::sequence<gbbs::uintE> EnforceMaxClusterSize(
    const SizeConstraintConfig& size_constraint_config,
    absl::Span<const gbbs::uintE> cluster_ids,
    parlay::sequence<Edge>&& best_neighbors) {
  std::size_t n = cluster_ids.size();
  ABSL_CHECK_EQ(best_neighbors.size(), n);

  const auto& size_constraint = size_constraint_config.size_constraint;

  // Group node ides by connected component ids.
  auto cluster_groups = OutputIndicesById<gbbs::uintE, gbbs::uintE>(
      std::vector<gbbs::uintE>(cluster_ids.begin(), cluster_ids.end()),
      [](gbbs::uintE i) { return i; }, n);

  std::vector<double> node_weights = size_constraint_config.node_weights;
  if (node_weights.empty()) {
    node_weights.resize(n, 1);
  }

  AsynchronousUnionFind<gbbs::uintE> labels(n);

  parlay::parallel_for(0, cluster_groups.size(), [&](std::size_t i) {
    auto& node_idx = cluster_groups[i];

    // For a group of node indices, sort by descending edge weight, ascending
    // node weight, and ascending (integer) node id.
    parlay::sample_sort_inplace(
        parlay::make_slice(node_idx), [&](gbbs::uintE lhs, gbbs::uintE rhs) {
          return std::forward_as_tuple(-best_neighbors[lhs].weight,
                                       node_weights[lhs], lhs) <
                 std::forward_as_tuple(-best_neighbors[rhs].weight,
                                       node_weights[rhs], rhs);
        });

    // Sequentially process nodes in the group according to the sorted node
    // indices. Given that each node belongs to exactly one group, it is safe to
    // use the non-atomic version of union-find for the per-group sequential
    // processing.
    for (const auto node_id : node_idx) {
      auto best_neighbor_node_id = best_neighbors[node_id].neighbor_id;
      ABSL_CHECK_LT(best_neighbor_node_id, n);

      auto node_root = labels.Find(node_id);
      auto neighbor_root = labels.Find(best_neighbor_node_id);
      if (node_root == neighbor_root) {
        continue;
      }

      // If small clusters are preferred, and both clusters are already above
      // the min_cluster_size, do not merge them.
      if (size_constraint.prefer_min_cluster_size() &&
          size_constraint.has_min_cluster_size() &&
          node_weights[node_root] >= size_constraint.min_cluster_size() &&
          node_weights[neighbor_root] >= size_constraint.min_cluster_size()) {
        continue;
      }

      // If the combined weight of the two nodes does not exceed
      // max_cluster_size, assign the current node as parent of this neighbor.
      double total_weight =
          node_weights[node_root] + node_weights[neighbor_root];

      if (!size_constraint.has_max_cluster_size() ||
          total_weight <= size_constraint.max_cluster_size()) {
        labels.Unite(node_root, neighbor_root);

        // Update the node weight of the new parent after the merge. Note that
        // there is no need to update the weight of the other node which becomes
        // a leaf node after the merge, because that weight will never be used
        // for subsequent computation.
        node_weights[labels.Find(node_root)] = total_weight;
      }
    }
  });

  return std::move(labels).ComponentSequence();
}

}  // namespace internal

}  // namespace graph_mining::in_memory
