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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_

#include <cstdint>
#include <cstdio>

#include "gbbs/gbbs.h"
#include "gbbs/graph_io.h"
#include "gbbs/macros.h"
#include "in_memory/parallel/parallel_sequence_ops.h"

namespace graph_mining::in_memory {

struct OffsetsEdges {
  std::vector<std::size_t> offsets;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges;
  std::size_t num_edges;
};

// Given get_key, which is nondecreasing, defined for 0, ..., num_keys-1, and
// returns an unsigned integer less than n, return an array of length n + 1
// where array[i] := minimum index k such that get_key(k) >= i.
// Note that array[n] = the total number of keys, num_keys.
std::vector<std::size_t> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    std::size_t num_keys, std::size_t n);

// Remap the endpoints of edges using the provided cluster_ids. Despite the
// name, the function also returns intra-cluster edges, as self-loops (unless
// they are filtered out by is_valid_func). The input graph should be
// undirected: the weight of a self-loop is obtained from all self-loops plus
// *one* of the two directed edges that represent an undirected edge.
//
// Uses scale_func to first scale edge weights, and then uses aggregate_func to
// combine multiple edges on the same cluster ids. Note that aggregate_func must
// be commutative and associative, with 0 as its identity. Returns sorted edges
// and offsets array in edges and offsets respectively. The number of compressed
// vertices should be 1 + the maximum cluster id in cluster_ids. The
// implementation uses parallel sorting.
OffsetsEdges ComputeInterClusterEdgesSort(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func,
    const std::function<float(std::tuple<gbbs::uintE, gbbs::uintE, float>)>&
        scale_func);

// Given an array of edges (given by a tuple consisting of the second endpoint
// and a weight if the edges are weighted) and the offsets marking the index
// of the first edge corresponding to each vertex (essentially, CSR format),
// return the corresponding graph in GBBS format.
// Note that the returned graph takes ownership of the edges array.
template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
MakeGbbsGraph(
    const std::vector<std::size_t>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  parlay::parallel_for(0, num_vertices, [&](std::size_t i) {
    vertices[i] = gbbs::symmetric_vertex<WeightType>(
        edges,
        gbbs::vertex_data{
            offsets[i], static_cast<gbbs::uintE>(offsets[i + 1] - offsets[i])},
        i);
  });

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

// Given new cluster ids in compressed_cluster_ids, remap the original
// cluster ids. A cluster id of UINT_E_MAX indicates that the vertex
// has already been placed into a finalized cluster, and this is
// preserved in the remapping.
std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids);

// Holds a GBBS graph and a corresponding node weights
struct GraphWithWeights {
  GraphWithWeights() = default;
  GraphWithWeights(
      std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
          graph_,
      std::vector<double> node_weights_)
      : graph(std::move(graph_)), node_weights(std::move(node_weights_)) {}
  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      graph;
  std::vector<double> node_weights;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
