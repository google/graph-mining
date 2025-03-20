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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/types.h"

namespace graph_mining {
namespace in_memory {

// Interface of an in-memory clustering algorithm. The classes implementing this
// interface maintain a mutable graph, which can be clustered using a given set
// of parameters.
class InMemoryClusterer {
 public:
  // This is a basic interface for building graphs. Note that the interface only
  // specifies how to build a graph, as different clusterers may use different
  // interfaces for accessing it.
  // The node ids are consecutive, 0-based integers. In particular, adding a
  // node of id k to an empty graph creates k+1 nodes 0, ..., k.
  class Graph {
   public:
    using NodeId = graph_mining::in_memory::NodeId;

    // Represents a weighted node with weighted outgoing edges.
    struct AdjacencyList {
      static constexpr double kDefaultNodeWeight = 1;
      NodeId id = -1;
      double weight = kDefaultNodeWeight;
      std::vector<std::pair<NodeId, double>> outgoing_edges;
      std::optional<int32_t> part;
    };

    virtual ~Graph() = default;

    // Prepares the graph for node importer.
    //
    // This is an optional step in the graph building process. When called, this
    // may enable subclass-specific optimizations. For an example of the
    // lock-free graph building optimization for GbbsOutEdgesOnlyGraph, see
    // GbbsOutEdgesOnlyGraph::PrepareImport.
    //
    // `num_nodes` should be the exact number of nodes in the input graph. When
    // the condition is violated, the behavior is defined by each subclass.
    virtual absl::Status PrepareImport(int64_t num_nodes);

    // Adds a weighted node and its weighted out-edges to the graph. Depending
    // on the Graph implementation, the symmetric edges may be added as well,
    // and edge weights may be adjusted for symmetry.
    //
    // Import must be called at most once for each node. If not called for a
    // node, that node defaults to weight 1.
    //
    // IMPLEMENTATIONS MUST ALLOW CONCURRENT CALLS TO Import()!
    //
    // If this interface is not convenient, consider either using NodeImporter
    // or loading your graph into a SimpleUndirectedGraph followed by
    // CopyGraph().
    virtual absl::Status Import(AdjacencyList adjacency_list) = 0;

    // Finalizes the graph after all calls to Import are complete.
    //
    // Calling this function is REQUIRED before any calls to Cluster or
    // RefineClusters are made.
    virtual absl::Status FinishImport();
  };

  using NodeId = Graph::NodeId;
  using AdjacencyList = Graph::AdjacencyList;

  // Represents clustering: each element of the vector contains the set of
  // NodeIds in one cluster. We call a clustering non-overlapping if the
  // elements of the clustering are nonempty vectors that together contain each
  // NodeId exactly once.
  using Clustering = graph_mining::in_memory::Clustering;

  // Represents a family of clusterings using a dendrogram in the parent-array
  // format (see dendrogram.h for more).
  using Dendrogram = graph_mining::in_memory::Dendrogram;

  virtual ~InMemoryClusterer() {}

  // Accessor to the maintained graph. Use it to build the graph.
  virtual Graph* MutableGraph() = 0;

  // Clusters the currently maintained graph using the given set of parameters.
  // Returns a clustering, or an error if the algorithm failed to cluster the
  // given graph.
  // Note that the same clustering may have multiple representations, and the
  // function may return any of them. You can use CanonicalizeClustering() to
  // get a canonical clustering representation.
  virtual absl::StatusOr<Clustering> Cluster(
      const graph_mining::in_memory::ClustererConfig& config) const = 0;

  // Similar to Cluster(), but returns the clustering in a different format,
  // namely a vector with length equal to the number of nodes in the graph,
  // where the i-th element is the ID of the cluster ID to which node i belongs.
  // The returned cluster IDs must be in the range [0, ..., number of nodes in
  // the graph - 1].
  virtual absl::StatusOr<std::vector<NodeId>> ClusterAndReturnClusterIds(
      const graph_mining::in_memory::ClustererConfig& config) const {
    // TODO: b/397376625 - Replace this with an actual implementation that
    // invokes the Cluster() method and converts the result to the desired
    // format.
    return absl::UnimplementedError(
        "'ClusterAsClusterIdSequence' is not implemented");
  }

  // Same as above, except that it returns a sequence of flat clusterings. The
  // last element of the sequence is the final clustering. This is primarily
  // used for hierarchical clusterings, but callers should NOT assume that there
  // is a strict hierarchy structure (i.e. that clusters in clustering i are
  // obtained by merging clusters from clustering i-1). The default
  // implementation returns a single-element vector with the result of
  // Cluster().
  virtual absl::StatusOr<std::vector<Clustering>> HierarchicalFlatCluster(
      const graph_mining::in_memory::ClustererConfig& config) const;

  // Returns a family of clusterings represented by a dendrogram in the
  // parent-array format (see dendrogram.h for more details). The resulting
  // dendrogram can be *flattened* given a user-specified linkage similarity to
  // obtain a flat clustering. Note that the default implementation returns an
  // error status, so callers should ensure that the Clusterer being used
  // implements this method.
  virtual absl::StatusOr<Dendrogram> HierarchicalCluster(
      const graph_mining::in_memory::ClustererConfig& config) const;

  // Refines a list of clusters and redirects the given pointer to new clusters.
  // This function is useful for methods that can refine / operate on an
  // existing clustering. It does not take ownership of clustering. The default
  // implementation does nothing and returns OkStatus.
  virtual absl::Status RefineClusters(
      const graph_mining::in_memory::ClustererConfig& config,
      Clustering* clustering) const {
    return absl::OkStatus();
  }

  // Provides a pointer to a vector that contains string ids corresponding to
  // the NodeIds. If set, the ids from the provided map are used in the log and
  // error messages. The vector must live during all method calls of this
  // object. This call does *not* take ownership of the pointee. Using this
  // function is not required. If this function is never called, the ids are
  // converted to strings using absl::StrCat.
  void set_node_id_map(const std::vector<std::string>* node_id_map) {
    node_id_map_ = node_id_map;
  }

 protected:
  // Returns the string id corresponding to a given NodeId. If set_node_id_map
  // was called, uses the map to get the ids. Otherwise, returns the string
  // representation of the id.
  std::string StringId(NodeId id) const;

 private:
  // NodeId map set by set_node_id_map(). May be left to nullptr even after
  // initialization.
  const std::vector<std::string>* node_id_map_ = nullptr;
};

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
