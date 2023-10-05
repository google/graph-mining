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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_UNDIRECTED_CONVERTER_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_UNDIRECTED_CONVERTER_GRAPH_H_

#include "absl/status/status.h"
#include "gbbs/bridge.h"
#include "gbbs/gbbs.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/undirected_converter_graph.pb.h"
#include "in_memory/parallel/streaming_writer.h"

namespace graph_mining {
namespace in_memory {

// UndirectedConverterGraph is intended to be used as an intermediate layer
// between the directed input and the undirected graph representation used in
// parallel clustering algorithms.
class UndirectedConverterGraph : public InMemoryClusterer::Graph {
 public:
  UndirectedConverterGraph(
      const ::graph_mining::ConvertToUndirectedConfig& config,
      InMemoryClusterer::Graph* out_graph);
  absl::Status Import(AdjacencyList adjacency_list) override;
  absl::Status FinishImport() override;

 private:
  // Copies the current UndirectedConverterGraph to `out_graph_`.
  absl::Status CopyGraph() const;

  // Sparsifies the input graph.
  absl::Status Sparsify();

  // Undirected conversion configuration.
  const ::graph_mining::ConvertToUndirectedConfig config_;

  // Holds edges after FinishImport is called.
  parlay::sequence<std::tuple<gbbs::uintE, gbbs::uintE, double>> edges_;

  // Holds edge offsets after FinishImport is called.
  std::vector<std::size_t> offsets_;

  // Holds edges before FinishImport is called.
  StreamingWriter<std::tuple<gbbs::uintE, gbbs::uintE, double>> edge_buffer_;

  // Holdes nodes after FinishImport is called.
  parlay::sequence<std::tuple<gbbs::uintE, double>> nodes_;

  // Holds nodes before FinishImport is called.
  StreamingWriter<std::tuple<gbbs::uintE, double>> node_buffer_;

  // Target output graph, to which we copy the undirected graph. Not owned.
  InMemoryClusterer::Graph* out_graph_;
};

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_UNDIRECTED_CONVERTER_GRAPH_H_
