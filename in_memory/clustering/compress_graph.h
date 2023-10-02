// Copyright 2010-2023 Google LLC
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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COMPRESS_GRAPH_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COMPRESS_GRAPH_H_

#include <functional>
#include <memory>
#include <vector>

#include "in_memory/clustering/graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining {
namespace in_memory {

// Compress cluster ids into vertices, and aggregate edges using the given
// edge_aggregation_function. Note that
//   a) each node x is compressed into a new node cluster_ids[x],
//   b) if cluster_ids[x] == -1, then x is ignored, and
//   c) the resulting graph has the same number of nodes as the initial one.
absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> CompressGraph(
    const SimpleUndirectedGraph& graph,
    const std::vector<InMemoryClusterer::NodeId>& cluster_ids,
    const std::function<double(double, double)>& edge_aggregation_function,
    bool ignore_self_loops = true);

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_COMPRESS_GRAPH_H_
