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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_CUT_SIZE_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_CUT_SIZE_H_

#include <vector>

#include "absl/status/statusor.h"
#include "gbbs/macros.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"

namespace graph_mining::in_memory {

// Computes and returns the cut ratio C / W  where C is the sum of all the
// cut-edge weights and W is the sum of all edge weights in the input graph. A
// cut-edge is an edge which does not have both of its ends in the same cluster
// (i.e. it crosses from one cluster into another one). Input clustering is
// given in a compact format where clustering[i] is the cluster-id for node i.
absl::StatusOr<double> ComputeCutRatio(
    const std::vector<gbbs::uintE>& clustering, const GbbsGraph& graph);

// Same as above but the input clustering is given in vector of vectors format.
// It check-fails if a node id in clustering is not in interval [0, num_nodes).
// TODO: Change the second parameter type to InMemoryClusterer::Graph.
absl::StatusOr<double> ComputeCutRatio(
    const InMemoryClusterer::Clustering& clustering, const GbbsGraph& graph);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_CUT_SIZE_H_
