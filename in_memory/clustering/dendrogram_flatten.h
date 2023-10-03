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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_FLATTEN_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_FLATTEN_H_

#include <vector>

#include "absl/status/statusor.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/types.h"
#include "in_memory/connected_components/asynchronous_union_find.h"

namespace graph_mining::in_memory {

// Given a merge sequence of a monotone dendrogram, applies the merges until
// |stopping_condition| is met. It then outputs the highest-scoring clustering
// it encountered along the way, according to |scoring_function|. If
// |scoring_function| does not return a positive score for any clustering along
// the way, the final clustering is output (the one where |stopping_condition|
// first returned true). Note that the |weight| passed into |scoring_function|
// will be set to -infinity in the case that there are no remaining edges in
// the graph, which will happen if the |stopping_condition| is not met before
// all edges are contracted.
absl::StatusOr<Clustering> FlattenClusteringWithScorer(
    const Dendrogram& dendrogram,

    std::function<bool(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        stopping_condition,
    std::function<double(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        scoring_function);

// Same as FlattenClusteringWithScorer, except that it outputs a vector of
// clusterings, one for each similarity threshold given in
// |similarity_thresholds|. This is equivalent to calling
// FlattenClusteringWithScorer repeatedly with a stopping_condition of
// "similarity < threshold" for each threshold in |similarity_thresholds|, and
// outputting the vector of results in the same order as given in
// |similarity_thresholds|. The input |similarity_thresholds| must be
// monotonically non-increasing, or else an error is returned.
absl::StatusOr<std::vector<Clustering>> HierarchicalFlattenClusteringWithScorer(
    const Dendrogram& dendrogram,
    const std::vector<double>& similarity_thresholds,
    std::function<double(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        scoring_function);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DENDROGRAM_FLATTEN_H_
