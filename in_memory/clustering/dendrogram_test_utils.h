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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_DENDROGRAM_TEST_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_DENDROGRAM_TEST_UTILS_H_

#include <vector>

#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/graph.h"

// Let n be the number of base objects that are being clustered (leaves of the
// dendrogram). Consider a representation of a dendrogram for a similarity graph
// in the parent array format, i.e., a vector or sequence of length <= 2*n-1
// containing, for each vertex, the id of its parent cluster and a floating
// point value indicating the similarity of this merge (the merge-similarity).
// Let m(u) be the *reported* merge-similarity of the merge creating a cluster
// u. For leaves (clusters of size 1), m(u) = \infty.
//
// Please note that the similarity stored in the dendrogram (the reported
// similarity) may be a multiplicative factor off from the true similarity of
// the underlying merge. For example, for the ParHac code, the true similarity
// of this merge may be a (1+epsilon) factor smaller than the stored similarity.
// In what follows, m(u) refers to the reported similarity.
//
// Note that for a dendrogram obtained by running the exact HAC algorithm using
// any reducible linkage function, e.g., unweighted average-linkage, the HAC
// algorithm ensures that the leaf-to-root paths are monotonically decreasing.
// However, if the HAC algorithm is approximate, the dendrogram may be
// non-monotone, i.e., we could have a leaf-to-root path where merge
// similarities *increase* along a leaf-to-root path. This piece of code
// computes three types of approximation factors for a dendrogram. Given a node
// u, let p(u) be the parent of u and let A(u) be the set of ancestors of u,
// including u. Let C be the set of all clusters in the graph.
//
// The utilities in this file help compute four kinds of approximation factors
// for a dendrogram:
// (1) max global approximation factor, i.e.,
//     max_{u in C} [max_{a in A(u)} m(a)] / m(u)
// (2) max local approximation factor, i.e.,
//     max_{u in C} m(p(u)) / m(u)
// (3) the maximum approximation for a merge, i.e.,
//     max_{u in C} MaxSim(u) / m(u)
//     where MaxSim(u) is the highest similarity edge at the time u is merged.
// (4) the goodness of the dendrogram, as defined in go/terahac-paper. For
//     complete dendrograms, this is equal to (3), but the two may diverge if
//     we only look at a subset of dendrogram merges.
//
// For exact algorithms, (1) = 1 and (2) <= 1.
// If (3) is equal to 1, then the dendrogram is exact (up to ties).
//
// These approximation factors can be used, for example, to check that an
// approximate HAC algorithm is working correctly, or to empirically bound the
// error from a hierarchical clustering algorithm.

namespace graph_mining::in_memory {

// Computes the max global approximation factor, i.e., item [(1)] described in
// the long comment above.
double GlobalApproximationFactor(const Dendrogram& dendrogram);

// Computes the max local approximation factor, i.e., item [(2)] described in
// the long comment above.
double LocalApproximationFactor(const Dendrogram& dendrogram);

// Computes the maximum approximation of all merges of similarity at least
// weight_threshold, and all the merges (possibly of smaller similarity) that
// must be made to perform these merges. This is item [(3)] described in the
// long comment above.
// If there are no merges of similarity at least weight_threshold, returns 0.
// This function requires all node weights in graph to be equal to 1.
absl::StatusOr<double> ClosenessApproximationFactor(
    const Dendrogram& dendrogram, const SimpleUndirectedGraph& graph,
    double weight_threshold);

// Computes the maximum goodness of all dendrogam merges (item [(4)] above).
// Somewhat counter-intuitively, lower goodness values are better. Hence, this
// function computes the worst-case goodness of all merges.
// Also, ensures that
//  * the graph after applying all dendrogram merges does not
//    have edges of weight >= weight_threshold (returns an OutOfRangeError
//    otherwise).
//  * the merge similarities whose weight is >= weight threshold and accurrately
//    reported (up to an additive 1e-6 error) in the dendrogram.
// Supports starting with an "intermediate" state, i.e., with arbitrary
// min_merge_similarities and node weights provided. When weight_threshold = 0,
// the node weights are all 1 and all min_merge_similarities are infinite (or
// sufficiently large), the result is equal to what ClosenessApproximationFactor
// computes.
absl::StatusOr<double> DendrogramGoodness(
    const Dendrogram& dendrogram, const SimpleUndirectedGraph& graph,
    double weight_threshold, std::vector<double> min_merge_similarities);

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_DENDROGRAM_TEST_UTILS_H_
