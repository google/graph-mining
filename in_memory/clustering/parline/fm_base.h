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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_FM_BASE_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_FM_BASE_H_

#include "absl/container/flat_hash_set.h"
#include "in_memory/clustering/parline/cluster_pair_improver.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

class FMBase : public ClusterPairImprover {
 public:
  // Runs one pass of the Fiduccia-Mattheyses post-processing algorithm on
  // cluster1 and cluster2. The original paper for FM is located at:
  // http://web.eecs.umich.edu/~mazum/fmcut1.pdf
  //
  // This algorithm moves nodes between the two clusters in an
  // attempt to reduce the cut and escape local minima. It can be run
  // repeatedly, potentially improving the partition multiple times.
  //
  // For each of cluster1 and cluster2, the algorithm keeps track of the move
  // to the opposite cluster that would maximally improve cut value (the best
  // improvement may be negative). If neither move would exceed the maximum
  // cluster weight, it heuristically decides which of the two moves to make.
  // The heuristic for this choice can be changed in fm_base.cc. If only one of
  // the moves is feasible given the balance constraints, it makes that move. If
  // neither move is feasible, it locks-in the larger node's cluster, exluding
  // it from further consideration. After moving a node, that node's position is
  // similarly locked and it cannot be moved again. This continues until every
  // node's position has been locked. Some of the moves may have made the
  // partition worse, so after moving every node, the algorithm finds the
  // subsequence of moves that yielded the best partition.
  //
  // FM will be ineffective if the maximum cluster size is too small,
  // as some slack is required to move nodes between clusters in search of
  // a good cut.
  //
  // The function returns the improvement to the cut weight. Note that the
  // function may return 0.0 and then a positive number if called successively,
  // because it may find an alternate arrangement with exactly the same cut
  // weight, which it will prefer to the original ordering. The improvement
  // will always be nonnegative.
  //
  // The output sets, cluster1_to_cluster2 and cluster2_to_cluster1, are the ids
  // of nodes in each cluster that should be moved by the caller.
  double Improve(const GbbsGraph& graph,
                 const absl::flat_hash_set<NodeId>& cluster1,
                 const absl::flat_hash_set<NodeId>& cluster2,
                 double max_cluster_weight,
                 absl::flat_hash_set<NodeId>& cluster1_to_cluster2,
                 absl::flat_hash_set<NodeId>& cluster2_to_cluster1) override;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_FM_BASE_H_
