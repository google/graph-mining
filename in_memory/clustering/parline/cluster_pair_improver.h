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

// An interface for classes that attempt to reduce the cut weight between two
// input clusters, subject to some maximum cluster size.

#ifndef RESEARCH_GRAPH_IN_MEMORY_BALANCED_PARTITIONER_CLUSTER_PAIR_IMPROVER_H_
#define RESEARCH_GRAPH_IN_MEMORY_BALANCED_PARTITIONER_CLUSTER_PAIR_IMPROVER_H_

#include "absl/container/flat_hash_set.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"

namespace graph_mining::in_memory {

class ClusterPairImprover {
 public:
  virtual ~ClusterPairImprover() = default;

  // For two sets of node ids (cluster1 and cluster2), run some post-processing
  // to decrease the cut weight between those clusters, subject to the
  // constraint that the maximum node weight of a cluster be max_cluster_weight.
  //
  // max_cluster_weight is a hard limit on cluster size. If this constraint is
  // already violated for the input clusters, the behavior is specific to the
  // instantiated post-processor and it is only guaranteed that the
  // algorithm will return some partition.
  //
  // The output is given as a set of nodes to move from cluster1 to cluster 2
  // and a set of nodes to move from cluster2 to cluster 1.
  virtual double Improve(const GbbsGraph& graph,
                         const absl::flat_hash_set<NodeId>& cluster1,
                         const absl::flat_hash_set<NodeId>& cluster2,
                         double max_cluster_weight,
                         absl::flat_hash_set<NodeId>& cluster1_to_cluster2,
                         absl::flat_hash_set<NodeId>& cluster2_to_cluster1) = 0;
};

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_BALANCED_PARTITIONER_CLUSTER_PAIR_IMPROVER_H_
