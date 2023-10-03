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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRWISE_IMPROVER_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRWISE_IMPROVER_H_

#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/in_memory_clusterer.h"
#include "in_memory/clustering/parline/parline.pb.h"

namespace graph_mining::in_memory {

// Given an input graph and an input clustering it improves the clusters by
// first pairing them (see PairwiseImproverConfig.cluster_pairing_method) and
// then running an in-memory pairwise partition improving algorithm (for
// example see third_pary/graph_mining/in_memory/clustering/parline/fm_base.h).
// It makes sure that the maximum cluster size is
//         avg_cluster_size * (1 + config.max_imbalance).
// This function assumes that the ordering of clusters in the input
// initial_clustering vector corresponds to an underlying linear ordering of
// clusters and will be used when nearby clusters are paired.
InMemoryClusterer::Clustering ImproveClustersPairwise(
    const GbbsGraph& graph,
    const InMemoryClusterer::Clustering& initial_clustering,
    const LinePartitionerConfig& line_config);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRWISE_IMPROVER_H_
