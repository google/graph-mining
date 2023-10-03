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

#include "in_memory/clustering/dendrogram_flatten.h"

#include <functional>
#include <limits>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/clustering_utils.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/types.h"
#include "in_memory/connected_components/asynchronous_union_find.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

absl::StatusOr<Clustering> FlattenClusteringWithScorer(
    const Dendrogram& dendrogram,
    std::function<bool(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        stopping_condition,
    std::function<double(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        scoring_function) {
  auto merges = dendrogram.GetMergeSequence();
  auto num_nodes = dendrogram.NumClusteredNodes();

  SequentialUnionFind<NodeId> cc(num_nodes);

  std::vector<NodeId> components;
  double max_score = 0.0;

  auto ScoreComponents = [&](double weight) {
    // See if this clustering has positive score (i.e., it's worth keeping), and
    // is higher scoring than the current high-scorer.
    double score = scoring_function(weight, cc);
    if (score > max_score) {
      max_score = score;
      auto components_span = cc.ComponentIds();
      components = std::vector(components_span.begin(), components_span.end());
    }
  };

  for (auto [merge_similarity, node_a, node_b, parent_id, _] : merges) {
    ScoreComponents(merge_similarity);
    if (stopping_condition(merge_similarity, cc)) {
      break;
    }

    cc.Unite(node_a, node_b);
  }
  // Call ScoreComponents a final time for the last edge merged using AddEdge.
  ScoreComponents(-std::numeric_limits<double>::infinity());

  if (max_score == 0.0) {  // Didn't find a clustering.
    ABSL_CHECK(components.empty());
    auto components_span = cc.ComponentIds();
    components = std::vector(components_span.begin(), components_span.end());
  }

  return ClusterIdSequenceToClustering(
      absl::MakeSpan(components).subspan(0, num_nodes));
}

absl::StatusOr<std::vector<Clustering>> HierarchicalFlattenClusteringWithScorer(
    const Dendrogram& dendrogram,

    const std::vector<double>& similarity_thresholds,
    std::function<double(double /*weight*/, SequentialUnionFind<NodeId>& cc)>
        scoring_function) {
  // Enforce that the thresholds are in non-ascending order.
  if (similarity_thresholds.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("similarity_thresholds must be non-empty."));
  }

  if (!absl::c_is_sorted(similarity_thresholds, std::greater<double>())) {
    return absl::InvalidArgumentError(
        absl::StrCat("similarity_thresholds must be in descending order."));
  }

  auto merges = dendrogram.GetMergeSequence();
  auto num_nodes = dendrogram.NumClusteredNodes();

  SequentialUnionFind<NodeId> cc(num_nodes);

  std::vector<Clustering> clusterings;

  // |components| holds (connected components of) the highest positive scoring
  // clustering found so far, and |max_score| holds the score of |components|.
  std::vector<NodeId> components;
  double max_score = 0.0;
  // Function to update |components| and |max_score| with the current clustering
  // stored in |cc| (ignores clusterings with non-positive score since max_score
  // is initialized to 0.0).
  auto ScoreComponents = [&](double weight) {
    double score = scoring_function(weight, cc);
    if (score > max_score) {
      max_score = score;
      auto components_span = cc.ComponentIds();
      components = std::vector(components_span.begin(), components_span.end());
    }
  };

  int threshold_index = 0;
  for (auto [merge_similarity, node_a, node_b, parent_id, _] : merges) {
    ScoreComponents(merge_similarity);

    if (merge_similarity < similarity_thresholds[threshold_index]) {
      if (max_score == 0.0) {
        // Didn't find a clustering; just use the current one.
        auto components_span = cc.ComponentIds();
        components =
            std::vector(components_span.begin(), components_span.end());
      }
      Clustering clustering;
      ASSIGN_OR_RETURN(clustering,
                       ClusterIdSequenceToClustering(
                           absl::MakeSpan(components).subspan(0, num_nodes)));

      while (threshold_index < similarity_thresholds.size() &&
             merge_similarity < similarity_thresholds[threshold_index]) {
        clusterings.push_back(clustering);
        threshold_index++;
      }

      if (threshold_index >= similarity_thresholds.size()) break;
    }

    cc.Unite(node_a, node_b);
  }

  if (threshold_index < similarity_thresholds.size()) {
    // We got to the end of the merges without reaching some of the thresholds ;
    // just return what we ended up with for the remaining thresholds.

    // Need to call ScoreComponents one more time to capture the last AddEdge().
    ScoreComponents(-std::numeric_limits<double>::infinity());
    if (max_score == 0.0) {
      // Didn't find a clustering, just use current one.
      auto components_span = cc.ComponentIds();
      components = std::vector(components_span.begin(), components_span.end());
    }

    Clustering clustering;
    ASSIGN_OR_RETURN(clustering,
                     ClusterIdSequenceToClustering(
                         absl::MakeSpan(components).subspan(0, num_nodes)));

    for (; threshold_index < similarity_thresholds.size(); ++threshold_index) {
      clusterings.push_back(clustering);
    }
  }

  return clusterings;
}

}  // namespace graph_mining::in_memory
