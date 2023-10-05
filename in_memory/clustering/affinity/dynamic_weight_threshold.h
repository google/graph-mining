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

#ifndef RESEARCH_GRAPH_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_
#define RESEARCH_GRAPH_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_

#include "absl/status/statusor.h"
#include "in_memory/clustering/affinity/dynamic_weight_threshold.pb.h"

namespace graph_mining::in_memory {

// Computes weight threshold for the given iteration of affinity clustering,
// based on the provided DynamicWeightThresholdConfig.
absl::StatusOr<double> DynamicWeightThreshold(
    const graph_mining::DynamicWeightThresholdConfig& config, int num_iteration,
    int iteration);

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_
