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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_WEIGHT_THRESHOLD_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_WEIGHT_THRESHOLD_H_

#include "absl/status/statusor.h"
#include "in_memory/clustering/affinity/affinity.pb.h"

namespace graph_mining::in_memory {

// Gives the edge weight threshold used in the provided iteration of affinity
// clustering, depending on the provided config.
// If none of the weight_threshold_config fields are set, returns 0.0.
// Returns an error in case of invalid arguments.
absl::StatusOr<double> AffinityWeightThreshold(
    const research_graph::in_memory::AffinityClustererConfig& config,
    int iteration);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_AFFINITY_WEIGHT_THRESHOLD_H_
