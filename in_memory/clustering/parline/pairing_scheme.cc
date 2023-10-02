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

#include "in_memory/clustering/parline/pairing_scheme.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/random/distributions.h"

namespace graph_mining::in_memory {

// static
uint32_t PairingScheme::ComputeClusterSize(uint64_t num_ids,
                                           uint32_t num_clusters,
                                           uint32_t index) {
  uint32_t cluster_size = ceil(static_cast<double>(num_ids) / num_clusters);
  if (index * cluster_size >= num_ids) return 0;
  return std::min(static_cast<uint32_t>(num_ids), (index + 1) * cluster_size) -
         index * cluster_size;
}

uint32_t PairingScheme::ComputeClusterSizeForPairing(uint32_t index) {
  return ComputeClusterSize(num_ids_, num_clusters_, index);
}

OddEvenPairingScheme::OddEvenPairingScheme(int num_ids, int num_clusters)
    : PairingScheme(num_ids, num_clusters), gen_(bit_gen_) {}

std::vector<std::pair<int, int>> OddEvenPairingScheme::Next() {
  std::vector<std::pair<int, int>> pairs;
  pairs.reserve(num_ids_ / 2);
  // First cluster will always have the normal cluster size.
  uint32_t num_ids_in_cluster = ComputeClusterSizeForPairing(0);
  int random_shift = absl::Uniform<uint32_t>(gen_, 0, num_ids_in_cluster);
  int num_paired_clusters = num_clusters_ - num_clusters_ % 2;
  for (int i = 0; i < num_paired_clusters; i += 2) {
    uint32_t p1 = i + step_;
    uint32_t p2 = (p1 + 1) % num_clusters_;
    uint32_t min_cluster_size = std::min(ComputeClusterSizeForPairing(p1),
                                         ComputeClusterSizeForPairing(p2));
    for (int j = 0; j < min_cluster_size; ++j) {
      uint32_t i1 = p1 * num_ids_in_cluster + j;
      uint32_t i2 =
          p2 * num_ids_in_cluster + ((j + random_shift) % min_cluster_size);
      pairs.push_back(std::make_pair(i1, i2));
    }
  }
  step_ = 1 - step_;  // Switch between odd/even
  return pairs;
}

DistancePairingScheme::DistancePairingScheme(int num_ids, int num_clusters,
                                             int distance)
    : PairingScheme(num_ids, num_clusters), gen_(bit_gen_) {
  ABSL_CHECK_GT(distance, 0) << "distance must be greater than 0";
  distance_ = std::min(num_clusters_ / 2, distance);
}

std::vector<std::pair<int, int>> DistancePairingScheme::Next() {
  int distance = distance_ - step_ / 2;
  // No block offset when there are only two clusters.
  int block_offset = num_clusters_ == 2 ? 0 : (step_ % 2) * distance;
  std::vector<std::pair<int, int>> pairs;
  pairs.reserve(num_ids_ / 2);
  // First cluster will always have the normal cluster size.
  uint32_t num_ids_in_cluster = ComputeClusterSizeForPairing(0);
  int random_shift = absl::Uniform<uint32_t>(gen_, 0, num_ids_in_cluster);
  for (int i = block_offset; i < num_clusters_ + block_offset;
       i += 2 * distance) {
    int block = std::min(distance, (num_clusters_ + block_offset - i) / 2);
    for (int j = 0; j < block; ++j) {
      int p1 = (i + j) % num_clusters_;
      int p2 = (p1 + block) % num_clusters_;
      uint32_t min_cluster_size = std::min(ComputeClusterSizeForPairing(p1),
                                           ComputeClusterSizeForPairing(p2));
      for (int k = 0; k < min_cluster_size; ++k) {
        int i1 = p1 * num_ids_in_cluster + k;
        int i2 =
            p2 * num_ids_in_cluster + ((k + random_shift) % min_cluster_size);
        pairs.push_back(std::make_pair(i1, i2));
      }
    }
  }
  step_ = (step_ + 1) % (2 * distance_);
  return pairs;
}

}  // namespace graph_mining::in_memory
