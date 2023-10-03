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

// Classes to pair a set of values by various schemes. This is useful for
// example to pair a set of clusters to do post processing on them in parallel.

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRING_SCHEME_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRING_SCHEME_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"

namespace graph_mining::in_memory {

// Abstract base class for a PairingScheme. Subclasses should implement
// CycleSize() and Next() methods. The pairing is returned as a vector
// of pair<int, int> where the first and second entries are always
// from different id ranges within [0, num_ids) interval.
// NOTE: It is not guaranteed that all the ids are paired so callers should
// handle that case.
class PairingScheme {
 public:
  PairingScheme(int num_ids, int num_clusters)
      : num_ids_(num_ids), num_clusters_(num_clusters) {
    ABSL_CHECK_GE(num_ids, num_clusters)
        << "Number of clusters can not be more than number of ids.";
    ABSL_CHECK_GT(num_clusters_, 1)
        << "Number of clusters must be greater than 1";
  }
  virtual ~PairingScheme() = default;

  PairingScheme(const PairingScheme&) = delete;
  PairingScheme& operator=(const PairingScheme&) = delete;

  // Number of pairings returned before the cycle repeats again. For pairing
  // schemes that do not have any cycle (for example random) 0 should be
  // returned.
  virtual int CycleSize() = 0;

  // Each call returns the next pairing of ids. This function should be called
  // together with CycleSize() to control the usage of repeated pairings.
  // The first and second entries in each pair are always from different
  // id ranges within [0, num_ids) interval.
  virtual std::vector<std::pair<int, int>> Next() = 0;

  // Computes and returns the number of ids in a given cluster index.
  static uint32_t ComputeClusterSize(uint64_t num_ids, uint32_t num_clusters,
                                     uint32_t cluster_index);

 protected:
  // Computes and returns the cluster size for a given cluster index used
  // in this pairing scheme.
  uint32_t ComputeClusterSizeForPairing(uint32_t cluster_index);

  const int num_ids_;
  const int num_clusters_;
};

// At even numbered steps it pairs (2*i, 2*i+1) for i=0,1,.. and at
// odd numbered steps it pairs (2*i-1, 2*i) for i=1,2,...
// A "step" corresponds to the number of times "Next()" function is called
// after its construction, with the first call treated as even step
// (i.e. 0-based). Note that if the number of clusters is odd then the ids
// in the last cluster won't be paired at even numbered steps and the ids in
// the first cluster won't be paired at odd numbered steps.
// NOTE: It is not guaranteed that all the ids are paired so callers should
// handle that case.
class OddEvenPairingScheme : public PairingScheme {
 public:
  OddEvenPairingScheme(int num_ids, int num_clusters);
  ~OddEvenPairingScheme() override = default;
  OddEvenPairingScheme(const OddEvenPairingScheme&) = delete;
  OddEvenPairingScheme& operator=(const OddEvenPairingScheme&) = delete;
  int CycleSize() override { return 2; }
  std::vector<std::pair<int, int>> Next() override;

 private:
  friend class OddEvenPairingSchemeTest;  // To allow mocking random_ for
                                          // deterministic testing.
  int step_ = 0;
  // Used for random shift within each id group (as divided by the num_clusters
  // constructor parameter) so that we don't always pair the same ids at
  // each iteration.
  absl::BitGenRef gen_;  // For mocking
  absl::BitGen bit_gen_;
};

// Generalization of OddEvenPairingScheme to arbitrary distance (where
// OddEvenPairingScheme is distance 1).
// NOTE: It is not guaranteed that all the ids are paired so callers should
// handle that case.
class DistancePairingScheme : public PairingScheme {
 public:
  DistancePairingScheme(int num_ids, int num_clusters, int distance);
  ~DistancePairingScheme() override = default;
  DistancePairingScheme(const DistancePairingScheme&) = delete;
  DistancePairingScheme& operator=(const DistancePairingScheme&) = delete;
  int CycleSize() override { return 2 * distance_; }
  std::vector<std::pair<int, int>> Next() override;

 private:
  friend class DistancePairingSchemeTest;  // To allow mocking random_ for
                                           // deterministic testing.
  uint32_t distance_ = 0;
  uint32_t step_ = 0;
  // Used for random shift within each id group (as divided by the num_clusters
  // constructor parameter) so that we don't always pair the same ids at
  // each iteration.
  absl::BitGenRef gen_;  // For mocking
  absl::BitGen bit_gen_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARLINE_PAIRING_SCHEME_H_
