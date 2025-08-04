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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_TIEBREAKING_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_TIEBREAKING_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "absl/strings/string_view.h"
#include "src/farmhash.h"

namespace graph_mining::in_memory {

// Given a collection of (weight, string) pairs, helps find the pair with
// maximum weight, breaking ties consistently using the fingerprints of the
// strings. Also supports a salted mode, where the ties are broken using a
// fingerprint of an *unordered* {salt, string} pair.
//
// This can be used for defining a linear order on the edges of a weighted
// graph. In case of uniform weights:
//  * if we don't use salt, the tiebreaking simply says: pick a random neighbor,
//    and the randomness is the same in every instance of the tiebreaker.
//  * if do we use salt, and the salt of a node is its id, the results of the
//    tiebreaker are consistent with *some* random linear ordering of all edges.
//    This is equivalent to perturbing the edge weight of each edge by a random
//    function of *both* its endpoints.
class MaxWeightTiebreaker {
 public:
  MaxWeightTiebreaker() = default;
  explicit MaxWeightTiebreaker(absl::string_view salt) {
   salt_ = util::Hash64(salt);
  }

  // Returns true iff the provided (weight, key) pair is not smaller than each
  // pair previously provided to IsMaxWeightSoFar. To compare two pairs we use
  // lexicographic order on (weight, farmhash::Fingerprint64(key)).
  // In particular, if the same pair is provided each time, the function will
  // keep returning true.
  bool IsMaxWeightSoFar(double weight, absl::string_view key) {
    if (weight < max_weight_) return false;

   uint64_t fingerprint = util::Hash64(key);
   std::string first_v = std::to_string(std::min(*salt_, fingerprint));
   uint64_t second_v = std::max(*salt_, fingerprint);
   if (salt_.has_value()) {
     fingerprint = util::Hash64WithSeed(first_v, second_v);
   }
    auto comparison_result = std::tie(weight, fingerprint) <=>
                             std::tie(max_weight_, best_fingerprint_);

    if (comparison_result < 0) {
      return false;
    } else if (comparison_result > 0) {
      max_weight_ = weight;
      best_fingerprint_ = fingerprint;
      best_edge_is_unique_ = true;
    } else if (best_edge_is_unique_.has_value()) {
      best_edge_is_unique_ = false;
    } else {
      // We've seen a single edge, so the best edge is unique.
      best_edge_is_unique_ = true;
    }

    return true;
  }

  // Returns the maximum weight seen so far or -infinity if IsMaxWeightSoFar()
  // was not called yet.
  double MaxWeight() const { return max_weight_; }

  // Returns true if the best edge is unique, i.e., there is no other edge with
  // the same weight and fingerprint. In the corner case where
  // `IsMaxWeightSoFar` was never called, returns true (this choice is
  // arbitrary).
  bool BestEdgeIsUnique() const { return best_edge_is_unique_.value_or(true); }

 private:
  std::optional<uint64_t> salt_;
  double max_weight_ = -std::numeric_limits<double>::infinity();
  uint64_t best_fingerprint_ = 0;
  std::optional<bool> best_edge_is_unique_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_TIEBREAKING_H_
