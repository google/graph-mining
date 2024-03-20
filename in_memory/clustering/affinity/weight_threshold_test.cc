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

#include "in_memory/clustering/affinity/weight_threshold.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "in_memory/clustering/affinity/affinity.pb.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/parse_proto/parse_text_proto.h"

namespace graph_mining::in_memory {
namespace {

using ::absl::StatusCode;



TEST(AffinityWeightThreshold, NoThreshold) {
  EXPECT_THAT(AffinityWeightThreshold({}, 0), IsOkAndHolds(0.0));
  EXPECT_THAT(AffinityWeightThreshold({}, 2), IsOkAndHolds(0.0));
}

TEST(AffinityWeightThreshold, FixedThreshold) {
  EXPECT_THAT(
      AffinityWeightThreshold(PARSE_TEXT_PROTO("weight_threshold: 1.5"), 0),
      IsOkAndHolds(1.5));
  EXPECT_THAT(
      AffinityWeightThreshold(PARSE_TEXT_PROTO("weight_threshold: 1.5"), 1),
      IsOkAndHolds(1.5));
  EXPECT_THAT(
      AffinityWeightThreshold(PARSE_TEXT_PROTO("weight_threshold: 1.5"), 100),
      IsOkAndHolds(1.5));
}

TEST(AffinityWeightThreshold, PerIterationThresholdEmpty) {
  EXPECT_THAT(AffinityWeightThreshold(
                  PARSE_TEXT_PROTO("per_iteration_weight_thresholds: {} "), 0),
              IsOkAndHolds(0.0));
  EXPECT_THAT(AffinityWeightThreshold(
                  PARSE_TEXT_PROTO("per_iteration_weight_thresholds: {} "), 2),
              IsOkAndHolds(0.0));
}

TEST(AffinityWeightThreshold, PerIterationThreshold) {
  EXPECT_THAT(
      AffinityWeightThreshold(
          PARSE_TEXT_PROTO(
              "per_iteration_weight_thresholds: { thresholds: [ 3, 2, 1 ] } "),
          0),
      IsOkAndHolds(3.0));
  EXPECT_THAT(
      AffinityWeightThreshold(
          PARSE_TEXT_PROTO(
              "per_iteration_weight_thresholds: { thresholds: [ 3, 2, 1 ] } "),
          1),
      IsOkAndHolds(2.0));
  EXPECT_THAT(
      AffinityWeightThreshold(
          PARSE_TEXT_PROTO(
              "per_iteration_weight_thresholds: { thresholds: [ 3, 2, 1 ] } "),
          2),
      IsOkAndHolds(1.0));
  EXPECT_THAT(
      AffinityWeightThreshold(
          PARSE_TEXT_PROTO(
              "per_iteration_weight_thresholds: { thresholds: [ 3, 2, 1 ] } "),
          3),
      IsOkAndHolds(1.0));
}

TEST(AffinityWeightThreshold, InvalidArgument) {
  EXPECT_THAT(AffinityWeightThreshold({}, -1),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace graph_mining::in_memory
