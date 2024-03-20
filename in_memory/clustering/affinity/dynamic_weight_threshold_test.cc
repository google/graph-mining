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

#include "in_memory/clustering/affinity/dynamic_weight_threshold.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "in_memory/clustering/affinity/dynamic_weight_threshold.pb.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep
#include "utils/parse_proto/parse_text_proto.h"

namespace graph_mining::in_memory {
namespace {

using ::absl::StatusCode;
using ::testing::DoubleNear;



TEST(DynamicWeightThresholdTest, TestLinearDecay) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: LINEAR_DECAY
    upper_bound: 2.0
    lower_bound: 1.0
  )pb");

  std::vector<double> expected_weight_thresholds = {2.0, 1.8, 1.6,
                                                    1.4, 1.2, 1.0};

  for (int round = 0; round < expected_weight_thresholds.size(); ++round) {
    EXPECT_THAT(DynamicWeightThreshold(
                    config, expected_weight_thresholds.size(), round),
                IsOkAndHolds(expected_weight_thresholds[round]));
  }
}

TEST(DynamicWeightThresholdTest, TestLinearDecayReverse) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: LINEAR_DECAY
    upper_bound: 1.0
    lower_bound: 2.0
  )pb");

  std::vector<double> expected_weight_thresholds = {1.0, 1.2, 1.4,
                                                    1.6, 1.8, 2.0};

  for (int round = 0; round < expected_weight_thresholds.size(); ++round) {
    EXPECT_THAT(DynamicWeightThreshold(
                    config, expected_weight_thresholds.size(), round),
                IsOkAndHolds(expected_weight_thresholds[round]));
  }
}

TEST(DynamicWeightThresholdTest, TestExponentialDecay) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: EXPONENTIAL_DECAY
    upper_bound: 10.0
    lower_bound: 1.0
  )pb");

  std::vector<double> expected_weight_thresholds = {10.0,    6.30957, 3.98107,
                                                    2.51189, 1.58489, 1.0};

  for (int round = 0; round < expected_weight_thresholds.size(); ++round) {
    EXPECT_THAT(
        DynamicWeightThreshold(config, expected_weight_thresholds.size(),
                               round),
        IsOkAndHolds(DoubleNear(expected_weight_thresholds[round], 1e-5)));
  }
}

TEST(DynamicWeightThresholdTest, TestExponentialDecayReverse) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: EXPONENTIAL_DECAY
    upper_bound: 1.0
    lower_bound: 8.0
  )pb");

  std::vector<double> expected_weight_thresholds = {1.0, 2.0, 4.0, 8.0};
  for (int round = 0; round < expected_weight_thresholds.size(); ++round) {
    EXPECT_THAT(
        DynamicWeightThreshold(config, expected_weight_thresholds.size(),
                               round),
        IsOkAndHolds(DoubleNear(expected_weight_thresholds[round], 1e-5)));
  }
}

TEST(DynamicWeightThresholdTest, TestExponentialDecayThreeIterations) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: EXPONENTIAL_DECAY
    upper_bound: 10.0
    lower_bound: 1.0
  )pb");

  std::vector<double> expected_weight_thresholds = {10.0, 3.16228, 1.0};

  for (int round = 0; round < expected_weight_thresholds.size(); ++round) {
    EXPECT_THAT(
        DynamicWeightThreshold(config, expected_weight_thresholds.size(),
                               round),
        IsOkAndHolds(DoubleNear(expected_weight_thresholds[round], 1e-5)));
  }
}

TEST(DynamicWeightThresholdTest, InvalidIterations) {
  EXPECT_THAT(DynamicWeightThreshold({}, 3, 3),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(DynamicWeightThreshold({}, 3, -1),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(DynamicWeightThreshold({}, 0, 0),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(DynamicWeightThreshold({}, 0, 1),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(DynamicWeightThreshold({}, -1, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(DynamicWeightThresholdTest, OneIteration) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: EXPONENTIAL_DECAY
    upper_bound: 10.0
    lower_bound: 1.0
  )pb");

  EXPECT_THAT(DynamicWeightThreshold(config, 1, 0),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(DynamicWeightThreshold(config, 2, 0), IsOkAndHolds(10.0));
}

TEST(DynamicWeightThresholdTest, WeightDecayFunctionUnset) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    upper_bound: 10.0
    lower_bound: 1.0
  )pb");

  EXPECT_THAT(DynamicWeightThreshold(config, 2, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(DynamicWeightThresholdTest, NegativeBounds) {
  graph_mining::DynamicWeightThresholdConfig config = PARSE_TEXT_PROTO(R"pb(
    weight_decay_function: EXPONENTIAL_DECAY
    upper_bound: 10.0
    lower_bound: -1.0
  )pb");

  EXPECT_THAT(DynamicWeightThreshold(config, 2, 0),
              StatusIs(StatusCode::kInvalidArgument));

  config.set_upper_bound(-1);
  config.set_lower_bound(10);

  EXPECT_THAT(DynamicWeightThreshold(config, 2, 0),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace graph_mining::in_memory
