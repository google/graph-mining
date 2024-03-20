// Copyright 2024 Google LLC
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

#include "in_memory/clustering/dynamic/hac/color_utils.h"

#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "in_memory/clustering/types.h"

namespace {

using graph_mining::in_memory::DynamicHacNodeColor;
using graph_mining::in_memory::DynamicHacNodeColorTest;
using graph_mining::in_memory::NodeId;
using NodeColor = DynamicHacNodeColor::NodeColor;
using PriorityType = DynamicHacNodeColor::PriorityType;

// Test GetNodeColor always return the same value.
TEST(DynamicHacNodeColor, GetNodeColorTest) {
  absl::BitGen gen;
  DynamicHacNodeColor color;
  for (NodeId i = 0; i < 10; ++i) {
    const NodeId id = absl::Uniform(gen, 0, 1000000);
    const auto node_color = color.GetNodeColor(id);
    for (std::size_t attempt = 0; attempt < 100; ++attempt) {
      EXPECT_EQ(node_color, color.GetNodeColor(id));
    }
  }
}

// Test GetNodeColor always return the same value as the set value.
TEST(DynamicHacNodeColorTest, GetNodeColorPriorityTest) {
  absl::BitGen gen;
  std::vector<std::tuple<NodeId, NodeColor, PriorityType>> node_colors;
  for (NodeId i = 0; i < 10; ++i) {
    const NodeId id = absl::Uniform(gen, 0, 1000000);
    const NodeColor node_color =
        absl::Uniform(gen, 0, 1) ? NodeColor::kRed : NodeColor::kBlue;
    node_colors.push_back(std::make_tuple(
        id, node_color,
        absl::Uniform(gen, std::numeric_limits<std::size_t>::lowest(),
                      std::numeric_limits<std::size_t>::max())));
  }
  DynamicHacNodeColorTest color(node_colors);
  for (NodeId i = 0; i < 10; ++i) {
    const auto& [id, node_color, priority] = node_colors[i];
    for (size_t attempt = 0; attempt < 100; ++attempt) {
      EXPECT_EQ(node_color, color.GetNodeColor(id));
      EXPECT_EQ(priority, color.GetNodePriority(id));
    }
  }
}

// Test GetPriority always return the same value.
TEST(DynamicHacNodeColor, GetPriorityTest) {
  absl::BitGen gen;
  DynamicHacNodeColor color;
  for (NodeId i = 0; i < 10; ++i) {
    const NodeId id = absl::Uniform(gen, 0, 1000000);
    const auto node_priority = color.GetNodePriority(id);
    for (size_t attempt = 0; attempt < 100; ++attempt) {
      EXPECT_EQ(node_priority, color.GetNodePriority(id));
    }
  }
}

// Test GetNodeColor returns different value for different seeds.
TEST(DynamicHacNodeColor, DifferentSeedTest) {
  int different_count = 0;
  const DynamicHacNodeColor color1(1);

  for (NodeId i = 0; i < 100; ++i) {
    const DynamicHacNodeColor color2(i);
    if (color1.GetNodeColor(i) != color2.GetNodeColor(i)) {
      different_count++;
    }
  }
  EXPECT_GT(different_count, 40);
}

TEST(DynamicHacNodeColor, DifferentColorTest) {
  for (NodeId seed = -50; seed < 50; ++seed) {
    if (seed == 0) continue;
    const DynamicHacNodeColor color1(seed);

    int red_counts = 0;
    for (NodeId i = 0; i < 100; ++i) {
      if (color1.GetNodeColor(i) == NodeColor::kRed) {
        red_counts++;
      }
    }
    EXPECT_GT(red_counts, 0);
    EXPECT_LT(red_counts, 100);
  }
}
}  // namespace
