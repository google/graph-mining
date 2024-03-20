/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_COLOR_UTILS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_COLOR_UTILS_H_

#include <cstddef>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "in_memory/clustering/types.h"
#include "src/farmhash.h"

namespace graph_mining::in_memory {

class DynamicHacNodeColorBase {
 public:
  // Represents the color of node
  enum class NodeColor {
    kNoColor,
    kBlue,
    kRed,
  };

  // Represents the type of node ids
  using NodeId = graph_mining::in_memory::NodeId;
  using PriorityType = std::size_t;

  virtual ~DynamicHacNodeColorBase() = default;

  // Return a color for node `id`. The same input gives the same output.
  virtual NodeColor GetNodeColor(NodeId id) const = 0;

  // Return a priority for node `id`. The same input gives the same output.
  virtual PriorityType GetNodePriority(NodeId id) const = 0;
};

// Node color and priority are assigned by a hash function.
// Warning: when `seed` is 0, all `id` has the same color, and `id` 0 has the
// same color for all values of `seed_`.
class DynamicHacNodeColor : public DynamicHacNodeColorBase {
 public:
  DynamicHacNodeColor() = default;
  explicit DynamicHacNodeColor(const NodeId seed)
      : seed_(seed),
     seed_hash_code_(util::Fingerprint64(absl::StrCat(seed))) {}

  NodeColor GetNodeColor(const NodeId id) const override {
    return GetNodePriority(id) % 2 == 0 ? NodeColor::kBlue : NodeColor::kRed;
  }

  PriorityType GetNodePriority(const NodeId id) const override {
    PriorityType hash_code = seed_hash_code_;
       hash_code += util::Fingerprint64(absl::StrCat(id));
       hash_code += util::Fingerprint64(absl::StrCat(seed_ ^ id));
    return hash_code;
  }

 private:
  const NodeId seed_ = 1;
  const PriorityType seed_hash_code_ = 0;
};

// It is used for testing purpose only. Node color and priority are assigned at
// initialization.
class DynamicHacNodeColorTest
    : public graph_mining::in_memory::DynamicHacNodeColorBase {
 public:
  explicit DynamicHacNodeColorTest(
      absl::Span<const std::tuple<NodeId, NodeColor, PriorityType>> nodes) {
    for (const auto& [id, color, priority] : nodes) {
      SetNodeColorAndPriority(id, color, priority);
    }
  }

  // The color must be set first.
  NodeColor GetNodeColor(const NodeId id) const override {
    return colors_.find(id)->second;
  }

  // The priority must be set first.
  PriorityType GetNodePriority(const NodeId id) const override {
    return color_priorities_.find(id)->second;
  }

 private:
  // Set the color and priority of node `id`.
  void SetNodeColorAndPriority(const NodeId id, const NodeColor color,
                               const PriorityType priority) {
    colors_[id] = color;
    color_priorities_[id] = priority;
  }

  // The color of each node.
  absl::flat_hash_map<NodeId, NodeColor> colors_;

  // The color priority of each node.
  absl::flat_hash_map<NodeId, size_t> color_priorities_;
};

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_DYNAMIC_HAC_COLOR_UTILS_H_
