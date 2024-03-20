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

#ifndef THIRD_PARTY_GRAPH_MINING_UTILS_CONTAINER_FIXED_SIZE_PRIORITY_QUEUE_H_
#define THIRD_PARTY_GRAPH_MINING_UTILS_CONTAINER_FIXED_SIZE_PRIORITY_QUEUE_H_

#include <cstdint>
#include <vector>

#include "absl/log/check.h"

namespace graph_mining {

// Priority queue over a fixed set of elements which, contrary to
// std::priority_queue, supports updating priorities. This can be used to obtain
// items of maximum priority (uses std::less<T> as the comparison function).
template <class T = double>
class FixedSizePriorityQueue {
 public:
  // The queue can store a subset of 0, ..., size-1.
  explicit FixedSizePriorityQueue(int32_t size)
      : nodes_(2 * size, kInvalidPriority) {}

  // Inserts a new element or updates the priority of a given element.
  // Using priority = std::numeric_limits<T>::max() causes the element to get
  // deleted.
  void InsertOrUpdate(int32_t element, T priority) {
    CHECK_LT(element, NumLeaves());
    CHECK_GE(element, 0);

    int32_t id = FirstLeaf() + element;
    nodes_[id] = priority;

    // Update max priorities by traversing the tree up.
    id = Parent(id);
    while (id > 0) {
      const auto& left = nodes_[LeftChild(id)];
      const auto& right = nodes_[RightChild(id)];
      if (left == kInvalidPriority || right == kInvalidPriority) {
        nodes_[id] = std::min(left, right);
      } else {
        nodes_[id] = std::max(left, right);
      }
      id = Parent(id);
    }

    // Update top_ by traversing the tree down.
    id = 1;
    while (id < FirstLeaf()) {
      if (nodes_[id] == nodes_[LeftChild(id)]) {
        // Note that when both left and right children are equal to the parent,
        // we descend left. This ensures that for an empty queue, Top() == 0.
        id = LeftChild(id);
      } else {
        id = RightChild(id);
      }
    }
    top_ = id - FirstLeaf();
  }

  // If the element is currently not in the queue and the index is valid, this
  // function will return std::numeric_limits<T>::max().
  T Priority(int32_t element) const { return nodes_[FirstLeaf() + element]; }

  // Returns true iff the queue has no elements.
  bool Empty() const {
    return nodes_.size() == 0 || nodes_[1] == kInvalidPriority;
  }

  // Removes the given element. Should not be called on an empty queue. Note
  // that this function silently ignores non-existing elements.
  void Remove(int32_t element) { InsertOrUpdate(element, kInvalidPriority); }

  // Returns the element with the largest priority. Returns 0 on an empty queue.
  int32_t Top() const { return top_; }

 private:
  int32_t Parent(int32_t id) const { return id >> 1; }

  int32_t LeftChild(int32_t id) const { return id << 1; }

  int32_t RightChild(int32_t id) const { return (id << 1) + 1; }

  int32_t FirstLeaf() const { return nodes_.size() / 2; }

  int32_t NumLeaves() const { return nodes_.size() / 2; }

  // Stores a complete binary tree as follows. Let n be the number of elements.
  // Then:
  // * 0 is unused,
  // * 1 is the root node,
  // * n, ..., 2*n-1 are the leaves.
  // Each non-leaf node stores the maximum of all leaves in its subtree.
  std::vector<T> nodes_;

  // Element with the largest id, or 0 if the queue is empty.
  int32_t top_ = 0;

  static constexpr T kInvalidPriority = std::numeric_limits<T>::max();
};

}  // namespace graph_mining

#endif  // RESEARCH_GRAPH_GENERIC_CONTAINER_FIXED_SIZE_PRIORITY_QUEUE_H_
