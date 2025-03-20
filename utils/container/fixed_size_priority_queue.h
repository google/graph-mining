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
#include <functional>
#include <limits>
#include <vector>

#include "absl/log/absl_check.h"

namespace graph_mining {

// Priority queue over a fixed set of elements which, contrary to
// std::priority_queue, supports updating priorities. By default returns items
// of maximum priority (but this can be customized by providing a different
// comparator). Priorities are of type PriorityType, and elements are indexed by
// integers of type IndexType (typically int32_t or int64_t).
template <class PriorityType = double, class IndexType = int32_t,
          class LargerPriority = std::greater<PriorityType>>
  requires std::is_integral_v<IndexType>
class FixedSizePriorityQueue {
 public:
  // The queue can store a subset of 0, ..., size-1.
  explicit FixedSizePriorityQueue(IndexType size)
      : nodes_(2 * size, kInvalidPriority) {}

  // Inserts a new element or updates the priority of a given element.
  // Using priority = kInvalidPriority causes the element to get deleted.
  void InsertOrUpdate(IndexType element, PriorityType priority) {
    ABSL_CHECK_LT(element, NumLeaves());
    ABSL_CHECK_GE(element, 0);

    IndexType id = FirstLeaf() + element;
    nodes_[id] = priority;

    // Update max priorities by traversing the tree up.
    id = Parent(id);
    while (id > 0) {
      const auto& left = nodes_[LeftChild(id)];
      const auto& right = nodes_[RightChild(id)];
      if (left == kInvalidPriority) {
        nodes_[id] = right;
      } else if (right == kInvalidPriority) {
        nodes_[id] = left;
      } else {
        nodes_[id] = LargerPriority()(left, right) ? left : right;
      }
      id = Parent(id);
    }

    // Update top_ by traversing the tree down.
    id = 1;
    while (id < FirstLeaf()) {
      if (nodes_[id] == nodes_[LeftChild(id)]) {
        // Note that when both left and right children are equal to the parent,
        // we descend left. This ensures that for an empty queue, Top() == 0 and
        // in the case when the size of the queue is a power of two we can break
        // ties by returning the element with the smallest index.
        id = LeftChild(id);
      } else {
        id = RightChild(id);
      }
    }
    top_ = id - FirstLeaf();
  }

  // If the element is currently not in the queue and the index is valid, this
  // function will return kInvalidPriority.
  PriorityType Priority(IndexType element) const {
    return nodes_[FirstLeaf() + element];
  }

  // Returns true iff the queue has no elements.
  bool Empty() const {
    return nodes_.size() == 0 || nodes_[1] == kInvalidPriority;
  }

  // Removes the given element. Should not be called on an empty queue. Note
  // that this function silently ignores non-existing elements.
  void Remove(IndexType element) { InsertOrUpdate(element, kInvalidPriority); }

  // Returns the element with the largest priority. Returns 0 on an empty queue.
  // ***If*** the size of the priority queue is a power of two, ties are broken
  // by returning the element with the smallest index.
  IndexType Top() const { return top_; }

  static constexpr PriorityType kInvalidPriority =
      std::numeric_limits<PriorityType>::max();

 private:
  IndexType Parent(IndexType id) const { return id >> 1; }

  IndexType LeftChild(IndexType id) const { return id << 1; }

  IndexType RightChild(IndexType id) const { return (id << 1) + 1; }

  IndexType FirstLeaf() const { return nodes_.size() / 2; }

  IndexType NumLeaves() const { return nodes_.size() / 2; }

  // Stores a complete binary tree as follows. Let n be the number of elements.
  // Then:
  // * 0 is unused,
  // * 1 is the root node,
  // * n, ..., 2*n-1 are the leaves, which store the priorities of the elements.
  // Each non-leaf node stores the maximum of all leaves in its subtree.
  // If the size of the queue is a power of two, then the consecutive leaves
  // (from the leftmost to the rightmost) correspond to the sequence of the
  // elements in the queue. In this case, we can break ties by returning the
  // element with the smallest index.
  std::vector<PriorityType> nodes_;

  // Element with the largest id, or 0 if the queue is empty.
  IndexType top_ = 0;
};

}  // namespace graph_mining

#endif  // RESEARCH_GRAPH_GENERIC_CONTAINER_FIXED_SIZE_PRIORITY_QUEUE_H_
