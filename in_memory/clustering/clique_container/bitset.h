#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_BITSET_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_BITSET_H_

#include <bit>
#include <cstdint>
#include <vector>

#include "absl/log/absl_check.h"

namespace graph_mining::in_memory {

// A bit-packed representation of a set of integers from a fixed range.
// Warning: since this is intended to be when performance is key, for
// efficiency, this class does not validate the parameters passed to its
// methods.
// Contrary to existing bitset implementations (in standard library and boost),
// supports efficient iteration over the elements.
class BitSet {
 public:
  // Constructs an empty bitset that can store a subset of {0, ...,
  // universe_size - 1}.
  explicit BitSet(int universe_size) : bits_((universe_size + 63) / 64, 0) {}

  // Inserts `element_id` into the bitset. Does nothing if `element_id` is
  // already present.
  void Insert(int element_id) {
    uint64_t& word = bits_[element_id >> 6];
    const uint64_t mask = 1ULL << (element_id & 63);
    if (!(word & mask)) {
      word |= mask;
      ++size_;
    }
  }

  // Returns true if `element_id` is present in the bitset. `element_id` must be
  // in the range [0, universe_size).
  bool Contains(int element_id) const {
    return bits_[element_id >> 6] & (1ULL << (element_id & 63));
  }

  int Size() const { return size_; }

  // Applies `f` to each element of the bitset, in ascending order. The argument
  // to `f` is the index of the element in the bitset.
  template <typename F>
  void MapElements(F&& f) const {
    for (int i = 0; i < bits_.size(); ++i) {
      IterateBits(bits_[i], i << 6, f);
    }
  }

  // Applies `f` to each element of the bitset that is also present in `other`,
  // in ascending order. The argument to `f` is the index of the element in the
  // bitset. Requires that `other` has the same universe size as this bitset.
  template <typename F>
  void MapCommonElements(const BitSet& other, F&& f) const {
    ABSL_CHECK_EQ(bits_.size(), other.bits_.size());
    for (int i = 0; i < bits_.size(); ++i) {
      IterateBits(bits_[i] & other.bits_[i], i << 6, f);
    }
  }

  // Returns true if all elements of `other` are also present in this bitset.
  // Requires that `other` has the same universe size as this bitset.
  bool IsSupersetOf(const BitSet& other) const {
    ABSL_CHECK_EQ(bits_.size(), other.bits_.size());
    for (int i = 0; i < bits_.size(); ++i) {
      if ((bits_[i] & other.bits_[i]) != other.bits_[i]) {
        return false;
      }
    }
    return true;
  }

 private:
  // Calls `f` for each bit set in `block`, offset by `offset`.
  template <typename F>
  void IterateBits(uint64_t block, int offset, F&& f) const {
    while (block) {
      int index = std::countr_zero(block);
      f(index + offset);
      block &= block - 1;
    }
  }

  int size_ = 0;
  std::vector<uint64_t> bits_;
};
}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_CLIQUE_CONTAINER_BITSET_H_
