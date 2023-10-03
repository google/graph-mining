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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_INTERNAL_H_

#include <algorithm>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "gbbs/bridge.h"
#include "gbbs/gbbs.h"
#include "gbbs/macros.h"

namespace graph_mining {
namespace in_memory {

// This file contains definitions of helper objects that are used in the
// parallel ClusteredGraph representation (see parallel-clustered-graph.h).
// This file specifically presents definitions of different weight types,
// which correspond to different aggregation strategies when merging nodes
// and updating the weights of merged edges (for example average-linkage,
// max-linkage, etc.).
//
// First, we define various weight types used to manage the edge weights in
// the ClusteredGraph. Each weight struct must define:
//   StoredWeightType : datatype required to initialize the object and also
//      perform merges.
//   static StoredWeightType MergeWeights(StoredWeightType, StoredWeightType) :
//      adds two elements, must be associative
//   StoredWeightType GetMergeData() const : gets the underlying data for an
//      edge (used to perform merges).
//   double Similarity(size_t) const : takes the size of the source cluster and
//      returns the similarity of the edge
//   void UpdateWeight(StoredWeightType) : updates the weight given (merged)
//      data about new edges across this cut.
//
// The remaining definitions indicate that the merge algorithm should perform
// neighbor or edge updates:
//   static constexpr bool kRequiresNeighborUpdate : for each edge e = (u,v)
//       where u is an active node affected by a merge, call UpdateNeighborSize.
//   static constexpr bool kRequiresEndpointsUpdate : for each edge e = (u,v)
//       where u is an active node affected by a merge, call
//       UpdateEndpointsSize, which takes the current cluster sizes of both
//       endpoints.
//   void UpdateNeighborSize(uintE) : update the weight given the neighbor
//       cluster size. Used by AverageLinkageWeight.
//   void UpdateEndpointsSize(uintE, uintE) : update the weight given the size
//       of the source and neighbor clusters. Used by CutSparsityLinkageWeight.
//
// TODO: Define an abstract class that weight classes inherit from
// using non-virtual inheritance + final. This would help enforce definitions of
// MergeWeights, GetMergeData, etc.
//
// TODO: Add more unit tests for other weight classes and
// SimpleConcurrentTable.
class AverageLinkageWeight {
 public:
  using uintE = gbbs::uintE;
  using StoredWeightType = float;

  constexpr AverageLinkageWeight() : cut_weight_(0), neighbor_size_(1) {}
  explicit AverageLinkageWeight(StoredWeightType cut_weight)
      : cut_weight_(cut_weight), neighbor_size_(1) {}

  StoredWeightType GetMergeData() const { return cut_weight_; }

  // Return the actual weight of this edge, given the (source) cluster size.
  double Similarity(uintE cluster_size) const {
    return OneSidedWeight() / static_cast<double>(cluster_size);
  }

  // Update the stored weight given the weight of a new edge across this cut.
  void UpdateWeight(StoredWeightType contribution) {
    gbbs::write_add(&cut_weight_, contribution);
  }

  // Update the stored neighbor size. This method should be a noop for weights
  // that do not store the neighbor size.
  void UpdateNeighborSize(uintE size) {
    gbbs::atomic_store(&neighbor_size_, size);
  }

  static StoredWeightType MergeWeights(StoredWeightType l, StoredWeightType r) {
    return l + r;
  }

  // Returns true iff l is smaller than r.
  static bool Smaller(AverageLinkageWeight l, AverageLinkageWeight r) {
    return l.OneSidedWeight() < r.OneSidedWeight();
  }

  // Indicates that the algorithm should call UpdateNeighborSize (below) on
  // any edge incident to an active merge participating in merges.
  static constexpr bool kRequiresNeighborUpdate = true;
  // Used only for cut_sparsity.
  static constexpr bool kRequiresEndpointsUpdate = false;

 private:
  // Return the partially normalized weight. To obtain the correct
  // average linkage weight we still need to multiply by
  // (1/source_cluster_size). However, the OneSidedWeight is sufficient for
  // selecting the max_weight edge incident to a node.
  double OneSidedWeight() const {
    return cut_weight_ / static_cast<double>(neighbor_size_);
  }

  // Total weight on this edge to the neighbor.
  StoredWeightType cut_weight_;
  // The size of the neighbor's cluster.
  uintE neighbor_size_;
};

class MaxLinkageWeight {
 public:
  using uintE = gbbs::uintE;
  using StoredWeightType = float;

  // Not needed for max-linkage (no cluster-size updates are necessary)
  static constexpr bool kRequiresNeighborUpdate = false;
  static constexpr bool kRequiresEndpointsUpdate = false;

  constexpr MaxLinkageWeight() : cut_weight_(0) {}
  explicit MaxLinkageWeight(float cut_weight) : cut_weight_(cut_weight) {}

  StoredWeightType GetMergeData() const { return cut_weight_; }

  double Similarity(uintE cluster_size) const { return cut_weight_; }

  void UpdateWeight(float contribution) {
    cut_weight_ = std::max(cut_weight_, contribution);
  }

  static StoredWeightType MergeWeights(StoredWeightType l, StoredWeightType r) {
    return std::max(l, r);
  }

  static bool Smaller(MaxLinkageWeight l, MaxLinkageWeight r) {
    return l.cut_weight_ < r.cut_weight_;
  }

  static MaxLinkageWeight max(MaxLinkageWeight l, MaxLinkageWeight r) {
    return (l.cut_weight_ > r.cut_weight_) ? l : r;
  }

 private:
  // Total weight on this edge.
  StoredWeightType cut_weight_;
};

class SumLinkageWeight {
 public:
  using uintE = gbbs::uintE;
  using StoredWeightType = float;

  // Not needed for sum-linkage (no cluster-size updates are necessary)
  static constexpr bool kRequiresNeighborUpdate = false;
  static constexpr bool kRequiresEndpointsUpdate = false;

  constexpr SumLinkageWeight() : cut_weight_(0) {}
  explicit SumLinkageWeight(StoredWeightType cut_weight)
      : cut_weight_(cut_weight) {}

  StoredWeightType GetMergeData() const { return cut_weight_; }

  double Similarity(uintE cluster_size) const { return cut_weight_; }

  void UpdateWeight(StoredWeightType contribution) {
    cut_weight_ += contribution;
  }

  static bool Smaller(SumLinkageWeight l, SumLinkageWeight r) {
    return l.cut_weight_ < r.cut_weight_;
  }

  static StoredWeightType MergeWeights(StoredWeightType l, StoredWeightType r) {
    return l + r;
  }

 private:
  // Total weight on this edge.
  StoredWeightType cut_weight_;
};

class CutSparsityLinkageWeight {
 public:
  using uintE = gbbs::uintE;
  using StoredWeightType = float;
  using Weight = CutSparsityLinkageWeight;

  static constexpr bool kRequiresNeighborUpdate = false;
  static constexpr bool kRequiresEndpointsUpdate = true;

  constexpr CutSparsityLinkageWeight() : cut_weight_(0), min_size_(1) {}
  explicit CutSparsityLinkageWeight(StoredWeightType cut_weight)
      : cut_weight_(cut_weight), min_size_(1) {}

  StoredWeightType GetMergeData() const { return cut_weight_; }

  double Similarity(uintE cluster_size) const { return GetWeight(); }

  // Update min_size based on the size of both endpoints.
  void UpdateEndpointsSize(uintE source_size, uintE neighbor_size) {
    min_size_ = std::min(source_size, neighbor_size);
  }

  void UpdateWeight(StoredWeightType contribution) {
    cut_weight_ += contribution;
  }

  static bool Smaller(CutSparsityLinkageWeight l, CutSparsityLinkageWeight r) {
    return l.GetWeight() < r.GetWeight();
  }

  static StoredWeightType MergeWeights(StoredWeightType l, StoredWeightType r) {
    return l + r;
  }

 private:
  double GetWeight() const {
    return cut_weight_ / static_cast<double>(min_size_);
  }

  // Total weight on this edge to the neighbor.
  StoredWeightType cut_weight_;
  // Always equal to  min(source_cluster_size, neighbor_cluster_size).
  uintE min_size_;
};

class ExplicitAverageLinkageWeight {
 public:
  using uintE = gbbs::uintE;
  using Weight = ExplicitAverageLinkageWeight;
  using StoredWeightType = std::pair<float, uintE>;

  static constexpr bool kRequiresNeighborUpdate = false;
  static constexpr bool kRequiresEndpointsUpdate = false;

  constexpr ExplicitAverageLinkageWeight() : cut_weight_(0), cut_size_(1) {}
  explicit ExplicitAverageLinkageWeight(float cut_weight)
      : cut_weight_(cut_weight), cut_size_(1) {}
  explicit ExplicitAverageLinkageWeight(StoredWeightType weight_and_size)
      : cut_weight_(weight_and_size.first), cut_size_(weight_and_size.second) {}

  StoredWeightType GetMergeData() const { return {cut_weight_, cut_size_}; }

  // Return the actual weight of this edge, given the (source) cluster size.
  double Similarity(uintE cluster_size) const { return GetWeight(); }

  void UpdateWeight(StoredWeightType contribution) {
    cut_weight_ += contribution.first;
    cut_size_ += contribution.second;
  }

  static bool Smaller(ExplicitAverageLinkageWeight l,
                      ExplicitAverageLinkageWeight r) {
    return l.GetWeight() < r.GetWeight();
  }

  static StoredWeightType MergeWeights(StoredWeightType l, StoredWeightType r) {
    const auto& [l_weight, l_size] = l;
    const auto& [r_weight, r_size] = r;
    return {l_weight + r_weight, l_size + r_size};
  }

 private:
  double GetWeight() const {
    return cut_weight_ / static_cast<double>(cut_size_);
  }

  // Total weight on this edge to the neighbor.
  float cut_weight_;
  // The number of edges going across the cut.
  uintE cut_size_;
};

namespace internal {

// SimpleConcurrentTable is a lightweight implementation of a concurrent
// linear-probing hash table storing a set of key-value tuples of type K and V.
// There are some important assumptions made in this table design:
//   1. K, the key-type needs to be an arithmetic type.
//   2. The table reserves std::numeric_limits<K>::max() and
//   std::numeric_limits<K>::max() - 1 as special values.
//   3. The hash-table does not support concurrent resizing. Users that are
//   inserting new elements into the table must call the "AdjustSizeForIncoming"
//   function with an upper-bound on the number of new elements before
//   performing the inserts to guarantee that the table has sufficient space.
//   4. The table size is always a power-of-two.
template <class K, class V>
class SimpleConcurrentTable {
 public:
  // Use std::tuple to support empty-base optimization if sizeof(V) == 0 (i.e.
  // the table is representing a set, not a map).
  using T = std::tuple<K, V>;

  SimpleConcurrentTable() : table_(parlay::sequence<T>()) {}

  // initial_size is the maximum number of values the hash table will hold.
  // Overfilling the table will put it into an infinite loop.
  explicit SimpleConcurrentTable(size_t initial_size) {
    initial_size = SizeForElements(initial_size);
    table_ = parlay::sequence<T>::uninitialized(initial_size);
    ClearTable();
  }

  // Total number of cells in the table.
  size_t Capacity() const { return table_.size(); }
  // Total number of non-empty cells in table.
  size_t Size() const {
    size_t count = 0;
    for (size_t i = 0; i < Capacity(); ++i) {
      if (Nonempty(table_[i])) {
        count++;
      }
    }
    return count;
  }

  // Relinquishes the underlying memory for the table.
  void Clear() { table_ = parlay::sequence<T>(); }

  // Inserts the key-value pair kv into the table. If the key k is already
  // present in the table, the algorithm calls value_update_fn : V* -> void
  // on the key's value.
  // Returns true if this operation inserted a new key (previously not present).
  bool InsertOrUpdate(K k, V v, std::function<void(V*)> value_update_fn) {
    size_t h = FirstIndex(k);
    // Try updating the value of k in the table using value update fn.
    if (!Update(k, value_update_fn)) {
      // Otherwise, k doesn't exist in the table. Insert it.
      while (true) {
        auto read_key = Key(table_[h]);
        if (read_key >= kTombstone) {
          if (gbbs::atomic_compare_and_swap(MutableKey(table_[h]), read_key,
                                            k)) {
            *(MutableValue(table_[h])) = v;
            return true;
          }
        } else if (Key(table_[h]) == k) {
          value_update_fn(MutableValue(table_[h]));
          return false;
        }
        h = IncrementIndex(h);
      }
    }
    return false;
  }

  // Remove the key k from the table, returning true if this operation
  // (atomically) removed k, and false otherwise.
  bool Remove(K k) {
    T* entry = Find(k);
    if (entry) {
      return gbbs::atomic_compare_and_swap(MutableKey(*entry), k, kTombstone);
    }
    return false;
  }

  // Returns true iff the table contains k.
  bool Contains(K k) { return Find(k) != nullptr; }

  // Finds the key k and applies value_update_fn (ValueUpdate : V* -> void).
  // Returns true if k already existed in the table, and false otherwise.
  bool Update(K k, std::function<void(V*)> value_update_fn) {
    T* entry = Find(k);
    if (entry) {
      value_update_fn(MutableValue(*entry));
      return true;
    }
    return false;
  }

  parlay::sequence<T> Entries() const {
    auto pred = [&](const T& t) { return Nonempty(t); };
    return parlay::filter(table_, pred);
  }

  // Class Reduce must be a structure representing a monoid with:
  //   T = a type
  //   T identity;
  //   T add(T, T);  an associative operation on T
  // Function map_fn must be of type (K&, V&) -> T
  template <class Reduce, class MapFn>
  typename Reduce::T MapReduce(const MapFn map_fn, Reduce reduce) const {
    using T = typename Reduce::T;
    auto seq = parlay::delayed_seq<T>(Capacity(), [&](size_t i) {
      if (!Nonempty(table_[i])) return reduce.identity;
      auto& [key, value] = table_[i];
      return map_fn(key, value);
    });
    return parlay::reduce(seq, reduce);
  }

  void MapIndex(std::function<void(K, V&, size_t)> map_fn) {
    size_t count = 0;
    for (size_t i = 0; i < Capacity(); i++) {
      if (Nonempty(table_[i])) {
        auto& [k, v] = table_[i];
        map_fn(k, v, count);
        count++;
      }
    }
  }

  void Map(std::function<void(K, V&)> map_fn) {
    parlay::parallel_for(
        0, Capacity(),
        [&](size_t i) {
          if (Nonempty(table_[i])) {
            auto& [k, v] = table_[i];
            map_fn(k, v);
          }
        },
        1024);
  }

  void IterateUntil(std::function<bool(K, V)> iter_fn) const {
    for (size_t i = 0; i < Capacity(); i++) {
      if (Nonempty(table_[i])) {
        const auto& [k, v] = table_[i];
        if (iter_fn(k, v)) return;
      }
    }
  }

  void AdjustSizeForIncoming(size_t num_incoming) {
    auto [num_live, num_tombs] = SizeAndTombs();
    size_t total_elements = num_live + num_incoming;
    bool overfull =
        (total_elements + num_tombs) > (kOverfullThreshold * Capacity());
    bool underfull = (Capacity() > kMinCapacity) &&
                     (total_elements < (kUnderFullThreshold * Capacity()));
    ABSL_CHECK_GT(total_elements, 0);
    if (overfull || underfull || (Capacity() == 0)) {
      size_t new_size = SizeForElements(total_elements);
      auto old_table = std::move(table_);
      table_ = parlay::sequence<T>(new_size, kEmpty);

      auto update_f = [&](V* value) {};
      parlay::parallel_for(0, old_table.size(), [&](size_t i) {
        if (Nonempty(old_table[i])) {
          const auto& [k, v] = old_table[i];
          InsertOrUpdate(k, v, update_f);
        }
      });
    }
  }

 private:
  inline size_t HashToRange(size_t h) const { return h & (Capacity() - 1); }
  inline size_t FirstIndex(K& k) const {
    static_assert(sizeof(K) == 4);
    // TODO: what about using absl::Hash here? Is the performance
    // better or worse?
    return HashToRange(parlay::hash32_2(k));
  }
  inline size_t IncrementIndex(size_t h) const { return HashToRange(h + 1); }

  static K Key(const T& kv) { return std::get<0>(kv); }
  static V Value(const T& kv) { return std::get<1>(kv); }

  static V* MutableValue(T& kv) { return &std::get<1>(kv); }
  static K* MutableKey(T& kv) { return &std::get<0>(kv); }

  static bool Nonempty(const T& kv) { return std::get<0>(kv) < kTombstone; }

  void ClearTable() {
    parlay::parallel_for(0, Capacity(), [&](size_t i) { table_[i] = kEmpty; });
  }

  // Returns the size of the array used to store s elements.
  static size_t SizeForElements(size_t s) {
    return std::max(
        1 << parlay::log2_up(1 + (static_cast<size_t>(kSpaceMultiplier * s))),
        8);
  }

  // Total number of non-empty and kTombstone cells in table.
  std::pair<size_t, size_t> SizeAndTombs() const {
    size_t non_empty = 0;
    size_t tombs = 0;
    for (size_t i = 0; i < Capacity(); i++) {
      if (Nonempty(table_[i])) {
        non_empty++;
      } else if (Key(table_[i]) == kTombstone) {
        tombs++;
      }
    }
    return {non_empty, tombs};
  }

  // Returns a pointer to the table entry corresponding to the key k, or nullptr
  // if k does not exist in the table.
  T* Find(K k) {
    size_t h = FirstIndex(k);
    while (true) {
      if (Key(table_[h]) == k) {
        return &table_[h];
      } else if (Key(table_[h]) == kEmptyKey) {
        return nullptr;
      }
      h = IncrementIndex(h);
    }
    return nullptr;
  }

  static constexpr K kEmptyKey = std::numeric_limits<K>::max();
  static constexpr T kEmpty = {kEmptyKey, V()};
  // kTombstone is used when performing deletions to avoid "putting a hole" in
  // an existing key's probe sequence.
  static constexpr K kTombstone = std::numeric_limits<K>::max() - 1;
  static constexpr size_t kMinCapacity = 2;
  // The value of 1.1 for kSpaceMultiplier was tuned by hand on a 72-core
  // machine (this was about 15% faster than a multiplier of 1.5).
  static constexpr double kSpaceMultiplier = 1.1;
  // Reallocate the array if the number of used (and tombstone) slots is more
  // than a kOverfullThreshold fraction of the capacity.
  static constexpr double kOverfullThreshold = 1 / kSpaceMultiplier;
  // Reallocate the array if the number of used slots is less than a
  // kUnderFull fraction of the capacity.
  static constexpr double kUnderFullThreshold = 1 - (1 / kSpaceMultiplier);

  // Owning view of the hash-table's memory.
  parlay::sequence<T> table_;
};

}  // namespace internal

template <class Weight>
using HashBasedNeighborhood =
    internal::SimpleConcurrentTable<gbbs::uintE, Weight>;

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CLUSTERING_PARALLEL_CLUSTERED_GRAPH_INTERNAL_H_
