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

// This file contains the definitions of parallel sequence primitives
// which exploit multi-core parallelism.
//
// WARNING: The interfaces in this file are experimental and are liable to
// change.

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_

#include "absl/types/span.h"
#include "parlay/internal/counting_sort.h"
#include "parlay/internal/integer_sort.h"
#include "parlay/primitives.h"
#include "gbbs/bridge.h"

namespace graph_mining::in_memory {

// ======== Parallel reduction, scan, filtering, and packing methods ======== //

// Given an array of A's, In, and an associative f: A -> A -> A, computes the
// "sum" of all elements in In with respect to f.
template <class A>
A Reduce(absl::Span<const A> In, const std::function<A(A, A)>& f, A zero) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  auto monoid = parlay::make_monoid(f, zero);
  return parlay::reduce(seq_in, monoid);
}

// Given an array of A's, In, computes the sum (using the operator "+") of all
// elements in In.
template <class A>
A ReduceAdd(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  return parlay::reduce(seq_in);
}

// Given an array of A's, In, computes the sum (using "std::max") of all
// elements in In.
template <class A>
A ReduceMax(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  return parlay::reduce_max(seq_in);
}

// Given an array of A's, In, computes the sum (using "std::min") of all
// elements in In.
template <class A>
A ReduceMin(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  return parlay::reduce_min(seq_in);
}

// Given an array of A's, In, an
// associative function, f : A -> A -> A, and an identity, zero, computes the
// prefix sum of A with respect to f and zero. That is, the output is [zero,
// f(zero, In[0]), f(f(zero, In[0]),n[1]), ..]. The output is a parlay::sequence
// of the scan'd vector, and the overall sum.
template <class A>
std::pair<parlay::sequence<A>, A> Scan(absl::Span<const A> In,
                                       const std::function<A(A, A)>& f,
                                       A zero) {
  auto in_slice = parlay::make_slice(In.cbegin(), In.cend());
  auto monoid = parlay::make_monoid(f, zero);
  return parlay::scan(in_slice, monoid);
}

// Computes the prefix-sum of In returns it using + as f, and 0 as
// zero (see the description for scan above for more details).
template <class A>
std::pair<parlay::sequence<A>, A> ScanAdd(absl::Span<const A> In) {
  auto in_slice = parlay::make_slice(In.cbegin(), In.cend());
  return parlay::scan(in_slice);
}

// Given an array of A's, In, and a predicate function, Flags, indicating the
// indices corresponding to elements that should be in the output, Pack computes
// an array of A's containing only elements, In[i] in In s.t. Flags[i] == true.
// Flags must be a function that is callable on indices in the range [0, n).
template <class A>
parlay::sequence<A> Pack(absl::Span<const A> In,
                         const std::function<bool(size_t)>& Flags) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  auto bool_seq = parlay::delayed_seq<bool>(In.size(), Flags);
  return parlay::pack(seq_in, bool_seq);
}

// Returns an array of Idx_Type (e.g., size_t's) containing only those indices i
// s.t. Flags[i] == true. Flags must be a function that is callable on each
// element of the range [0, n).
template <class Idx_Type>
parlay::sequence<Idx_Type> PackIndex(const std::function<bool(size_t)>& Flags,
                                     size_t num_elements) {
  auto flags_sequence = parlay::delayed_seq<bool>(num_elements, Flags);
  return parlay::pack_index<Idx_Type>(flags_sequence);
}

// Given an array of A's, In, and a predicate function, p : A -> bool, computes
// an array of A's containing only elements, e in In s.t. p(e) == true.
template <class A>
parlay::sequence<A> Filter(absl::Span<const A> In,
                           const std::function<bool(A)>& p) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  return parlay::filter(seq_in, p);
}

template <class A>
std::vector<A> FilterOut(absl::Span<const A> In,
                         const std::function<bool(A)>& p) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = parlay::delayed_seq<A>(In.size(), get_in);
  auto filtered = parlay::filter(seq_in, p);
  return std::vector<A>(filtered.begin(), filtered.begin() + filtered.size());
}

// Given a number of keys and a key_eq_func that takes as input two indices
// and / outputs whether the corresponding keys are equal, create a vector
// consisting / of all i such that either {i == 0 or 1 <= i < num_keys and
// key(i) != key(i-1)}, sorted in / increasing order. An additional element is
// appended to the end of the vector, / with value num_keys.
template <class A>
std::vector<A> GetBoundaryIndices(
    std::size_t num_keys,
    const std::function<bool(std::size_t, std::size_t)>& key_eq_func) {
  auto is_start = [&](size_t i) { return (i == 0) || !key_eq_func(i, i - 1); };
  auto filtered = parlay::filter(parlay::iota<A>(num_keys), is_start);
  auto ret =
      std::vector<A>(filtered.begin(), filtered.end());
  ret.push_back(num_keys);
  return ret;
}

// Given ids (class A) and a function indicating which indices (class B) to
// process, re-arrange the indices such that they are grouped by id, and return
// a vector of these groups.
template <class A, class B>
std::vector<std::vector<B>> OutputIndicesById(
    const absl::Span<const A>& index_ids,
    const std::function<B(B)>& get_indices_func, B num_indices) {
  auto idxs = parlay::sequence<B>::from_function(
      num_indices, [&](std::size_t i) -> B { return get_indices_func(i); });
  auto comp = [&](const B& l, const B& r) {
    return index_ids[l] < index_ids[r];
  };
  parlay::stable_sort_inplace(parlay::make_slice(idxs), comp);

  // Compute indices which start each block
  auto is_start = [&](std::size_t i) {
    return i == 0 || (index_ids[idxs[i - 1]] != index_ids[idxs[i]]);
  };
  auto starts =
      parlay::filter(parlay::iota<std::size_t>(num_indices), is_start);

  std::vector<std::vector<B>> finished_indices(starts.size());
  parlay::parallel_for(0, starts.size(), [&](auto i) {
    auto start = starts[i];
    auto end = (i == starts.size() - 1) ? num_indices : starts[i + 1];
    auto size = end - start;
    auto ret = std::vector<B>(size);
    for (size_t j = 0; j < size; j++) {
      ret[j] = idxs[start + j];
    }
    finished_indices[i] = std::move(ret);
  });
  return finished_indices;
}

// ===================   Parallel sorting methods ========================= //

// Takes a span of A's, In, a function get_key which specifies the integer key
// for each element, and val_bits which specifies how many bits of the keys to
// use. Sorts In using a top-down recursive radix sort with respect to the keys
// provided by get_key. The result is returned as a new sorted sequence.
template <typename A>
parlay::sequence<A> ParallelIntegerSort(
    absl::Span<A> In, const std::function<size_t(size_t)>& get_key) {
  auto seq_in = parlay::make_slice(In.data(), In.data() + In.size());
  return parlay::internal::integer_sort(seq_in, get_key);
}

// Takes a span of A's, In, and a comparison function f. Sorts In with respect
// to f using a sample sort. The result is returned as a new sorted sequence.
template <class A>
parlay::sequence<A> ParallelSampleSort(absl::Span<A> In,
                                       const std::function<bool(A, A)>& f) {
  auto seq_in = parlay::make_slice(In.data(), In.data() + In.size());
  return parlay::sample_sort(seq_in, f, true);
}

// Takes a span of A's, In, a function keys which takes the index of an element
// and returns the integer key (the bucket) for the element, and the total
// number of buckets (the size of the range of keys). In is sorted with respect
// to the buckets given by keys.
template <typename A>
parlay::sequence<A> ParallelCountSort(
    absl::Span<A> In, const std::function<size_t(size_t)>& get_key,
    size_t num_buckets) {
  auto seq_in = parlay::make_slice(In.data(), In.data() + In.size());
  auto key_seq = parlay::delayed_seq<size_t>(In.size(), get_key);
  auto [out, counts] =
      parlay::internal::count_sort(seq_in, key_seq, num_buckets);
  return out;
}

}  // namespace graph_mining::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_
