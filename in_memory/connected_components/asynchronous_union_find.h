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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CONNECTED_COMPONENTS_ASYNCHRONOUS_UNION_FIND_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CONNECTED_COMPONENTS_ASYNCHRONOUS_UNION_FIND_H_

#include <type_traits>

#include "absl/types/span.h"
#include "benchmarks/Connectivity/SimpleUnionAsync/Connectivity.h"

namespace graph_mining::in_memory {

// The functions in this file provide an interface for dense connected
// components using asynchronous union-find. The algorithm currently used is the
// UF-Async algorithm from https://www.vldb.org/pvldb/vol14/p653-dhulipala.pdf,
// based on the recent paper by Jayanti-Tarjan in PODC'16. As required by
// go/atomic-danger, this algorithm has been machine-verified by sjayanti@ using
// Meta-configuration Tracking and by Jayanti, Jayanti, Yavuz, and Hernandez in
// their POPL '24 paper. For the proof in TLAPS, see:
// https://github.com/visveswara/machine-certified-linearizability/blob/master/UnionFindTracker.tla
// and
// https://github.com/uguryavuz/machine-certified-linearizability/blob/main/proofs/POPL24_JayantiTarjanUF.tla
//
// IntT is the integer type used to represent a parent-id. This type needs to be
// large enough to store the number of nodes in the data structure.
//
// ParentArray is the type of the underlying container used to store the ids of
// all parents. If ParentArray is a parlay::sequence<IntT>, then the
// implementation will use parallelism from parlay. Otherwise, it runs
// sequentially.
//
// The returned component ids (e.g., obtained using ComponentIds()) are between
// 0 and (num_nodes - 1).
//
// For a component C containing vertices {v_1, ... v_k}, the minimum id among
// these nodes, i.e., min(v_1, ..., v_k) is used as the component id of each
// node in C.
//
// As mentioned above, some of the functions in this class can be implemented in
// parallel when using ParentArray = parlay::sequence<IntT>. If this is the
// case, for maximum performance one should ensure that the Parlay scheduler
// has been initialized.
//
// Both Unite and Find are thread-safe and can be called concurrently with each
// other. Any other function cannot be called concurrently with any other
// function; e.g., ComponentIds() should not be used in parallel with calling
// any other function.
template <class IntT, class ParentArray = parlay::sequence<IntT>>
class AsynchronousUnionFind {
 public:
  AsynchronousUnionFind() = default;

  AsynchronousUnionFind(IntT num_nodes) {
    if constexpr (std::is_same_v<ParentArray, parlay::sequence<IntT>>) {
      parents_ = parlay::sequence<IntT>::from_function(
          num_nodes, [&](IntT i) { return i; });
    } else {
      parents_ = ParentArray(num_nodes);
      for (size_t i = 0; i < num_nodes; ++i) {
        parents_[i] = i;
      }
    }
  }

  AsynchronousUnionFind(ParentArray&& parents) : parents_(parents) {}

  // Given two node ids, node_u and node_v, this function joins the two trees
  // containing node_u and node_v. If both ids are already in the same tree,
  // this function is a noop with respect to the underlying connectivity
  // information.
  bool Unite(IntT node_u, IntT node_v) {
    return gbbs::simple_union_find::unite_impl_atomic(node_u, node_v,
                                                      parents_.data());
  }

  // Given node_u, a node id, returns the id of the root of the tree containing
  // node_u. This function is thread-safe.
  IntT Find(IntT node_u) {
    return gbbs::simple_union_find::find_compress_atomic(node_u,
                                                         parents_.data());
  }

  // Calls find() for every node to compress the parents sequence, and returns
  // a ParentArray (without a copy); this object is no longer usable after
  // calling ComponentSequence().
  ParentArray&& ComponentSequence() && {
    FindAllParents();
    return std::move(parents_);
  }

  // The same as ComponentSequence above, but returns a span and thus leaves the
  // object usable.
  absl::Span<const IntT> ComponentIds() {
    FindAllParents();
    return absl::Span<const IntT>(parents_.data(), parents_.size());
  }

  // Returns the number of nodes.
  IntT NumberOfNodes() const { return parents_.size(); }

 private:
  // Calls Find(i) on each vertex i in parallel. This function flattens the
  // union-find hierarchy so that every node points directly to the root of the
  // tree containing it.
  void FindAllParents() {
    if constexpr (std::is_same_v<ParentArray, parlay::sequence<IntT>>) {
      parlay::parallel_for(0, parents_.size(),
                           [&](size_t i) { parents_[i] = Find(i); });
    } else {
      for (size_t i = 0; i < parents_.size(); ++i) {
        parents_[i] = Find(i);
      }
    }
  }

  // The underlying parents array.
  ParentArray parents_;
};

// SequentialUnionFind provides a sequential implementation using std::vector as
// the underlying container.
//
// Both Unite and Find are still thread-safe. The warnings about calling other
// functions concurrently (see the comments on the main class) still apply.
template <class IntT>
using SequentialUnionFind = AsynchronousUnionFind<IntT, std::vector<IntT>>;

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_CONNECTED_COMPONENTS_ASYNCHRONOUS_UNION_FIND_H_
