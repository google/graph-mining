// Copyright 2010-2023 Google LLC
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

#include "in_memory/triangle_counting/parallel_triangle_counting.h"

#include "absl/status/status.h"
#include "benchmarks/TriangleCounting/ShunTangwongsan15/Triangle.h"
#include "gbbs/macros.h"
#include "in_memory/parallel/scheduler.h"

using ::gbbs::uintE;

namespace third_party::graph_mining {

absl::StatusOr<uint64_t> ParallelTriangleCounting::Count() const {
  if (graph_.Graph() == nullptr) {
    return absl::FailedPreconditionError(
        "graph_ must be initialized before triangle counting.");
  }

  

  // unused_per_triangle_function is unused. This is to fit the interface of
  // gbbs::Triangle_degree_ordering.
  auto unused_per_triangle_function = [&](uintE u, uintE v, uintE w) {};

  // Triangle_degree_ordering assumes that the neighbors are sorted by node ids.
  // This is an implicit assumption in the implementation of
  // gbbs::intersect::intersect_f_par
  return ::gbbs::Triangle_degree_ordering(*(graph_.Graph()),
                                          unused_per_triangle_function);
}

}  // namespace third_party::graph_mining
