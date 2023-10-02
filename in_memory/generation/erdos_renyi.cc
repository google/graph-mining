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

#include "in_memory/generation/erdos_renyi.h"

#include <cstddef>
#include <memory>

#include "absl/log/absl_check.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "in_memory/clustering/graph.h"
#include "in_memory/status_macros.h"

namespace graph_mining::in_memory {

absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> UnweightedErdosRenyi(
    size_t n, double p) {
  ABSL_CHECK_LE(0, p);
  ABSL_CHECK_LE(p, 1);
  ABSL_CHECK_GE(n, 1);

  auto result = std::make_unique<SimpleUndirectedGraph>();
  result->SetNumNodes(n);

  absl::BitGen gen;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; j++) {
      if (absl::Bernoulli(gen, p)) {
        RETURN_IF_ERROR(result->AddEdge(i, j, 1));
      }
    }
  }
  return result;
}

}  // namespace graph_mining::in_memory
