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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_ERDOS_RENYI_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_ERDOS_RENYI_H_

#include "in_memory/clustering/graph.h"

namespace graph_mining::in_memory {

// Construct a G(n,p) graph for n >=1 and 0 <= p <= 1 (see:
// https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) with given
// values of n and p. Note that this method spends O(n^2) time to consider each
// possible edge.
// TODO: For larger graphs, the (sparse) G(n,M) model would be more
// appropriate, but is not yet implemented. If higher performance is necessary
// for implementing G(n,p), this piece of code is useful:
absl::StatusOr<std::unique_ptr<SimpleUndirectedGraph>> UnweightedErdosRenyi(
    size_t n, double p);

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_GENERATION_ERDOS_RENYI_H_
