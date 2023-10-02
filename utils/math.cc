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

#include "utils/math.h"

#include <cmath>
#include <utility>

namespace graph_mining {
int64_t UpperTriangleSize(int n) { return int64_t{n} * (n - 1) / 2; }

// See https://stackoverflow.com/a/27088560
std::pair<int, int> UpperTriangleIndex(int64_t linear_index, int n) {
  const int i =
      n - 2 -
      std::floor(std::sqrt(-8 * linear_index + 4 * int64_t{n} * (n - 1) - 7) /
                     2.0 -
                 0.5);
  const int j =
      linear_index + i + 1 - UpperTriangleSize(n) + UpperTriangleSize(n - i);
  return {i, j};
}
}  // namespace graph_mining
