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

#ifndef THIRD_PARTY_GRAPH_MINING_UTILS_MATH_H_
#define THIRD_PARTY_GRAPH_MINING_UTILS_MATH_H_

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "absl/log/absl_check.h"
#include "absl/numeric/bits.h"

namespace graph_mining {

// Checks if the difference between two given floating point numbers are within
// a margin or a relative margin (i.e. fraction of their max).
template <typename T>
bool WithinMarginOrFraction(const T x, const T y, const T margin,
                            const T fraction) {
  static_assert(std::is_floating_point_v<T>,
                "WithinMarginOrFraction only supports floating point types.");
  if (std::isinf(x) || std::isinf(y)) {
    return false;
  }
  T diff = std::abs(x - y);
  return diff <= margin ||
         diff <= fraction * std::max(std::abs(x), std::abs(y));
}

// Checks if two given floating point numbers are almost equal.
template <typename T>
static bool AlmostEquals(const T x, const T y) {
  if (x == y)
    return true;  // Covers +inf and -inf, and is a shortcut for finite values.
  static constexpr T margin = 32 * std::numeric_limits<T>::epsilon();
  static constexpr T fraction = 32 * std::numeric_limits<T>::epsilon();
  return WithinMarginOrFraction(x, y, margin, fraction);
}

// Returns floor(log2(x)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32_t n) { return absl::bit_width(n) - 1; }

// Returns 1 + 2 + 3 + ... + (n-1)
int64_t UpperTriangleSize(int n);

// Converts between linear and matrix indices for a strict upper triangular
// matrix (no diagonal entries). The inverse operation looks like this:
//   LinearIndex(i,j,n) => UTS(n) - UTS(n - i) + j - i - 1
//   where UTS is UpperTriangleSize
std::pair<int, int> UpperTriangleIndex(int64_t linear_index, int n);

// Converts between linear and matrix indices for a lower triangular
// matrix with diagonal entries. Similar to the `UpperTriangleIndex()` function,
// but includes the diagonal entries.
std::pair<int, int> LowerTriangleWithDiagIndex(int64_t linear_index, int n);

}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_UTILS_MATH_H_
