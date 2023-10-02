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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_STATUS_MACROS_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_STATUS_MACROS_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

// Defines the following utilities:
//
// ===============
// IsOkAndHolds(m)
// ===============
//
// This matcher matches a StatusOr<T> value whose status is OK and whose inner
// value matches matcher m. Example:
//
//   using ::graph_mining::in_memory::IsOkAndHolds;
//   ...
//   StatusOr<int> status_or_int(0);
//   EXPECT_THAT(status_or_message, IsOkAndHolds(0)));
//
// ===============================
// StatusIs(status_code_matcher,
//          error_message_matcher)
// ===============================
//
// This matcher matches a Status or StatusOr<T> if the following are true:
//
//   - the status's code() matches status_code_matcher, and
//   - the status's error_message() matches error_message_matcher.
//
// Example:
//
//   using ::graph_mining::in_memory::StatusIs;
//   ...
//
//   EXPECT_THAT(ReturnsInternalError(42),
//               StatusIs(absl::statuscode::Internal, _));
//
// ===============================
// StatusIs(status_code_matcher,
//          error_message_matcher)
// ===============================
//
// This matcher matches a Status or StatusOr<T> if the following are true:
//
//   - the status's code() matches status_code_matcher, and
//   - the status's error_message() matches error_message_matcher.
//
// Example:
//
//  // using ::testing::MatchesRegex;
//
//   // The status code must be INTERNAL; the error message can be anything.
//   EXPECT_THAT(ReturnsError(),
//               StatusIs(absl::statuscode::kInternal, _));
//
//   // The status code must be INTERNAL; the error message must match the
//   regex. EXPECT_THAT(ReturnsError(),
//               StatusIs(absl::statuscode::kInternal, _));
//               StatusIs(, MatchesRegex("server.*time-out")));
//
//
// =============================
// StatusIs(status_code_matcher)
// =============================
//
// This is a shorthand for
//   StatusIs(status_code_matcher, ::testing::_)
//
// In other words, it's like the two-argument StatusIs(), except that it ignores
// error messages.
//
// ======
// IsOk()
// ======
//
// Matches a Status or StatusOr<T> whose status value is OK.
// Equivalent to 'StatusIs(absl::StatusCode::kOk)'.
//
// Example:
//   ...
//   StatusOr<std::string> message("Hello, world");
//   EXPECT_THAT(message, IsOk());
//

namespace graph_mining {
namespace in_memory {
namespace internal_status {

inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}

template <typename T>
inline const absl::Status& GetStatus(const absl::StatusOr<T> status_or) {
  return status_or.status();
}

template <typename T>
inline const absl::Status& GetStatus(
    const absl::StatusOr<std::unique_ptr<T>>& status_or) {
  return status_or.status();
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().

// Monomorphic implementation of matcher IsOkAndHolds(m).  StatusOrType is a
// reference to StatusOr<T>.
template <typename StatusOrType>
class IsOkAndHoldsMatcherImpl
    : public ::testing::MatcherInterface<StatusOrType> {
 public:
  typedef
      typename std::remove_reference<StatusOrType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      StatusOrType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has status " << actual_value.status();
      return false;
    }

    ::testing::StringMatchResultListener inner_listener;
    const bool matches =
        inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
    const std::string inner_explanation = inner_listener.str();
    if (!inner_explanation.empty()) {
      *result_listener << "which contains value "
                       << ::testing::PrintToString(*actual_value) << ", "
                       << inner_explanation;
    }
    return matches;
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  StatusOrType can be either StatusOr<T> or a
  // reference to StatusOr<T>.
  template <typename StatusOrType>
  operator ::testing::Matcher<StatusOrType>() const {  // NOLINT
    return ::testing::Matcher<StatusOrType>(
        new IsOkAndHoldsMatcherImpl<const StatusOrType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of StatusIs().
class StatusIsMatcherCommonImpl {
 public:
  StatusIsMatcherCommonImpl(
      ::testing::Matcher<absl::StatusCode> code_matcher,
      ::testing::Matcher<const std::string&> message_matcher)
      : code_matcher_(std::move(code_matcher)),
        message_matcher_(std::move(message_matcher)) {}

  void DescribeTo(std::ostream* os) const;

  void DescribeNegationTo(std::ostream* os) const;

  bool MatchAndExplain(const absl::Status& status,
                       ::testing::MatchResultListener* result_listener) const;

 private:
  const ::testing::Matcher<absl::StatusCode> code_matcher_;
  const ::testing::Matcher<const std::string&> message_matcher_;
};

/////////////////////////////////////////////////////////////////////
// Monomorphic implementation of matcher StatusIs() for a given type
// T.  T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoStatusIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoStatusIsMatcherImpl(StatusIsMatcherCommonImpl common_impl)
      : common_impl_(std::move(common_impl)) {}

  void DescribeTo(std::ostream* os) const override {
    common_impl_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    common_impl_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    return common_impl_.MatchAndExplain(GetStatus(actual_value),
                                        result_listener);
  }

 private:
  StatusIsMatcherCommonImpl common_impl_;
};

/////////////////////////////////////////////////////////////
// Implements StatusIs() as a polymorphic matcher.
class StatusIsMatcher {
 public:
  template <typename StatusCodeMatcher, typename StatusMessageMatcher>
  StatusIsMatcher(StatusCodeMatcher&& code_matcher,
                  StatusMessageMatcher&& message_matcher)
      : common_impl_(::testing::MatcherCast<absl::StatusCode>(
                         std::forward<StatusCodeMatcher>(code_matcher)),
                     ::testing::MatcherCast<const std::string&>(
                         std::forward<StatusMessageMatcher>(message_matcher))) {
  }

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type.  T can be StatusOr<>, Status, or a reference to
  // either of them.
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(
        new MonoStatusIsMatcherImpl<const T&>(common_impl_));
  }

 private:
  const StatusIsMatcherCommonImpl common_impl_;
};

////////////////////////////////////////////////////////////////////
// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, StatusOr<>, or a reference to either of them.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override { *os << "is OK"; }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(T actual_value,
                       ::testing::MatchResultListener*) const override {
    return GetStatus(actual_value).ok();
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const {  // NOLINT
    return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
  }
};

}  // namespace internal_status

// Returns a matcher that matches a StatusOr<> whose status is OK and whose
// value matches the inner matcher.
template <typename InnerMatcher>
internal_status::IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>
IsOkAndHolds(InnerMatcher&& inner_matcher) {
  return internal_status::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcher>::type>(
      std::forward<InnerMatcher>(inner_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher, and whose error message matches message_matcher.
template <typename CodeMatcher, typename MessageMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher,
                                          MessageMatcher message_matcher) {
  return internal_status::StatusIsMatcher(std::move(code_matcher),
                                          std::move(message_matcher));
}

// Returns a matcher that matches a Status or StatusOr<> whose status code
// matches code_matcher.
template <typename CodeMatcher>
internal_status::StatusIsMatcher StatusIs(CodeMatcher code_matcher) {
  return StatusIs(std::move(code_matcher), ::testing::_);
}

// Returns a matcher that matches a Status or StatusOr<> which is OK.
inline internal_status::IsOkMatcher IsOk() {
  return internal_status::IsOkMatcher();
}

// A set of helpers to test the result of absl::Status
#ifndef TESTING_BASE_PUBLIC_GMOCK_UTILS_STATUS_MATCHERS_H_

#define EXPECT_OK(expr) EXPECT_TRUE((expr).ok())
#define ASSERT_OK(expr) ASSERT_TRUE((expr).ok())

#define CONCAT_IMPL(x, y) x##y
#define CONCAT_MACRO(x, y) CONCAT_IMPL(x, y)

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  ASSERT_OK_AND_ASSIGN_IMPL(CONCAT_MACRO(_status_or, __COUNTER__), lhs, rexpr)

#define ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)     \
  auto statusor = (rexpr);                                  \
  ASSERT_TRUE(statusor.status().ok()) << statusor.status(); \
  lhs = std::move(statusor.value())

#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                          \
  if (!statusor.ok()) {                             \
    return statusor.status();                       \
  }                                                 \
  lhs = std::move(statusor.value())

// Executes an expression that returns an absl::StatusOr, extracting its value
// into the variable defined by lhs (or returning on error).
//
// Example: Assigning to an existing value
//   ValueType value;
//   ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// WARNING: ASSIGN_OR_RETURN expands into multiple statements; it cannot be used
//  in a single statement (e.g. as the body of an if statement without {})!
#define ASSIGN_OR_RETURN(lhs, rexpr) \
  ASSIGN_OR_RETURN_IMPL(CONCAT_MACRO(_status_or, __COUNTER__), lhs, rexpr)

// Run a command that returns a util::Status.  If the called code returns an
// error status, return that status up out of this method too.
//
// Example:
//   RETURN_IF_ERROR(DoThings(4));
#define RETURN_IF_ERROR(expr)                                         \
  do {                                                                \
    const ::absl::Status _status = (expr);                            \
    if (!_status.ok()) {                                              \
      ABSL_LOG(ERROR) << "Return Error: " << #expr << " failed with " \
                      << _status;                                     \
      return _status;                                                 \
    }                                                                 \
  } while (0)

#define RETURN_VOID_IF_ERROR(expr)                                    \
  {                                                                   \
    const ::absl::Status _status = (expr);                            \
    if (!_status.ok()) {                                              \
      ABSL_LOG(ERROR) << "Return Error: " << #expr << " failed with " \
                      << _status;                                     \
      return;                                                         \
    }                                                                 \
  }

#endif

}  // namespace in_memory
}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_STATUS_MACROS_H_
