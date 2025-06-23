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

#ifndef THIRD_PARTY_GRAPH_MINING_UTILS_PARSE_PROTO_PARSE_TEXT_PROTO_INTERNAL_H_
#define THIRD_PARTY_GRAPH_MINING_UTILS_PARSE_PROTO_PARSE_TEXT_PROTO_INTERNAL_H_

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"

#define PARSE_TEXT_PROTO(STR) ParseProtoHelper(STR)

class ParseProtoHelper {
 public:
  explicit ParseProtoHelper(absl::string_view string_view)
      : string_view_(string_view) {}

  template <class T>
  operator T() {
    static_assert(std::is_base_of<google::protobuf::Message, T>::value &&
                  !std::is_same<google::protobuf::Message, T>::value);
    T msg;
    ABSL_CHECK(google::protobuf::TextFormat::ParseFromString(string_view_, &msg));
    return msg;
  }

 private:
  absl::string_view string_view_;
};
#endif  // THIRD_PARTY_GRAPH_MINING_UTILS_PARSE_PROTO_PARSE_TEXT_PROTO_INTERNAL_H_
