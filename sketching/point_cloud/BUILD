# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
        default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "fixed_dimensional_encoding",
    srcs = ["fixed_dimensional_encoding.cc"],
    hdrs = ["fixed_dimensional_encoding.h"],
    deps = [
        ":fixed_dimensional_encoding_config_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen",
    ],
)

proto_library(
    name = "fixed_dimensional_encoding_config_proto",
    srcs = ["fixed_dimensional_encoding_config.proto"],
)

cc_proto_library(
    name = "fixed_dimensional_encoding_config_cc_proto",
    deps = [":fixed_dimensional_encoding_config_proto"],
)
