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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "parallel_sequence_ops",
    hdrs = ["parallel_sequence_ops.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
        "@com_github_gbbs//gbbs:bridge",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
        "@parlaylib//parlay/internal:counting_sort",
        "@parlaylib//parlay/internal:integer_sort",
    ],
)

cc_library(
    name = "scheduler",
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@parlaylib//parlay:scheduler",
    ],
)

cc_library(
    name = "parallel_graph_utils",
    srcs = ["parallel_graph_utils.cc"],
    hdrs = ["parallel_graph_utils.h"],
    deps = [
        ":parallel_sequence_ops",
        "@abseil-cpp//absl/types:span",
        "@com_github_gbbs//gbbs",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:graph_io",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "scheduler_test_util",
    testonly = 1,
    hdrs = ["scheduler_test_util.h"],
    deps = [
        ":scheduler",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "streaming_writer",
    hdrs = ["streaming_writer.h"],
    deps = [
        ":per_worker",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/synchronization",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "per_worker",
    hdrs = ["per_worker.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@parlaylib//parlay:scheduler",
    ],
)
