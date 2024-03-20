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

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "parallel_sequence_ops",
    hdrs = ["parallel_sequence_ops.h"],
    deps = [
        "@com_github_gbbs//gbbs:bridge",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay/internal:counting_sort",
        "@parlaylib//parlay/internal:integer_sort",
    ],
)

cc_library(
    name = "scheduler",
    srcs = ["scheduler.cc"],
    hdrs = ["scheduler.h"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@parlaylib//parlay:scheduler",
    ],
)

cc_library(
    name = "parallel_graph_utils",
    srcs = ["parallel_graph_utils.cc"],
    hdrs = ["parallel_graph_utils.h"],
    deps = [
        ":parallel_sequence_ops",
        "@com_github_gbbs//gbbs",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:graph_io",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/types:span",
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
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "streaming_writer",
    hdrs = ["streaming_writer.h"],
    deps = [
        ":per_worker",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "per_worker",
    hdrs = ["per_worker.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@parlaylib//parlay:scheduler",
    ],
)
