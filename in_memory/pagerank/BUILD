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

# Parallel PageRank library.

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
        default_visibility = ["//visibility:public"],
)

proto_library(
    name = "pagerank_proto",
    srcs = ["pagerank.proto"],
)

cc_proto_library(
    name = "pagerank_cc_proto",
    deps = [":pagerank_proto"],
)

cc_library(
    name = "parallel_pagerank",
    srcs = ["parallel_pagerank.cc"],
    hdrs = ["parallel_pagerank.h"],
    deps = [
        ":pagerank_cc_proto",
        "//in_memory:status_macros",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:types",
        "//in_memory/parallel:scheduler",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_github_gbbs//benchmarks/PageRank",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs/helpers:parallel_for_with_status",
        "@com_github_gbbs//gbbs/helpers:progress_reporting",
        "@parlaylib//parlay:sequence",
    ],
)
