# Copyright 2024 Google LLC
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
load("//utils:build_defs.bzl", "graph_mining_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "approximate_subgraph_hac_node",
    srcs = ["approximate_subgraph_hac_node.cc"],
    hdrs = ["approximate_subgraph_hac_node.h"],
    deps = [
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//utils:math",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "approximate_subgraph_hac_graph",
    srcs = ["approximate_subgraph_hac_graph.cc"],
    hdrs = ["approximate_subgraph_hac_graph.h"],
    deps = [
        ":approximate_subgraph_hac_node",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//utils:math",
        "//utils/container:fixed_size_priority_queue",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "approximate_subgraph_hac",
    srcs = ["approximate_subgraph_hac.cc"],
    hdrs = ["approximate_subgraph_hac.h"],
    deps = [
        ":approximate_subgraph_hac_graph",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
    ],
)

graph_mining_cc_test(
    name = "approximate_subgraph_hac_graph_test",
    size = "small",
    srcs = ["approximate_subgraph_hac_graph_test.cc"],
    deps = [
        ":approximate_subgraph_hac_graph",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//utils:math",
        "@abseil-cpp//absl/algorithm:container",
        "@googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "approximate_subgraph_hac_test",
    size = "small",
    srcs = ["approximate_subgraph_hac_test.cc"],
    deps = [
        ":approximate_subgraph_hac",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//utils:math",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "approximate_subgraph_hac_node_test",
    size = "small",
    srcs = ["approximate_subgraph_hac_node_test.cc"],
    deps = [
        ":approximate_subgraph_hac_node",
        "//in_memory:status_macros",
        "//in_memory/clustering:in_memory_clusterer",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
    ],
)
