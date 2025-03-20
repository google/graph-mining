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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
