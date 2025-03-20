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

cc_library(
    name = "dynamic_dendrogram",
    srcs = ["dynamic_dendrogram.cc"],
    hdrs = ["dynamic_dendrogram.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:parallel_dendrogram",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dynamic_clustered_graph",
    srcs = ["dynamic_clustered_graph.cc"],
    hdrs = ["dynamic_clustered_graph.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:parallel_clustered_graph",
        "//in_memory/clustering:parallel_clustered_graph_internal",
        "//in_memory/clustering:types",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "dynamic_hac_proto",
    srcs = ["dynamic_hac.proto"],
)

cc_proto_library(
    name = "dynamic_hac_cc_proto",
    deps = [":dynamic_hac_proto"],
)

cc_library(
    name = "hac",
    srcs = ["hac.cc"],
    hdrs = ["hac.h"],
    deps = [
        ":color_utils",
        ":dynamic_clustered_graph",
        ":dynamic_dendrogram",
        ":dynamic_hac_cc_proto",
        ":dynamic_hac_updater",
        ":hac_internal",
        "//in_memory:status_macros",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/clustering/hac/subgraph:approximate_subgraph_hac",
        "//utils:timer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "hac_internal",
    srcs = ["hac_internal.cc"],
    hdrs = ["hac_internal.h"],
    deps = [
        ":color_utils",
        ":dynamic_clustered_graph",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/clustering/hac/subgraph:approximate_subgraph_hac",
        "//utils/status:thread_safe_status",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:parallel",
    ],
)

cc_library(
    name = "color_utils",
    hdrs = ["color_utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//in_memory/clustering:types",
               "@farmhash_archive//:farmhash",
    ],
)

graph_mining_cc_test(
    name = "color_utils_test",
    timeout = "short",
    srcs = ["color_utils_test.cc"],
    deps = [
        ":color_utils",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "adjacency_lists",
    srcs = ["adjacency_lists.cc"],
    hdrs = ["adjacency_lists.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:parallel",
    ],
)

graph_mining_cc_test(
    name = "adjacency_lists_test",
    srcs = ["adjacency_lists_test.cc"],
    deps = [
        ":adjacency_lists",
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:types",
        "//in_memory/parallel:scheduler_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dynamic_hac_updater",
    srcs = ["dynamic_hac_updater.cc"],
    hdrs = ["dynamic_hac_updater.h"],
    deps = [
        ":dynamic_clustered_graph",
        ":dynamic_dendrogram",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/clustering/hac/subgraph:approximate_subgraph_hac",
        "//utils/status:thread_safe_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:parallel",
    ],
)

graph_mining_cc_test(
    name = "dynamic_hac_updater_test",
    timeout = "short",
    srcs = ["dynamic_hac_updater_test.cc"],
    deps = [
        ":adjacency_lists",
        ":dynamic_clustered_graph",
        ":dynamic_dendrogram",
        ":dynamic_hac_updater",
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/clustering/hac/subgraph:approximate_subgraph_hac",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "hac_internal_test",
    timeout = "short",
    srcs = ["hac_internal_test.cc"],
    deps = [
        ":adjacency_lists",
        ":color_utils",
        ":dynamic_clustered_graph",
        ":hac_internal",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:scheduler_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "dynamic_dendrogram_test",
    timeout = "short",
    srcs = ["dynamic_dendrogram_test.cc"],
    deps = [
        ":dynamic_dendrogram",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:parallel_dendrogram",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "dynamic_clustered_graph_test",
    timeout = "short",
    srcs = ["dynamic_clustered_graph_test.cc"],
    deps = [
        ":color_utils",
        ":dynamic_clustered_graph",
        "//in_memory:status_macros",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "hac_small_test",
    timeout = "long",
    srcs = ["hac_small_test.cc"],
    deps = [
        ":adjacency_lists",
        ":color_utils",
        ":dynamic_dendrogram",
        ":dynamic_hac_cc_proto",
        ":hac",
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/generation:add_edge_weights",
        "//in_memory/generation:erdos_renyi",
        "//in_memory/parallel:scheduler",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "hac_test",
    size = "enormous",
    timeout = "eternal",
    srcs = ["hac_test.cc"],
    deps = [
        ":adjacency_lists",
        ":color_utils",
        ":dynamic_dendrogram",
        ":dynamic_hac_cc_proto",
        ":hac",
        "//in_memory:status_macros",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:dendrogram_test_utils",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/generation:add_edge_weights",
        "//in_memory/generation:erdos_renyi",
        "//in_memory/parallel:scheduler_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)
