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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

load("//utils:build_defs.bzl", "graph_mining_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

graph_mining_cc_test(
    name = "weight_threshold_test",
    srcs = ["weight_threshold_test.cc"],
    deps = [
        ":affinity_cc_proto",
        ":weight_threshold",
        "//in_memory:status_macros",
        "//utils/parse_proto:parse_text_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

graph_mining_cc_test(
    name = "affinity_test",
    srcs = ["affinity_test.cc"],
    deps = [
        ":affinity",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//utils/parse_proto:parse_text_proto",
               "@farmhash_archive//:farmhash",
    ],
)

graph_mining_cc_test(
    name = "affinity_internal_test",
    srcs = ["affinity_internal_test.cc"],
    deps = [
        ":affinity_internal",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory:status_macros",
        "//utils/parse_proto:parse_text_proto",
               "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "weight_threshold",
    srcs = ["weight_threshold.cc"],
    hdrs = ["weight_threshold.h"],
    deps = [
        ":affinity_cc_proto",
        ":dynamic_weight_threshold",
        "//in_memory/clustering:config_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "affinity",
    srcs = ["affinity.cc"],
    hdrs = ["affinity.h"],
    deps = [
        ":affinity_cc_proto",
        ":affinity_internal",
        ":weight_threshold",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "affinity_internal",
    srcs = ["affinity_internal.cc"],
    hdrs = ["affinity_internal.h"],
    deps = [
        ":affinity_cc_proto",
        "//in_memory:status_macros",
        "//in_memory/clustering:compress_graph",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:tiebreaking",
        "//in_memory/connected_components:asynchronous_union_find",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
    ],
)

proto_library(
    name = "affinity_proto",
    srcs = ["affinity.proto"],
    deps = [
        ":dynamic_weight_threshold_proto",
    ],
)

proto_library(
    name = "dynamic_weight_threshold_proto",
    srcs = ["dynamic_weight_threshold.proto"],
)

cc_proto_library(
    name = "dynamic_weight_threshold_cc_proto",
    deps = [":dynamic_weight_threshold_proto"],
)

py_proto_library(
    name = "dynamic_weight_threshold_py_pb2",
    deps = [":dynamic_weight_threshold_proto"],
)

cc_proto_library(
    name = "affinity_cc_proto",
    deps = [":affinity_proto"],
)

cc_library(
    name = "parallel_affinity",
    srcs = ["parallel_affinity.cc"],
    hdrs = ["parallel_affinity.h"],
    deps = [
        ":parallel_affinity_internal",
        ":weight_threshold",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:scheduler",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

graph_mining_cc_test(
    name = "parallel_affinity_test",
    size = "small",
    timeout = "moderate",
    srcs = ["parallel_affinity_test.cc"],
    deps = [
        ":parallel_affinity",
        "//in_memory:status_macros",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//utils/parse_proto:parse_text_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parallel_affinity_internal",
    srcs = ["parallel_affinity_internal.cc"],
    hdrs = ["parallel_affinity_internal.h"],
    deps = [
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/connected_components:asynchronous_union_find",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/tree_partitioner:min_size_tree_partitioning",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:monoid",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
        "@parlaylib//parlay:slice",
    ],
)

cc_library(
    name = "dynamic_weight_threshold",
    srcs = ["dynamic_weight_threshold.cc"],
    hdrs = ["dynamic_weight_threshold.h"],
    deps = [
        ":dynamic_weight_threshold_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

graph_mining_cc_test(
    name = "parallel_affinity_internal_test",
    size = "small",
    srcs = ["parallel_affinity_internal_test.cc"],
    deps = [
        ":affinity_cc_proto",
        ":parallel_affinity_internal",
        "//in_memory:status_macros",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:gbbs_graph_test_utils",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/connected_components:asynchronous_union_find",
        "//in_memory/parallel:scheduler",
        "//utils/parse_proto:parse_text_proto",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@parlaylib//parlay:sequence",
        "@parlaylib//parlay/internal:group_by",
    ],
)

graph_mining_cc_test(
    name = "dynamic_weight_threshold_test",
    srcs = ["dynamic_weight_threshold_test.cc"],
    deps = [
        ":dynamic_weight_threshold",
        ":dynamic_weight_threshold_cc_proto",
        "//in_memory:status_macros",
        "//utils/parse_proto:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
