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

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        ":in_memory_clusterer",
        "//in_memory:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "in_memory_clusterer",
    srcs = ["in_memory_clusterer.cc"],
    hdrs = ["in_memory_clusterer.h"],
    deps = [
        ":config_cc_proto",
        ":dendrogram",
        ":types",
        "//in_memory:status_macros",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gbbs_graph",
    srcs = ["gbbs_graph.cc"],
    hdrs = ["gbbs_graph.h"],
    deps = [
        ":in_memory_clusterer",
        ":types",
        "//in_memory/parallel:scheduler",
        "//utils/status:thread_safe_status",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@parlaylib//parlay:parallel",
    ],
)

cc_library(
    name = "compress_graph",
    srcs = ["compress_graph.cc"],
    hdrs = ["compress_graph.h"],
    deps = [
        ":graph",
        ":in_memory_clusterer",
        "//in_memory:status_macros",
    ],
)

cc_library(
    name = "dendrogram",
    srcs = ["dendrogram.cc"],
    hdrs = ["dendrogram.h"],
    deps = [
        ":clustering_utils",
        ":types",
        "//in_memory:status_macros",
        "//in_memory/connected_components:asynchronous_union_find",
        "//utils:math",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "dendrogram_test_utils",
    srcs = ["dendrogram_test_utils.cc"],
    hdrs = ["dendrogram_test_utils.h"],
    deps = [
        ":compress_graph",
        ":dendrogram",
        ":graph",
        "//in_memory:status_macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "clustering_utils",
    srcs = ["clustering_utils.cc"],
    hdrs = ["clustering_utils.h"],
    deps = [
        ":types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "dendrogram_flatten",
    srcs = ["dendrogram_flatten.cc"],
    hdrs = ["dendrogram_flatten.h"],
    deps = [
        ":clustering_utils",
        ":dendrogram",
        ":types",
        "//in_memory:status_macros",
        "//in_memory/connected_components:asynchronous_union_find",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "parallel_clustered_graph",
    hdrs = ["parallel_clustered_graph.h"],
    deps = [
        ":parallel_clustered_graph_internal",
        ":parallel_dendrogram",
        ":types",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "parallel_clustered_graph_internal",
    hdrs = ["parallel_clustered_graph_internal.h"],
    deps = [
        "@com_github_gbbs//gbbs",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "undirected_converter_graph_proto",
    srcs = ["undirected_converter_graph.proto"],
)

cc_proto_library(
    name = "undirected_converter_graph_cc_proto",
    deps = [":undirected_converter_graph_proto"],
)

cc_library(
    name = "undirected_converter_graph",
    srcs = ["undirected_converter_graph.cc"],
    hdrs = ["undirected_converter_graph.h"],
    deps = [
        ":gbbs_graph",
        ":in_memory_clusterer",
        ":undirected_converter_graph_cc_proto",
        "//in_memory:status_macros",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:streaming_writer",
        "//utils/status:thread_safe_status",
        "@com_github_gbbs//gbbs",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
        "@parlaylib//parlay:slice",
    ],
)

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
    deps = [
        "//in_memory/clustering/affinity:affinity_proto",
        "//in_memory/clustering/coconductance:coconductance_proto",
        "//in_memory/clustering/correlation:correlation_proto",
        "//in_memory/clustering/correlation:modularity_proto",
        "//in_memory/clustering/parline:parline_proto",
    ],
)

cc_proto_library(
    name = "config_cc_proto",
    deps = [":config_proto"],
)

java_proto_library(
    name = "config_java_proto",
    deps = [":config_proto"],
)

cc_library(
    name = "graph_utils",
    srcs = ["graph_utils.cc"],
    hdrs = ["graph_utils.h"],
    deps = [
        ":graph",
        ":in_memory_clusterer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":graph",
        ":graph_utils",
        ":in_memory_clusterer",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "gbbs_graph_test_utils",
    testonly = 1,
    srcs = ["gbbs_graph_test_utils.cc"],
    hdrs = ["gbbs_graph_test_utils.h"],
    deps = [
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tiebreaking",
    hdrs = ["tiebreaking.h"],
    deps = [
        "@com_google_absl//absl/strings",
               "@farmhash_archive//:farmhash",
    ],
)

cc_library(
    name = "parallel_dendrogram",
    hdrs = ["parallel_dendrogram.h"],
    deps = [
        ":types",
        "//in_memory/connected_components:asynchronous_union_find",
        "//utils:math",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
    ],
)
