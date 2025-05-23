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

package(
        default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "parhac",
    srcs = ["parhac.cc"],
    hdrs = ["parhac.h"],
    deps = [
        ":parhac_internal",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:parallel_clustered_graph",
        "//in_memory/clustering:parallel_clustered_graph_internal",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:scheduler",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "terahac",
    srcs = ["terahac.cc"],
    hdrs = ["terahac.h"],
    deps = [
        ":terahac_internal",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:dendrogram",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:parallel_clustered_graph",
        "//in_memory/clustering:parallel_clustered_graph_internal",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parhac_internal",
    hdrs = ["parhac_internal.h"],
    deps = [
        "//utils:math",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "terahac_internal",
    hdrs = ["terahac_internal.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:parallel_clustered_graph_internal",
        "//in_memory/clustering/hac/subgraph:approximate_subgraph_hac",
        "//in_memory/connected_components:asynchronous_union_find",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@parlaylib//parlay:sequence",
    ],
)
