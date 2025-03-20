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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "parallel_line",
    srcs = ["parallel_line.cc"],
    hdrs = ["parallel_line.h"],
    deps = [
        ":affinity_hierarchy_embedder",
        ":linear_embedder",
        ":pairwise_improver",
        ":parline_cc_proto",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
    alwayslink = 1,
)

proto_library(
    name = "parline_proto",
    srcs = ["parline.proto"],
    deps = ["//in_memory/clustering/affinity:affinity_proto"],
)

cc_proto_library(
    name = "parline_cc_proto",
    deps = [":parline_proto"],
)

cc_library(
    name = "linear_embedder",
    hdrs = ["linear_embedder.h"],
    deps = [
        "//in_memory/clustering:gbbs_graph",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "affinity_hierarchy_embedder",
    srcs = ["affinity_hierarchy_embedder.cc"],
    hdrs = ["affinity_hierarchy_embedder.h"],
    deps = [
        ":linear_embedder",
        "//in_memory:status_macros",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering/affinity:affinity_cc_proto",
        "//in_memory/clustering/affinity:parallel_affinity_internal",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "cut_size",
    srcs = ["cut_size.cc"],
    hdrs = ["cut_size.h"],
    deps = [
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@parlaylib//parlay:monoid",
        "@parlaylib//parlay:parallel",
    ],
)

proto_library(
    name = "minla_proto",
    srcs = ["minla.proto"],
)

cc_proto_library(
    name = "minla_cc_proto",
    deps = [":minla_proto"],
)

cc_library(
    name = "minla_cost_metric",
    srcs = ["minla_cost_metric.cc"],
    hdrs = ["minla_cost_metric.h"],
    deps = [
        ":minla_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "minla",
    srcs = ["minla.cc"],
    hdrs = ["minla.h"],
    deps = [
        ":minla_cc_proto",
        ":minla_cost_metric",
        "//in_memory:status_macros",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/parallel:scheduler",
        "@com_google_absl//absl/log:absl_check",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "pairing_scheme",
    srcs = ["pairing_scheme.cc"],
    hdrs = ["pairing_scheme.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_library(
    name = "cluster_pair_improver",
    hdrs = ["cluster_pair_improver.h"],
    deps = [
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "fm_base",
    srcs = ["fm_base.cc"],
    hdrs = ["fm_base.h"],
    deps = [
        ":cluster_pair_improver",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:types",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "pairwise_improver",
    srcs = ["pairwise_improver.cc"],
    hdrs = ["pairwise_improver.h"],
    deps = [
        ":fm_base",
        ":pairing_scheme",
        ":parline_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
    ],
)
