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
    name = "parallel_correlation",
    srcs = ["parallel_correlation.cc"],
    hdrs = ["parallel_correlation.h"],
    deps = [
        ":parallel_correlation_util",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:edge_map_data",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_github_gbbs//gbbs:vertex_subset",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
    alwayslink = 1,
)

cc_library(
    name = "correlation_util",
    srcs = ["correlation_util.cc"],
    hdrs = ["correlation_util.h"],
    deps = [
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "parallel_correlation_util",
    srcs = ["parallel_correlation_util.cc"],
    hdrs = ["parallel_correlation_util.h"],
    deps = [
        ":correlation_util",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:delayed_sequence",
        "@parlaylib//parlay:monoid",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:primitives",
        "@parlaylib//parlay:sequence",
        "@parlaylib//parlay:slice",
    ],
)

proto_library(
    name = "correlation_proto",
    srcs = ["correlation.proto"],
)

py_proto_library(
    name = "correlation_py_pb2",
    deps = [":correlation_proto"],
)

cc_proto_library(
    name = "correlation_cc_proto",
    deps = [":correlation_proto"],
)

cc_library(
    name = "parallel_modularity",
    srcs = ["parallel_modularity.cc"],
    hdrs = ["parallel_modularity.h"],
    deps = [
        ":parallel_correlation",
        ":parallel_correlation_util",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@parlaylib//parlay:parallel",
    ],
    alwayslink = 1,
)

proto_library(
    name = "modularity_proto",
    srcs = ["modularity.proto"],
    deps = [":correlation_proto"],
)

cc_proto_library(
    name = "modularity_cc_proto",
    deps = [":modularity_proto"],
)

cc_library(
    name = "quick_cluster",
    srcs = ["quick_cluster.cc"],
    hdrs = ["quick_cluster.h"],
    deps = [
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
    ],
)
