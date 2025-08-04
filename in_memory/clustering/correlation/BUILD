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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
        default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "parallel_correlation",
    srcs = ["parallel_correlation.cc"],
    hdrs = ["parallel_correlation.h"],
    deps = [
        ":correlation_cc_proto",
        ":parallel_correlation_util",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/parallel:scheduler",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:edge_map_data",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_github_gbbs//gbbs:vertex_subset",
        "@com_github_gbbs//gbbs/helpers:progress_reporting",
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
        ":correlation_cc_proto",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

cc_library(
    name = "parallel_correlation_util",
    srcs = ["parallel_correlation_util.cc"],
    hdrs = ["parallel_correlation_util.h"],
    deps = [
        ":correlation_cc_proto",
        ":correlation_util",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
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
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_gbbs//gbbs:graph",
        "@com_github_gbbs//gbbs:macros",
        "@com_github_gbbs//gbbs:vertex",
        "@com_github_gbbs//gbbs/helpers:progress_reporting",
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
        ":correlation_cc_proto",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
    ],
)
