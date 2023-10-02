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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    srcs = ["dynamic_weight_threshold.proto"],
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

cc_library(
    name = "parallel_affinity_internal",
    srcs = ["parallel_affinity_internal.cc"],
    hdrs = ["parallel_affinity_internal.h"],
    deps = [
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/connected_components:asynchronous_union_find",
        "//in_memory/parallel:parallel_graph_utils",
        "//in_memory/parallel:parallel_sequence_ops",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status:statusor",
        "@parlaylib//parlay:sequence",
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
