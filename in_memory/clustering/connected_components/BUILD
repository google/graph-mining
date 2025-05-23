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

cc_library(
    name = "connected_components_graph",
    hdrs = ["connected_components_graph.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/connected_components:asynchronous_union_find",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "connected_components",
    srcs = ["connected_components.cc"],
    hdrs = ["connected_components.h"],
    deps = [
        ":connected_components_graph",
        "//in_memory:status_macros",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/parallel:scheduler",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@parlaylib//parlay:parallel",
    ],
    alwayslink = 1,
)
