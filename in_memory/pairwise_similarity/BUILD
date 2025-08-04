# Copyright 2025 Google LLC
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

proto_library(
    name = "pairwise_similarity_proto",
    srcs = ["pairwise_similarity.proto"],
)

cc_proto_library(
    name = "pairwise_similarity_cc_proto",
    deps = [":pairwise_similarity_proto"],
)

cc_library(
    name = "pairwise_similarity",
    srcs = ["pairwise_similarity.cc"],
    hdrs = ["pairwise_similarity.h"],
    deps = [
        ":pairwise_similarity_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@com_github_gbbs//gbbs:bridge",
        "@parlaylib//parlay:parallel",
    ],
)
