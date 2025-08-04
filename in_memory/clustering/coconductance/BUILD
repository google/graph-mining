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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "coconductance",
    srcs = ["coconductance.cc"],
    hdrs = ["coconductance.h"],
    deps = [
        ":coconductance_cc_proto",
        ":coconductance_internal",
        "//in_memory:status_macros",
        "//in_memory/clustering:clustering_utils",
        "//in_memory/clustering:compress_graph",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "coconductance_internal",
    srcs = ["coconductance_internal.cc"],
    hdrs = ["coconductance_internal.h"],
    deps = [
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
    ],
)

proto_library(
    name = "coconductance_proto",
    srcs = ["coconductance.proto"],
)

cc_proto_library(
    name = "coconductance_cc_proto",
    deps = [":coconductance_proto"],
)
