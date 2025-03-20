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

# Parallel triangle counting library.

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "parallel_triangle_counting",
    srcs = ["parallel_triangle_counting.cc"],
    hdrs = ["parallel_triangle_counting.h"],
    deps = [
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//benchmarks/TriangleCounting/ShunTangwongsan15:Triangle",
        "@com_github_gbbs//gbbs:macros",
        "@com_google_absl//absl/status",
    ],
)
