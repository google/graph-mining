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

cc_library(
    name = "connected_components_graph",
    hdrs = ["connected_components_graph.h"],
    deps = [
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/connected_components:asynchronous_union_find",
    ],
)

cc_library(
    name = "connected_components",
    srcs = ["connected_components.cc"],
    hdrs = ["connected_components.h"],
    deps = [
        ":connected_components_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/parallel:parallel_sequence_ops",
        "//in_memory/parallel:scheduler",
    ],
    alwayslink = 1,
)
