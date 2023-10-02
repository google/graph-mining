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
    name = "add_edge_weights",
    srcs = ["add_edge_weights.cc"],
    hdrs = ["add_edge_weights.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "barabasi_albert",
    srcs = ["barabasi_albert.cc"],
    hdrs = ["barabasi_albert.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
    ],
)

cc_library(
    name = "erdos_renyi",
    srcs = ["erdos_renyi.cc"],
    hdrs = ["erdos_renyi.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:graph",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status:statusor",
    ],
)
