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

# Parallel PageRank library.

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "pagerank_proto",
    srcs = ["pagerank.proto"],
)

cc_proto_library(
    name = "pagerank_cc_proto",
    deps = [":pagerank_proto"],
)

cc_library(
    name = "parallel_pagerank",
    srcs = ["parallel_pagerank.cc"],
    hdrs = ["parallel_pagerank.h"],
    deps = [
        ":pagerank_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/parallel:scheduler",
        "@com_github_gbbs//benchmarks/PageRank",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@parlaylib//parlay:sequence",
    ],
)
