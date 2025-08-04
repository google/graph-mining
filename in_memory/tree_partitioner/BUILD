# Tree partitioner library

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "min_size_tree_partitioning",
    srcs = ["min_size_tree_partitioning.cc"],
    hdrs = ["min_size_tree_partitioning.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)
