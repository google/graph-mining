# Tree partitioner library

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "min_size_tree_partitioning",
    srcs = ["min_size_tree_partitioning.cc"],
    hdrs = ["min_size_tree_partitioning.h"],
    deps = [
        "//in_memory:status_macros",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
