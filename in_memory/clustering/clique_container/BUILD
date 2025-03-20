load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "internal",
    srcs = ["internal.cc"],
    hdrs = ["internal.h"],
    deps = [
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:types",
        "//utils/container:fixed_size_priority_queue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "clique_container",
    srcs = ["clique_container.cc"],
    hdrs = ["clique_container.h"],
    deps = [
        ":clique_container_cc_proto",
        ":internal",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status:statusor",
    ],
)

proto_library(
    name = "clique_container_proto",
    srcs = ["clique_container.proto"],
)

cc_proto_library(
    name = "clique_container_cc_proto",
    deps = [":clique_container_proto"],
)
