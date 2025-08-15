load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(
        default_visibility = ["//visibility:public"],
)

cc_library(
    name = "clique_aggregator",
    srcs = ["clique_aggregator.cc"],
    hdrs = ["clique_aggregator.h"],
    deps = [
        ":bitset",
        ":clique_aggregator_cc_proto",
        ":degeneracy_orientation",
        ":graphs",
        "//in_memory:status_macros",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
    alwayslink = True,
)

cc_library(
    name = "bitset",
    hdrs = ["bitset.h"],
    deps = ["@abseil-cpp//absl/log:absl_check"],
)

cc_library(
    name = "degeneracy_orientation",
    srcs = ["degeneracy_orientation.cc"],
    hdrs = ["degeneracy_orientation.h"],
    deps = [
        ":graphs",
        "//in_memory/clustering:config_cc_proto",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:in_memory_clusterer",
        "//in_memory/clustering:types",
        "//utils/container:fixed_size_priority_queue",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "graphs",
    hdrs = ["graphs.h"],
    deps = [
        ":bitset",
        "//in_memory/clustering:gbbs_graph",
        "//in_memory/clustering:types",
        "@abseil-cpp//absl/base:nullability",
        "@com_github_gbbs//gbbs:bridge",
        "@com_github_gbbs//gbbs:macros",
    ],
)

proto_library(
    name = "clique_aggregator_proto",
    srcs = ["clique_aggregator.proto"],
)

cc_proto_library(
    name = "clique_aggregator_cc_proto",
    deps = [":clique_aggregator_proto"],
)
