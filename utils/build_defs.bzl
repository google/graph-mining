"""Custom Bazel build rules."""

def graph_mining_cc_test(name, **kwargs):
    """Builds a C++ test. Analogous to the default `cc_test` Bazel rule. graph_mining_cc_test will not be stiped by copybara."""
    native.cc_test(name = name, **kwargs)
