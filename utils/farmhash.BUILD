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

licenses(["notice"])

exports_files(["COPYING"])

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

cc_library(
    name = "farmhash",
    srcs = ["src/farmhash.cc"],
    hdrs = ["src/farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
    }),
    includes = ["src/."],
    visibility = ["//visibility:public"],
)
