// Copyright 2010-2023 Google LLC
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "in_memory/clustering/in_memory_clusterer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "in_memory/clustering/config.pb.h"
#include "in_memory/status_macros.h"

namespace graph_mining {
namespace in_memory {

absl::Status InMemoryClusterer::Graph::PrepareImport(const int64_t num_nodes) {
  return absl::OkStatus();
}

absl::Status InMemoryClusterer::Graph::FinishImport() {
  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Dendrogram>
InMemoryClusterer::HierarchicalCluster(
    const research_graph::in_memory::ClustererConfig& config) const {
  return absl::UnimplementedError("HierarchicalCluster not implemented.");
}

absl::StatusOr<std::vector<InMemoryClusterer::Clustering>>
InMemoryClusterer::HierarchicalFlatCluster(
    const research_graph::in_memory::ClustererConfig& config) const {
  Clustering clusters;
  ASSIGN_OR_RETURN(clusters, Cluster(config));
  return std::vector<Clustering>{clusters};
}

std::string InMemoryClusterer::StringId(NodeId id) const {
  if (node_id_map_ == nullptr) {
    return absl::StrCat(id);
  } else if (id >= 0 && id < node_id_map_->size()) {
    return (*node_id_map_)[id];
  } else {
    return absl::StrCat("missing-id-", id);
  }
}

}  // namespace in_memory
}  // namespace graph_mining
