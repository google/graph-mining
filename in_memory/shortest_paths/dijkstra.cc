// Copyright 2024 Google LLC
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

#include "in_memory/shortest_paths/dijkstra.h"

#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "in_memory/clustering/gbbs_graph.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"
#include "utils/container/fixed_size_priority_queue.h"

namespace graph_mining::in_memory {

using ::graph_mining::FixedSizePriorityQueue;
using ::graph_mining::in_memory::DirectedGbbsGraph;
using ::graph_mining::in_memory::NodeId;

absl::StatusOr<std::vector<float>> Dijkstra(
    const graph_mining::in_memory::DirectedGbbsGraph& graph,
    const graph_mining::in_memory::NodeId source_node_id) {
  const NodeId num_nodes = graph.Graph()->n;

  // Check that `source_node_id` is a valid node ID.
  if (source_node_id < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("'source_node_id' must be non-negative, but was set to ",
                     source_node_id));
  } else if (source_node_id >= num_nodes) {
    return absl::OutOfRangeError(absl::StrCat(
        "'source_node_id' must be strictly smaller than the number of nodes (",
        num_nodes, "), but was set to ", source_node_id));
  }

  // At the end of the algorithm, `distance[i]` will store the distance from
  // node `source_node_id` to node `i`, or
  // `std::numeric_limits<float>::infinity()` if node `i` is not reachable from
  // node `source_node_id`.
  std::vector<float> distances(num_nodes,
                               std::numeric_limits<float>::infinity());

  // `queue` stores the negated tentative distances from node `source_node_id`
  // to each node. `FixedSizePriorityQueue::Top()` returns the element with the
  // largest priority; we use negated distances as priorities in order to obtain
  // the candidate node with minimum distance at each step.
  using Queue =
      FixedSizePriorityQueue</*PriorityType=*/float, /*IndexType=*/NodeId>;
  Queue queue(/*size=*/num_nodes);
  queue.InsertOrUpdate(/*element=*/source_node_id, /*priority=*/-0.0);

  // Status indicating whether an error has occurred while computing the
  // distances.
  absl::Status status = absl::OkStatus();

  // Relaxes the edge from node `current_node_id` to node `neighbor_id` with
  // edge weight `edge_weight`, i.e., checks if this edge can be used to improve
  // upon the best known path from node `source_node_id` to node `neighbor_id`,
  // and if so updates the priority of `neighbor_id` accordingly. Sets `status`
  // to an error status if `edge_weight` is negative. The function is a no-op if
  // `status` is already a non-OK status before the call.
  auto relax_edge = [&distances, &queue, &status](const NodeId current_node_id,
                                                  const NodeId neighbor_id,
                                                  const float edge_weight) {
    if (!status.ok()) return;
    if (edge_weight < 0.0) {
      status = absl::InvalidArgumentError(
          absl::StrCat("Edges reachable from the source node must have "
                       "non-negative weight; the edge from node ",
                       current_node_id, " to node ", neighbor_id,
                       " has weight ", edge_weight));
      return;
    }

    // Skip this edge if it points to a node whose distance from
    // `source_node_id` is already known.
    if (distances[neighbor_id] != std::numeric_limits<float>::infinity()) {
      return;
    }

    // Update the priority of `neighbor_id` if this edge creates a better path
    // to it (or if no path to it had been found yet).
    const float distance = distances[current_node_id] + edge_weight;
    const float priority = queue.Priority(neighbor_id);
    if (priority == Queue::kInvalidPriority || distance < -priority) {
      queue.InsertOrUpdate(/*element=*/neighbor_id,
                           /*priority=*/-distance);
    }
  };

  // This while loop maintains the following invariants:
  //  - `distance[i]` is equal to the distance from `source_node_id` to `i` if
  //    it has already been determined, or to
  //    `std::numeric_limits<float>::infinity()` otherwise. Once the distance to
  //    node `i` is determined, `distance[i]` is updated to it, and then remains
  //    constant after that.
  //  - `queue` contains the nodes for which the distance from `source_node_id`
  //    has not been determined yet, with negated tentative distances as
  //    priorities. Once the distance to a node has been determined, the
  //    corresponding entry is removed from the queue and is never added back.
  while (!queue.Empty()) {
    NodeId node_id = queue.Top();
    distances[node_id] = -queue.Priority(node_id);
    queue.Remove(node_id);
    graph.Graph()->get_vertex(node_id).out_neighbors().map(relax_edge,
                                                           /*parallel=*/false);
    RETURN_IF_ERROR(status);
  }
  return distances;
}

}  // namespace graph_mining::in_memory
