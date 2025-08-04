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

#ifndef THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_
#define THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_

#include <functional>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "in_memory/clustering/types.h"
#include "in_memory/status_macros.h"
#include "utils/container/fixed_size_priority_queue.h"

namespace graph_mining::in_memory {

// Runs Dijkstra's algorithm on `graph` to compute single-source shortest-path
// distances from the source node `source_node_id`. Returns a vector whose
// length is equal to the number of nodes of the input graph, and whose `i`-th
// element is equal to the distance from node `source_node_id` to node `i` if
// node `i` is reachable from node `source_node_id`, or equal to infinity
// otherwise. Returns an error if `source_node_id` is not a valid node ID of the
// graph, or if an edge that can be reached from `source_node_id` via a directed
// path has a negative weight.
//
// `GraphType` must be one of the (symmetric or asymmetric) graph types defined
// in google3/third_party/gbbs/gbbs/graph.h; the edge weights must be
// floating-point numbers.
template <class GraphType>
absl::StatusOr<std::vector<typename GraphType::weight_type>> Dijkstra(
    const GraphType& graph, NodeId source_node_id) {
  using EdgeWeightType = typename GraphType::weight_type;
  static_assert(std::is_floating_point_v<EdgeWeightType>,
                "The edge weights must be floating-point numbers.");

  const NodeId num_nodes = graph.n;

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
  // node `source_node_id` to node `i`, or infinity if node `i` is not reachable
  // from node `source_node_id`.
  std::vector<EdgeWeightType> distances(
      num_nodes, std::numeric_limits<EdgeWeightType>::infinity());

  // `queue` stores the tentative distances from node `source_node_id`
  // to each node. `FixedSizePriorityQueue::Top()` returns the element with the
  // smallest tentative distance (since we use `std::less` as the comparator).
  using Queue =
      FixedSizePriorityQueue</*PriorityType=*/EdgeWeightType,
                             /*IndexType=*/NodeId,
                             /*LargerPriority=*/std::less<EdgeWeightType>>;
  Queue queue(/*size=*/num_nodes);
  queue.InsertOrUpdate(/*element=*/source_node_id, /*priority=*/0.0);

  // Status indicating whether an error has occurred while computing the
  // distances.
  absl::Status status = absl::OkStatus();

  // Relaxes the edge from node `current_node_id` to node `neighbor_id` with
  // edge weight `edge_weight`, i.e., checks if this edge can be used to improve
  // upon the best known path from node `source_node_id` to node `neighbor_id`,
  // and if so updates the priority of `neighbor_id` accordingly. Sets `status`
  // to an error status if `edge_weight` is negative. The function is a no-op if
  // `status` is already a non-OK status before the call.
  auto relax_edge = [&distances, &queue, &status](
                        const NodeId current_node_id, const NodeId neighbor_id,
                        const EdgeWeightType edge_weight) {
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
    if (distances[neighbor_id] !=
        std::numeric_limits<EdgeWeightType>::infinity()) {
      return;
    }

    // Update the priority of `neighbor_id` if this edge creates a better path
    // to it (or if no path to it had been found yet).
    const EdgeWeightType distance = distances[current_node_id] + edge_weight;
    const EdgeWeightType priority = queue.Priority(neighbor_id);
    if (priority == Queue::kInvalidPriority || distance < priority) {
      queue.InsertOrUpdate(/*element=*/neighbor_id,
                           /*priority=*/distance);
    }
  };

  // This while loop maintains the following invariants:
  //  - `distance[i]` is equal to the distance from `source_node_id` to `i` if
  //    it has already been determined, or to infinity otherwise. Once the
  //    distance to node `i` is determined, `distance[i]` is updated to it, and
  //    then remains constant after that.
  //  - `queue` contains the nodes for which the distance from `source_node_id`
  //    has not been determined yet, with the tentative distances as
  //    priorities. Once the distance to a node has been determined, the
  //    corresponding entry is removed from the queue and is never added back.
  while (!queue.Empty()) {
    NodeId node_id = queue.Top();
    distances[node_id] = queue.Priority(node_id);
    queue.Remove(node_id);
    graph.get_vertex(node_id).out_neighbors().map(relax_edge,
                                                  /*parallel=*/false);
    RETURN_IF_ERROR(status);
  }
  return distances;
}

}  // namespace graph_mining::in_memory

#endif  // THIRD_PARTY_GRAPH_MINING_IN_MEMORY_SHORTEST_PATHS_DIJKSTRA_H_
