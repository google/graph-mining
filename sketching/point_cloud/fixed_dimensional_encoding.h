#ifndef THIRD_PARTY_GRAPH_MINING_SKETCHING_POINT_CLOUD_FIXED_DIMENSIONAL_ENCODING_H_
#define THIRD_PARTY_GRAPH_MINING_SKETCHING_POINT_CLOUD_FIXED_DIMENSIONAL_ENCODING_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "sketching/point_cloud/fixed_dimensional_encoding_config.pb.h"

namespace graph_mining {

// These utility functions implement the process of generating a randomized
// "Fixed Dimensional Encoding" (FDE) from a variable sized set of vectors
// (called a point cloud). Specifically, the functions take as input a
// `point_cloud`, which is a concatenated list of vectors of the same dimension
// config.dimension(). The functions output a single vector (the FDE), such that
// the dot product between a query FDE and a document FDE approximates the
// Chamfer Similarity between the original query point cloud and document point
// cloud. See https://arxiv.org/pdf/2405.19504v1 for further details.

// This is a wrapper method that routes to either
// `GenerateQueryFixedDimensionalEncoding` or
// `GenerateDocumentFixedDimensionalEncoding` based on config.EncodingType().
absl::StatusOr<std::vector<float>> GenerateFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

absl::StatusOr<std::vector<float>> GenerateQueryFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

absl::StatusOr<std::vector<float>> GenerateDocumentFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config);

namespace internal {  // For testing only

// Returns the partition index of the given vector. This is computed by
// thresholding `input_vector` by mapping positive entries to 1 and negative
// entries to 0, and then interpreting the result as a binary vector and
// converting it to an int using the Gray code conversion from binary vectors to
// ints.
uint32_t SimHashPartitionIndex(const Eigen::VectorXf& input_vector);

uint32_t DistanceToSimHashPartition(const Eigen::VectorXf& input_vector,
                                    uint32_t index);

// Applies a random projection to a vector using Count-Sketch matrix, which is a
// sparse random matrix. Specifically, each each entry from the input is added
// to a single random entry in the output vector with a random sign.
std::vector<float> ApplyCountSketchToVector(
    absl::Span<const float> input_vector, uint32_t final_dimension,
    uint32_t seed);
}  // namespace internal

}  // namespace graph_mining

#endif  // THIRD_PARTY_GRAPH_MINING_SKETCHING_POINT_CLOUD_FIXED_DIMENSIONAL_ENCODING_H_
