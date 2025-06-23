#include "sketching/point_cloud/fixed_dimensional_encoding.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "sketching/point_cloud/fixed_dimensional_encoding_config.pb.h"

namespace graph_mining {

using Coordinate = float;
using Point = std::vector<Coordinate>;
using MatrixRowMajor =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ::Eigen::MatrixXf;

namespace internal {
namespace {

constexpr inline uint64_t AppendToGrayCode(uint64_t gray_code, bool bit) {
  return (gray_code << 1) + (bit ^ (gray_code & 1));
}

// Returns the binary representation of the given Gray Code.
constexpr inline uint32_t GrayCodeToBinary(uint32_t num) {
  return num ^ (num >> 1);
}

// Returns a matrix of independent +/-1 variables with dimension
// `dimension` x `num_projections`.
MatrixXf AMSProjectionMatrixFromSeed(uint32_t seed, uint32_t dimension,
                                     uint32_t projection_dim) {
  MatrixXf out = MatrixXf::Zero(dimension, projection_dim);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> index_dist(0, projection_dim - 1);
  std::bernoulli_distribution sign_dist(0.5);
  for (int i = 0; i < dimension; ++i) {
    size_t index = index_dist(gen);
    out(i, index) = 2.0 * sign_dist(gen) - 1.0;
  }
  return out;
}

// Returns a matrix of independent Gaussian variables with dimension
// `dimension` x `num_projections`.
MatrixXf SimHashMatrixFromSeed(uint32_t seed, uint32_t dimension,
                               uint32_t num_projections) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  return MatrixXf::NullaryExpr(dimension, num_projections,
                               [&]() { return dist(gen); });
}

}  // namespace

uint32_t SimHashPartitionIndex(const Eigen::VectorXf& input_vector) {
  uint32_t partition_index = 0;
  for (int i = 0; i < input_vector.size(); ++i) {
    partition_index = AppendToGrayCode(partition_index, input_vector[i] > 0);
  }
  return partition_index;
}

uint32_t DistanceToSimHashPartition(const Eigen::VectorXf& input_vector,
                                    uint32_t index) {
  uint32_t distance = 0;
  uint32_t binary_representation = GrayCodeToBinary(index);
  for (int i = input_vector.size() - 1; i >= 0; --i) {
    int32_t cur_bit = (input_vector[i] > 0 ? 1 : 0);
    distance += (cur_bit != (binary_representation & 1));
    binary_representation >>= 1;
  }
  return distance;
}

std::vector<float> ApplyCountSketchToVector(
    absl::Span<const float> input_vector, uint32_t final_dimension,
    uint32_t seed) {
  std::vector<float> out;
  out.resize(final_dimension);
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int32_t> index_dist(0, final_dimension - 1);
  std::bernoulli_distribution sign_dist(0.5);
  for (int i = 0; i < input_vector.size(); ++i) {
    int32_t index = index_dist(gen);
    float scaling = 2.0 * sign_dist(gen) - 1.0;
    out[index] += scaling * input_vector[i];
  }
  return out;
}

}  // namespace internal

absl::StatusOr<std::vector<Coordinate>> GenerateFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config) {
  switch (config.encoding_type()) {
    case FixedDimensionalEncodingConfig::DEFAULT_SUM:
      return GenerateQueryFixedDimensionalEncoding(point_cloud, config);
    case FixedDimensionalEncodingConfig::AVERAGE:
      return GenerateDocumentFixedDimensionalEncoding(point_cloud, config);
    default:
      return absl::InvalidArgumentError("Unsupported encoding type");
  }
}

absl::StatusOr<std::vector<Coordinate>> GenerateQueryFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config) {
  if (point_cloud.size() % config.dimension() != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatching point cloud size (", point_cloud.size(),
                     ") and dimension (", config.dimension(), ")."));
  }
  if (config.num_simhash_projections() >= 31 ||
      config.num_simhash_projections() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported number of simhash projections: ",
                     config.num_simhash_projections()));
  }
  if (config.fill_empty_partitions()) {
    return absl::InvalidArgumentError(
        "Query FDE encoding does not support fill_empty_partitions.");
  }

  int32_t num_points = point_cloud.size() / config.dimension();

  Eigen::Map<MatrixRowMajor> embedding_matrix(
      const_cast<float*>(point_cloud.data()), num_points, config.dimension());
  bool use_identity_encoding =
      (config.projection_type() ==
       FixedDimensionalEncodingConfig::DEFAULT_IDENTITY);
  int32_t projection_dim = use_identity_encoding
                               ? config.dimension()
                               : config.projection_dimension();
  std::vector<float> out;
  int32_t num_partitions = std::pow(2, config.num_simhash_projections());
  out.resize(config.num_repetitions() * num_partitions * projection_dim);

  for (int repetition_num = 0; repetition_num < config.num_repetitions();
       ++repetition_num) {
    MatrixXf sketch_matrix;
    if (config.num_simhash_projections() > 0) {
      sketch_matrix = embedding_matrix * internal::SimHashMatrixFromSeed(
                                             config.seed() + repetition_num,
                                             config.dimension(),
                                             config.num_simhash_projections());
    }
    MatrixXf projected_matrix;
    if (use_identity_encoding) {
      projected_matrix = embedding_matrix;
    } else {
      if (config.projection_dimension() <= 0) {
        return absl::InvalidArgumentError(
            "A positive projection dimension must be specified in the config "
            "if a non-identity projection type is used.");
      }
      projected_matrix =
          embedding_matrix * internal::AMSProjectionMatrixFromSeed(
                                 config.seed() + repetition_num,
                                 config.dimension(), projection_dim);
    }
    // Add projected points to their respective partitions in the repetition.
    for (int cur_point = 0; cur_point < num_points; ++cur_point) {
      int32_t partition_index =
          (config.num_simhash_projections() > 0)
              ? internal::SimHashPartitionIndex(sketch_matrix.row(cur_point))
              : 0;
      int32_t index = repetition_num * (num_partitions * projection_dim) +
                      partition_index * projection_dim;
      if (index + projection_dim - 1 >= out.size()) {
        return absl::InternalError("Index out of bounds.");
      }
      for (int k = 0; k < projection_dim; ++k) {
        out[index + k] += projected_matrix.coeff(cur_point, k);
      }
    }  // Finished processing all points in this repetition
  }  // Finished processing all repetitions.

  if (config.has_final_projection_dimension()) {
    return internal::ApplyCountSketchToVector(
        out, config.final_projection_dimension(), config.seed());
  }
  return out;
}

absl::StatusOr<std::vector<Coordinate>>
GenerateDocumentFixedDimensionalEncoding(
    const std::vector<float>& point_cloud,
    const FixedDimensionalEncodingConfig& config) {
  if (point_cloud.size() % config.dimension() != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatching point cloud size (", point_cloud.size(),
                     ") and dimension (", config.dimension(), ")."));
  }
  if (config.num_simhash_projections() >= 31 ||
      config.num_simhash_projections() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported number of simhash projections: ",
                     config.num_simhash_projections()));
  }

  int32_t num_points = point_cloud.size() / config.dimension();

  Eigen::Map<MatrixRowMajor> embedding_matrix(
      const_cast<float*>(point_cloud.data()), num_points, config.dimension());
  bool use_identity_encoding =
      (config.projection_type() ==
       FixedDimensionalEncodingConfig::DEFAULT_IDENTITY);
  int32_t projection_dim = use_identity_encoding
                               ? config.dimension()
                               : config.projection_dimension();
  std::vector<float> out;
  int32_t num_partitions = std::pow(2, config.num_simhash_projections());
  out.resize(config.num_repetitions() * num_partitions * projection_dim);
  for (int repetition_num = 0; repetition_num < config.num_repetitions();
       ++repetition_num) {
    MatrixXf sketch_matrix;
    if (config.num_simhash_projections() > 0) {
      sketch_matrix = embedding_matrix * internal::SimHashMatrixFromSeed(
                                             config.seed() + repetition_num,
                                             config.dimension(),
                                             config.num_simhash_projections());
    }
    MatrixXf projected_matrix;
    if (use_identity_encoding) {
      projected_matrix = embedding_matrix;
    } else {
      if (config.projection_dimension() <= 0) {
        return absl::InvalidArgumentError(
            "A positive projection dimension must be specified in the config "
            "if a non-identity projection type is used.");
      }
      projected_matrix =
          embedding_matrix * internal::AMSProjectionMatrixFromSeed(
                                 config.seed() + repetition_num,
                                 config.dimension(), projection_dim);
    }

    std::vector<float> partition_sizes;

    partition_sizes.resize(num_partitions);
    for (int cur_point = 0; cur_point < num_points; ++cur_point) {
      int32_t partition_index =
          (config.num_simhash_projections() > 0)
              ? internal::SimHashPartitionIndex(sketch_matrix.row(cur_point))
              : 0;
      int32_t index = repetition_num * (num_partitions * projection_dim) +
                      partition_index * projection_dim;
      if (index + projection_dim - 1 >= out.size()) {
        return absl::InternalError("Index out of bounds.");
      }
      for (int k = 0; k < projection_dim; ++k) {
        out[index + k] += projected_matrix.coeff(cur_point, k);
      }
      partition_sizes[partition_index] += 1.0;
    }  // Finished processing all points in this repetition

    for (int partition_index = 0; partition_index < num_partitions;
         ++partition_index) {
      int32_t index = repetition_num * (num_partitions * projection_dim) +
                      partition_index * projection_dim;
      // If a partition is empty and 'fill_empty_partitions' is enabled, we add
      // the closest point to the coordinates of the FDE corresponding to the
      // empty partition.
      if (partition_sizes[partition_index] == 0.0 &&
          config.num_simhash_projections() > 0) {
        if (config.fill_empty_partitions()) {
          int32_t nearest_point_index = -1;
          uint32_t min_distance = std::numeric_limits<uint32_t>::max();
          for (int cur_point = 0; cur_point < num_points; ++cur_point) {
            uint32_t cur_distance = internal::DistanceToSimHashPartition(
                sketch_matrix.row(cur_point), partition_index);
            if (cur_distance < min_distance) {
              min_distance = cur_distance;
              nearest_point_index = cur_point;
            }
          }
          for (int k = 0; k < projection_dim; ++k) {
            out[index + k] = projected_matrix.coeff(nearest_point_index, k);
          }
        }
        continue;
      }
      // Now normalize each bucket to be the centroid instead of sum of the
      // bucket.
      for (int k = 0; k < projection_dim; ++k) {
        out[index + k] = out[index + k] / partition_sizes[partition_index];
      }
    }
  }  // Finished processing all repetitions.

  if (config.has_final_projection_dimension()) {
    return internal::ApplyCountSketchToVector(
        out, config.final_projection_dimension(), config.seed());
  }
  return out;
}

}  // namespace graph_mining
