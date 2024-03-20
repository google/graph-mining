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

#include "in_memory/clustering/dynamic/hac/dynamic_dendrogram.h"

#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "in_memory/clustering/dendrogram.h"
#include "in_memory/clustering/parallel_dendrogram.h"
#include "in_memory/status_macros.h"  // IWYU pragma: keep

namespace graph_mining::in_memory {

namespace {

using ParentEdge = DynamicDendrogram::ParentEdge;
using ::testing::AnyOf;
using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::UnorderedElementsAre;



using NodeId = DynamicDendrogram::NodeId;
using graph_mining::in_memory::Dendrogram;
using graph_mining::in_memory::ParallelDendrogram;

TEST(ConstructionTest, Empty) {
  DynamicDendrogram dendrogram;
  EXPECT_EQ(0, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.Parent(1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Sibling(1), Eq(std::nullopt));
  EXPECT_THAT(dendrogram.AddInternalNode(0, 0, 0, 0),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("child node not in dendrogram")));
  EXPECT_THAT(dendrogram.RemoveSingletonLeafNode(1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(AddNodeTest, AddOneLeaf) {
  DynamicDendrogram dendrogram;
  EXPECT_EQ(0, dendrogram.NumNodes());
  ASSERT_OK(dendrogram.AddLeafNode(2));
  EXPECT_EQ(1, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Sibling(2), Eq(std::nullopt));
  EXPECT_OK(dendrogram.RemoveSingletonLeafNode(2));
  EXPECT_EQ(0, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.Parent(2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(AddNodeTest, Merge) {
  DynamicDendrogram dendrogram;
  EXPECT_EQ(0, dendrogram.NumNodes());
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  EXPECT_EQ(2, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));

  ASSERT_OK(dendrogram.AddInternalNode(4, 1, 2, 0.5));
  EXPECT_EQ(2, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.HasValidParent(4), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(true));
  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(true));

  EXPECT_THAT(dendrogram.Parent(1), IsOkAndHolds(FieldsAre(4, 0.5)));
  EXPECT_THAT(dendrogram.Parent(2), IsOkAndHolds(FieldsAre(4, 0.5)));
}

TEST(RemoveAncestorsTest, RemoveOneNode) {
  // Remove one node from a 3-level balanced dendrogram. Node 1's ancestors 5
  // and 7 are removed.
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& ancestors, dendrogram.RemoveAncestors({1}));
  EXPECT_EQ(4, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Parent(7),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(6), IsOkAndHolds(false));

  EXPECT_THAT(dendrogram.Parent(3), IsOkAndHolds(FieldsAre(6, 0.5)));
  EXPECT_THAT(dendrogram.Parent(4), IsOkAndHolds(FieldsAre(6, 0.5)));

  EXPECT_THAT(ancestors, UnorderedElementsAre(5, 7));
}

TEST(RemoveAncestorsTest, RemoveTwoNodes) {
  //    14
  //   /   \
  //  13    \
  //  / \    \
  // 10  11  12
  DynamicDendrogram dendrogram;
  for (const NodeId& i : {10, 11, 12}) {
    ASSERT_OK(dendrogram.AddLeafNode(i));
  }
  ASSERT_OK(dendrogram.AddInternalNode(13, 10, 11, 0.2));
  ASSERT_OK(dendrogram.AddInternalNode(14, 13, 12, 0.04));

  ASSERT_OK_AND_ASSIGN(const auto& ancestors,
                       dendrogram.RemoveAncestors({10, 11}));
  EXPECT_EQ(3, dendrogram.NumNodes());

  EXPECT_THAT(dendrogram.HasValidParent(10), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(11), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(12), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(13),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Parent(14),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_THAT(ancestors, UnorderedElementsAre(13, 14));
}

TEST(RemoveAncestorsTest, RemoveNonExistingNode) {
  //    14
  //   /   \
  //  13    \
  //  / \    \
  // 10  11  12
  DynamicDendrogram dendrogram;
  for (const NodeId& i : {10, 11, 12}) {
    ASSERT_OK(dendrogram.AddLeafNode(i));
  }
  ASSERT_OK(dendrogram.AddInternalNode(13, 10, 11, 0.2));
  ASSERT_OK(dendrogram.AddInternalNode(14, 13, 12, 0.04));

  EXPECT_THAT(dendrogram.RemoveAncestors({0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
}

TEST(RemoveAncestorsTest, RemoveFromMultipleTrees) {
  // Remove one node from a 3-level balanced dendrogram. Node 1's ancestors 5
  // and 7 are removed. Node 8's ancestor 10 is removed.
  //     7
  //   /   \
  //  5     6      10
  //  / \   /\     / \
  // 1   2  3  4  8   9
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));
  ASSERT_OK(dendrogram.AddLeafNode(8));
  ASSERT_OK(dendrogram.AddLeafNode(9));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(10, 8, 9, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& ancestors,
                       dendrogram.RemoveAncestors({1, 8}));
  EXPECT_EQ(6, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.HasValidParent(1), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.Parent(5),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Parent(7),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));
  EXPECT_THAT(dendrogram.Parent(10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_THAT(dendrogram.HasValidParent(2), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(6), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(9), IsOkAndHolds(false));

  EXPECT_THAT(dendrogram.Parent(3), IsOkAndHolds(FieldsAre(6, 0.5)));
  EXPECT_THAT(dendrogram.Parent(4), IsOkAndHolds(FieldsAre(6, 0.5)));

  EXPECT_THAT(ancestors, UnorderedElementsAre(5, 7, 10));
}

TEST(RemoveAncestorsTest, RemoveOneInternalNode) {
  // Remove one node from a 3-level balanced dendrogram. Node 5's ancestor 7 is
  // removed.
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& ancestors, dendrogram.RemoveAncestors({5}));
  EXPECT_EQ(4, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.Parent(7),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_THAT(dendrogram.HasValidParent(5), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(6), IsOkAndHolds(false));

  EXPECT_THAT(dendrogram.Parent(1), IsOkAndHolds(FieldsAre(5, 0.5)));
  EXPECT_THAT(dendrogram.Parent(2), IsOkAndHolds(FieldsAre(5, 0.5)));
  EXPECT_THAT(dendrogram.Parent(3), IsOkAndHolds(FieldsAre(6, 0.5)));
  EXPECT_THAT(dendrogram.Parent(4), IsOkAndHolds(FieldsAre(6, 0.5)));

  EXPECT_THAT(ancestors, UnorderedElementsAre(7));
}

TEST(RemoveAncestorsTest, RemoveTwoInternalNodes) {
  // Remove one node from a 3-level balanced dendrogram. Node 5 and 6's ancestor
  // 7 is removed.
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& ancestors,
                       dendrogram.RemoveAncestors({5, 6}));
  EXPECT_EQ(4, dendrogram.NumNodes());
  EXPECT_THAT(dendrogram.Parent(7),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("node not in dendrogram")));

  EXPECT_THAT(dendrogram.HasValidParent(5), IsOkAndHolds(false));
  EXPECT_THAT(dendrogram.HasValidParent(6), IsOkAndHolds(false));

  EXPECT_THAT(dendrogram.Parent(1), IsOkAndHolds(FieldsAre(5, 0.5)));
  EXPECT_THAT(dendrogram.Parent(2), IsOkAndHolds(FieldsAre(5, 0.5)));
  EXPECT_THAT(dendrogram.Parent(3), IsOkAndHolds(FieldsAre(6, 0.5)));
  EXPECT_THAT(dendrogram.Parent(4), IsOkAndHolds(FieldsAre(6, 0.5)));

  EXPECT_THAT(ancestors, UnorderedElementsAre(7));
}

TEST(SiblingTest, GetSibling) {
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  EXPECT_THAT(dendrogram.Sibling(5), Optional(6));
  EXPECT_THAT(dendrogram.Sibling(6), Optional(5));
  EXPECT_THAT(dendrogram.Sibling(1), Optional(2));
  EXPECT_THAT(dendrogram.Sibling(7), Eq(std::nullopt));
  EXPECT_THAT(dendrogram.Sibling(8), Eq(std::nullopt));
}

absl::Status ConstructDendrogram(
    const std::vector<DynamicDendrogram::Merge>& merges,
    const std::vector<NodeId>& leaves) {
  DynamicDendrogram dendrogram;
  for (const auto i : leaves) {
    RETURN_IF_ERROR(dendrogram.AddLeafNode(i));
  }
  for (const auto merge : merges) {
    const auto [merge_similarity, node_a, node_b, parent_id] = merge;
    RETURN_IF_ERROR(dendrogram.AddInternalNode(parent_id, node_a, node_b,
                                               merge_similarity));
  }
  return absl::OkStatus();
}

TEST(GetMergeSequenceTest, SimpleTest) {
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.7));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  const auto& [merges, leaves] = dendrogram.MergeSequence();
  EXPECT_THAT(leaves, UnorderedElementsAre(1, 2, 3, 4));
  EXPECT_THAT(
      merges,
      AnyOf(ElementsAre(FieldsAre(DoubleNear(0.7, 1e-6), 1, 2, 5),
                        FieldsAre(0.5, 3, 4, 6), FieldsAre(0.5, 5, 6, 7)),
            ElementsAre(FieldsAre(0.5, 3, 4, 6),
                        FieldsAre(DoubleNear(0.7, 1e-6), 1, 2, 5),
                        FieldsAre(0.5, 5, 6, 7))));
  EXPECT_OK(ConstructDendrogram(merges, leaves));
}

// Returns the parent of `id1` and `id2` in `dendrogram`. Test if `id1` and
// `id2` are merged in `dendrogram` with `weight`.
NodeId TestDendrogram(const Dendrogram& dendrogram, const NodeId& id1,
                      const NodeId& id2, float weight) {
  const auto parent_weight = dendrogram.GetParent(id1);
  EXPECT_THAT(parent_weight.merge_similarity, DoubleEq(weight)) << id1;

  const auto parent_weight_2 = dendrogram.GetParent(id2);
  EXPECT_THAT(parent_weight_2,
              FieldsAre(parent_weight.parent_id, DoubleEq(weight)))
      << id2;
  return parent_weight_2.parent_id;
}

TEST(ConvertToDendrogramTest, SimpleTest) {
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.7));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& result, dendrogram.ConvertToDendrogram());
  const auto& [new_dendrogram, node_ids] = result;
  EXPECT_THAT(node_ids, ElementsAre(1, 2, 3, 4));
  const auto parent1 = TestDendrogram(new_dendrogram, 0, 1, 0.7);
  const auto parent2 = TestDendrogram(new_dendrogram, 2, 3, 0.5);
  EXPECT_THAT(std::vector<NodeId>({parent1, parent2}),
              UnorderedElementsAre(4, 5));
  TestDendrogram(new_dendrogram, 4, 5, 0.5);
  EXPECT_FALSE(new_dendrogram.HasValidParent(6));
}

TEST(ConvertToDendrogramTest, NonConsecutiveNodesTest) {
  //    102
  //   /   \
  //  100   101
  //  / \    /\
  // 2   4  6  8
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(4));
  ASSERT_OK(dendrogram.AddLeafNode(6));
  ASSERT_OK(dendrogram.AddLeafNode(8));

  ASSERT_OK(dendrogram.AddInternalNode(100, 2, 4, 0.7));
  ASSERT_OK(dendrogram.AddInternalNode(101, 6, 8, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(102, 100, 101, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& result, dendrogram.ConvertToDendrogram());
  const auto& [new_dendrogram, node_ids] = result;
  EXPECT_THAT(node_ids, ElementsAre(2, 4, 6, 8));
  const auto parent1 = TestDendrogram(new_dendrogram, 0, 1, 0.7);
  const auto parent2 = TestDendrogram(new_dendrogram, 2, 3, 0.5);
  EXPECT_THAT(std::vector<NodeId>({parent1, parent2}),
              UnorderedElementsAre(4, 5));
  TestDendrogram(new_dendrogram, 4, 5, 0.5);
  EXPECT_FALSE(new_dendrogram.HasValidParent(6));
}

// Returns the parent of `id1` and `id2` in `dendrogram`. Test if `id1` and
// `id2` are merged in `dendrogram` with `weight`.
NodeId TestParallelDendrogram(const ParallelDendrogram& dendrogram,
                              const NodeId& id1, const NodeId& id2,
                              float weight) {
  const auto parent_weight = dendrogram.GetParent(id1);
  EXPECT_THAT(parent_weight.merge_similarity, FloatEq(weight)) << id1;

  const auto parent_weight_2 = dendrogram.GetParent(id2);
  EXPECT_THAT(parent_weight_2,
              FieldsAre(parent_weight.parent_id, FloatEq(weight)))
      << id2;
  return parent_weight_2.parent_id;
}

TEST(ConvertToParallelDendrogramTest, SimpleTest) {
  //     7
  //   /   \
  //  5     6
  //  / \   /\
  // 1   2  3  4
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(1));
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(3));
  ASSERT_OK(dendrogram.AddLeafNode(4));

  ASSERT_OK(dendrogram.AddInternalNode(5, 1, 2, 0.7));
  ASSERT_OK(dendrogram.AddInternalNode(6, 3, 4, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(7, 5, 6, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       dendrogram.ConvertToParallelDendrogram());
  const auto& [new_dendrogram, node_ids] = result;
  EXPECT_THAT(node_ids, ElementsAre(1, 2, 3, 4));
  const auto parent1 = TestParallelDendrogram(new_dendrogram, 0, 1, 0.7);
  const auto parent2 = TestParallelDendrogram(new_dendrogram, 2, 3, 0.5);
  EXPECT_THAT(std::vector<NodeId>({parent1, parent2}),
              UnorderedElementsAre(4, 5));
  TestParallelDendrogram(new_dendrogram, 4, 5, 0.5);
  EXPECT_FALSE(new_dendrogram.HasValidParent(6));
}

TEST(ConvertToParallelDendrogramTest, NonConsecutiveNodesTest) {
  //    102
  //   /   \
  //  100   101
  //  / \    /\
  // 2   4  6  8
  DynamicDendrogram dendrogram;
  ASSERT_OK(dendrogram.AddLeafNode(2));
  ASSERT_OK(dendrogram.AddLeafNode(4));
  ASSERT_OK(dendrogram.AddLeafNode(6));
  ASSERT_OK(dendrogram.AddLeafNode(8));

  ASSERT_OK(dendrogram.AddInternalNode(100, 2, 4, 0.7));
  ASSERT_OK(dendrogram.AddInternalNode(101, 6, 8, 0.5));
  ASSERT_OK(dendrogram.AddInternalNode(102, 100, 101, 0.5));

  ASSERT_OK_AND_ASSIGN(const auto& result,
                       dendrogram.ConvertToParallelDendrogram());
  const auto& [new_dendrogram, node_ids] = result;
  EXPECT_THAT(node_ids, ElementsAre(2, 4, 6, 8));
  const auto parent1 = TestParallelDendrogram(new_dendrogram, 0, 1, 0.7);
  const auto parent2 = TestParallelDendrogram(new_dendrogram, 2, 3, 0.5);
  EXPECT_THAT(std::vector<NodeId>({parent1, parent2}),
              UnorderedElementsAre(4, 5));
  TestParallelDendrogram(new_dendrogram, 4, 5, 0.5);
  EXPECT_FALSE(new_dendrogram.HasValidParent(6));
}

}  // namespace
}  // namespace graph_mining::in_memory
