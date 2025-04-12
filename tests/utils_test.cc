// Copyright 2025 Omkar Prabhu
#include "tokenizers/utils.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "tokenizers/common.h"

using tokenizers::Encoding;
using tokenizers::Padding;
using tokenizers::PaddingDirection;
using tokenizers::PaddingStrategy;
using tokenizers::PadEncoding;
using tokenizers::TruncateEncoding;
using tokenizers::Truncation;
using tokenizers::TruncationDirection;
using tokenizers::TruncationStrategy;

void assertUtilsValues(std::vector<Encoding> got,
                       std::vector<Encoding> expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    ASSERT_EQ(got[i].ids.size(), expected[i].ids.size());
    ASSERT_EQ(got[i].type_ids.size(), expected[i].type_ids.size());
    ASSERT_EQ(got[i].tokens.size(), expected[i].tokens.size());
    ASSERT_EQ(got[i].offsets.size(), expected[i].offsets.size());
    ASSERT_EQ(got[i].word_ids.size(), expected[i].word_ids.size());
    ASSERT_EQ(got[i].special_tokens_mask.size(),
              expected[i].special_tokens_mask.size());
    ASSERT_EQ(got[i].attention_mask.size(), expected[i].attention_mask.size());
    ASSERT_EQ(got[i].overflowing.size(), expected[i].overflowing.size());
    for (int j = 0; j < got[i].ids.size(); j++) {
      ASSERT_EQ(got[i].ids[j], expected[i].ids[j]);
      ASSERT_EQ(got[i].type_ids[j], expected[i].type_ids[j]);
      ASSERT_EQ(got[i].tokens[j], expected[i].tokens[j]);
      ASSERT_EQ(got[i].offsets[j], expected[i].offsets[j]);
      ASSERT_EQ(got[i].word_ids[j], expected[i].word_ids[j]);
      ASSERT_EQ(got[i].special_tokens_mask[j],
                expected[i].special_tokens_mask[j]);
      ASSERT_EQ(got[i].attention_mask[j], expected[i].attention_mask[j]);
    }
    for (int j = 0; j < got[i].overflowing.size(); j++) {
      assertUtilsValues({got[i].overflowing[j]}, {expected[i].overflowing[j]});
    }
  }
}

TEST(TruncateEncodingTest, GreaterMaxLength) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  Encoding expected({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                    {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                    {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  TruncateEncoding(&input, 10, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, MaxLengthZero) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  Encoding expected({}, {}, {}, {}, {}, {}, {});
  expected.overflowing = {input};
  TruncateEncoding(&input, 0, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, TruncateRight) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  Encoding expected({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                    {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  expected.overflowing = {
      Encoding({2, 3, 4}, {0, 0, 0}, {"b", "c", "d"}, {{2, 3}, {4, 5}, {6, 7}},
               {1, 2, 3}, {0, 0, 0}, {1, 1, 1}),
      Encoding({3, 4, 5}, {0, 0, 0}, {"c", "d", "e"}, {{4, 5}, {6, 7}, {8, 9}},
               {2, 3, 4}, {0, 0, 0}, {1, 1, 1})};
  TruncateEncoding(&input, 3, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, TruncateLeft) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  Encoding expected({3, 4, 5}, {0, 0, 0}, {"c", "d", "e"},
                    {{4, 5}, {6, 7}, {8, 9}}, {2, 3, 4}, {0, 0, 0}, {1, 1, 1});
  expected.overflowing = {
      Encoding({2, 3, 4}, {0, 0, 0}, {"b", "c", "d"}, {{2, 3}, {4, 5}, {6, 7}},
               {1, 2, 3}, {0, 0, 0}, {1, 1, 1}),
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1})};
  TruncateEncoding(&input, 3, 2, TruncationDirection::kLeft);
  assertUtilsValues({input}, {expected});
}

TEST(TruncationTest, MaxLengthZero) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 0, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
               {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1})};
  std::vector<Encoding> expected = {Encoding()};
  expected[0].overflowing = {input[0]};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyLongestFirst) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 6, 2);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
               {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}),
      Encoding({6, 7, 8, 9}, {0, 0, 0, 0}, {"f", "g", "h", "i"},
               {{10, 11}, {12, 13}, {14, 15}, {16, 17}}, {5, 6, 7, 8},
               {0, 0, 0, 0}, {1, 1, 1, 1})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({6, 7, 8}, {0, 0, 0}, {"f", "g", "h"},
               {{10, 11}, {12, 13}, {14, 15}}, {5, 6, 7}, {0, 0, 0},
               {1, 1, 1})};

  expected[0].overflowing = {
      Encoding({2, 3, 4}, {0, 0, 0}, {"b", "c", "d"}, {{2, 3}, {4, 5}, {6, 7}},
               {1, 2, 3}, {0, 0, 0}, {1, 1, 1}),
      Encoding({3, 4, 5}, {0, 0, 0}, {"c", "d", "e"}, {{4, 5}, {6, 7}, {8, 9}},
               {2, 3, 4}, {0, 0, 0}, {1, 1, 1})};
  expected[1].overflowing = {Encoding({7, 8, 9}, {0, 0, 0}, {"g", "h", "i"},
                                      {{12, 13}, {14, 15}, {16, 17}}, {6, 7, 8},
                                      {0, 0, 0}, {1, 1, 1})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyOnlyFirst) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlyFirst, 6, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5, 6, 7}, {0, 0, 0, 0, 0, 0, 0},
               {"a", "b", "c", "d", "e", "f", "g"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}},
               {0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0},
               {1, 1, 1, 1, 1, 1, 1}),
      Encoding({8, 9, 10}, {0, 0, 0}, {"h", "i", "j"},
               {{14, 15}, {16, 17}, {18, 19}}, {7, 8, 9}, {0, 0, 0},
               {1, 1, 1})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({8, 9, 10}, {0, 0, 0}, {"h", "i", "j"},
               {{14, 15}, {16, 17}, {18, 19}}, {7, 8, 9}, {0, 0, 0},
               {1, 1, 1})};
  expected[0].overflowing = {
      Encoding({4, 5, 6}, {0, 0, 0}, {"d", "e", "f"},
               {{6, 7}, {8, 9}, {10, 11}}, {3, 4, 5}, {0, 0, 0}, {1, 1, 1}),
      Encoding({7}, {0}, {"g"}, {{12, 13}}, {6}, {0}, {1})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyOnlySecond) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlySecond, 8, 3);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding(
          {4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 0, 0},
          {"d", "e", "f", "g", "h", "i", "j"},
          {{6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19}},
          {3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({4, 5, 6, 7, 8}, {0, 0, 0, 0, 0}, {"d", "e", "f", "g", "h"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}}, {3, 4, 5, 6, 7},
               {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1})};
  expected[1].overflowing = {
      Encoding({6, 7, 8, 9, 10}, {0, 0, 0, 0, 0}, {"f", "g", "h", "i", "j"},
               {{10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19}},
               {5, 6, 7, 8, 9}, {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(PadEncodingTest, GreaterTargetLength) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  Encoding expected(
      {1, 2, 3, 0, 0}, {0, 0, 0, 1, 1}, {"a", "b", "c", "[PAD]", "[PAD]"},
      {{0, 1}, {2, 3}, {4, 5}, {0, 0}, {0, 0}},
      {0, 1, 2, std::nullopt, std::nullopt}, {0, 0, 0, 1, 1}, {1, 1, 1, 0, 0});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(PadEncodingTest, PadLeft) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  Encoding expected(
      {0, 0, 1, 2, 3}, {1, 1, 0, 0, 0}, {"[PAD]", "[PAD]", "a", "b", "c"},
      {{0, 0}, {0, 0}, {0, 1}, {2, 3}, {4, 5}},
      {std::nullopt, std::nullopt, 0, 1, 2}, {1, 1, 0, 0, 0}, {0, 0, 1, 1, 1});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kLeft);
  assertUtilsValues({input}, {expected});
}

TEST(PadEncodingTest, PadRight) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  Encoding expected(
      {1, 2, 3, 0, 0}, {0, 0, 0, 1, 1}, {"a", "b", "c", "[PAD]", "[PAD]"},
      {{0, 1}, {2, 3}, {4, 5}, {0, 0}, {0, 0}},
      {0, 1, 2, std::nullopt, std::nullopt}, {0, 0, 0, 1, 1}, {1, 1, 1, 0, 0});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(PaddingTest, StrategyBatchLongest) {
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kBatchLongest, 0,
                  0, 0, 0, "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}}, {3, 4, 5, 6}, {0, 0, 0, 0},
               {1, 1, 1, 1}),
      Encoding({8, 9}, {0, 0}, {"h", "i"}, {{14, 15}, {16, 17}}, {7, 8}, {0, 0},
               {1, 1})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3, 0}, {0, 0, 0, 0}, {"a", "b", "c", "[PAD]"},
               {{0, 1}, {2, 3}, {4, 5}, {0, 0}}, {0, 1, 2, std::nullopt},
               {0, 0, 0, 1}, {1, 1, 1, 0}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}}, {3, 4, 5, 6}, {0, 0, 0, 0},
               {1, 1, 1, 1}),
      Encoding({8, 9, 0, 0}, {0, 0, 0, 0}, {"h", "i", "[PAD]", "[PAD]"},
               {{14, 15}, {16, 17}, {0, 0}, {0, 0}},
               {7, 8, std::nullopt, std::nullopt}, {0, 0, 1, 1}, {1, 1, 0, 0})};
  std::vector<Encoding> got = padding.PadEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(PaddingTest, StrategyFixed) {
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kFixed, 5, 0, 0, 0,
                  "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}}, {3, 4, 5, 6}, {0, 0, 0, 0},
               {1, 1, 1, 1}),
      Encoding({8, 9}, {0, 0}, {"h", "i"}, {{14, 15}, {16, 17}}, {7, 8}, {0, 0},
               {1, 1})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3, 0, 0}, {0, 0, 0, 0, 0},
               {"a", "b", "c", "[PAD]", "[PAD]"},
               {{0, 1}, {2, 3}, {4, 5}, {0, 0}, {0, 0}},
               {0, 1, 2, std::nullopt, std::nullopt}, {0, 0, 0, 1, 1},
               {1, 1, 1, 0, 0}),
      Encoding({4, 5, 6, 7, 0}, {0, 0, 0, 0, 0}, {"d", "e", "f", "g", "[PAD]"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}, {0, 0}},
               {3, 4, 5, 6, std::nullopt}, {0, 0, 0, 0, 1}, {1, 1, 1, 1, 0}),
      Encoding({8, 9, 0, 0, 0}, {0, 0, 0, 0, 0},
               {"h", "i", "[PAD]", "[PAD]", "[PAD]"},
               {{14, 15}, {16, 17}, {0, 0}, {0, 0}, {0, 0}},
               {7, 8, std::nullopt, std::nullopt, std::nullopt},
               {0, 0, 1, 1, 1}, {1, 1, 0, 0, 0})};
  std::vector<Encoding> got = padding.PadEncodings(input);
  assertUtilsValues(got, expected);
}
