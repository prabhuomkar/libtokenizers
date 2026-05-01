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
    for (int j = 0; j < got[i].ids.size(); j++) {
      ASSERT_EQ(got[i].ids[j], expected[i].ids[j]);
      ASSERT_EQ(got[i].type_ids[j], expected[i].type_ids[j]);
      ASSERT_EQ(got[i].tokens[j], expected[i].tokens[j]);
    }
  }
}

TEST(TruncateEncodingTest, GreaterMaxLength) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"});
  Encoding expected({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0},
                    {"a", "b", "c", "d", "e"});
  TruncateEncoding(&input, 10, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, MaxLengthZero) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"});
  Encoding expected({}, {}, {});
  TruncateEncoding(&input, 0, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, TruncateRight) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"});
  Encoding expected({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"});
  TruncateEncoding(&input, 3, 2, TruncationDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(TruncateEncodingTest, TruncateLeft) {
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"});
  Encoding expected({3, 4, 5}, {0, 0, 0}, {"c", "d", "e"});
  TruncateEncoding(&input, 3, 2, TruncationDirection::kLeft);
  assertUtilsValues({input}, {expected});
}

TEST(TruncationTest, MaxLengthZero) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 0, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"})};
  std::vector<Encoding> expected = {Encoding()};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyLongestFirst) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 6, 2);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"}),
      Encoding({6, 7, 8, 9}, {0, 0, 0, 0}, {"f", "g", "h", "i"})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({6, 7, 8}, {0, 0, 0}, {"f", "g", "h"})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyOnlyFirst) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlyFirst, 6, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5, 6, 7}, {0, 0, 0, 0, 0, 0, 0},
               {"a", "b", "c", "d", "e", "f", "g"}),
      Encoding({8, 9, 10}, {0, 0, 0}, {"h", "i", "j"})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({8, 9, 10}, {0, 0, 0}, {"h", "i", "j"})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(TruncationTest, StrategyOnlySecond) {
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlySecond, 8, 3);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 0, 0},
               {"d", "e", "f", "g", "h", "i", "j"})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({4, 5, 6, 7, 8}, {0, 0, 0, 0, 0}, {"d", "e", "f", "g", "h"})};
  std::vector<Encoding> got = truncation.TruncateEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(PadEncodingTest, GreaterTargetLength) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"});
  Encoding expected({1, 2, 3, 0, 0}, {0, 0, 0, 1, 1},
                    {"a", "b", "c", "[PAD]", "[PAD]"});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(PadEncodingTest, PadLeft) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"});
  Encoding expected({0, 0, 1, 2, 3}, {1, 1, 0, 0, 0},
                    {"[PAD]", "[PAD]", "a", "b", "c"});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kLeft);
  assertUtilsValues({input}, {expected});
}

TEST(PadEncodingTest, PadRight) {
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"});
  Encoding expected({1, 2, 3, 0, 0}, {0, 0, 0, 1, 1},
                    {"a", "b", "c", "[PAD]", "[PAD]"});
  PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
  assertUtilsValues({input}, {expected});
}

TEST(PaddingTest, StrategyBatchLongest) {
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kBatchLongest, 0,
                  0, 0, 0, "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"}),
      Encoding({8, 9}, {0, 0}, {"h", "i"})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3, 0}, {0, 0, 0, 0}, {"a", "b", "c", "[PAD]"}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"}),
      Encoding({8, 9, 0, 0}, {0, 0, 0, 0}, {"h", "i", "[PAD]", "[PAD]"})};
  std::vector<Encoding> got = padding.PadEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(PaddingTest, StrategyFixed) {
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kFixed, 5, 0, 0, 0,
                  "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"}),
      Encoding({8, 9}, {0, 0}, {"h", "i"})};
  std::vector<Encoding> expected = {
      Encoding({1, 2, 3, 0, 0}, {0, 0, 0, 0, 0},
               {"a", "b", "c", "[PAD]", "[PAD]"}),
      Encoding({4, 5, 6, 7, 0}, {0, 0, 0, 0, 0}, {"d", "e", "f", "g", "[PAD]"}),
      Encoding({8, 9, 0, 0, 0}, {0, 0, 0, 0, 0},
               {"h", "i", "[PAD]", "[PAD]", "[PAD]"})};
  std::vector<Encoding> got = padding.PadEncodings(input);
  assertUtilsValues(got, expected);
}

TEST(FindMatchesTest, SplitsFound) {
  icu::UnicodeString input =
      icu::UnicodeString::fromUTF8("Hello, world! [MASK] never said Hello");
  std::vector<icu::UnicodeString> patterns = {
      icu::UnicodeString::fromUTF8("[MASK]"),
      icu::UnicodeString::fromUTF8("Hello")};
  std::vector<std::pair<int, int>> expected = {{0, 5}, {14, 20}, {32, 37}};
  std::vector<std::pair<int, int>> got =
      tokenizers::FindMatches(input, patterns);
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    ASSERT_EQ(got[i].first, expected[i].first);
    ASSERT_EQ(got[i].second, expected[i].second);
  }
}

TEST(FindMatchesTest, NoSplitsFound) {
  icu::UnicodeString input = icu::UnicodeString::fromUTF8("Hello, world!");
  std::vector<icu::UnicodeString> patterns = {
      icu::UnicodeString::fromUTF8("Goodbye"),
      icu::UnicodeString::fromUTF8("moon"), icu::UnicodeString::fromUTF8("! ")};
  std::vector<std::pair<int, int>> expected = {};
  std::vector<std::pair<int, int>> got =
      tokenizers::FindMatches(input, patterns);
  ASSERT_EQ(got.size(), expected.size());
}
