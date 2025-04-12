// Copyright 2025 Omkar Prabhu
#include "tokenizers/normalizer.h"

#include <gtest/gtest.h>

#include <string>

using tokenizers::normalizers::BertNormalizer;
using tokenizers::normalizers::isChineseChar;
using tokenizers::normalizers::isControl;
using tokenizers::normalizers::isWhitespace;
using tokenizers::normalizers::Normalizer;
using tokenizers::normalizers::NormalizerResult;

void assertNormalizerValues(const NormalizerResult& got,
                            const NormalizerResult& expected) {
  std::string got_str, expected_str;
  got.normalized.toUTF8String(got_str);
  expected.normalized.toUTF8String(expected_str);
  ASSERT_EQ(got_str, expected_str);
  ASSERT_EQ(got.offsets.size(), expected.offsets.size());
  for (int i = 0; i < got.offsets.size(); i++) {
    ASSERT_EQ(got.offsets[i], expected.offsets[i]);
  }
}

TEST(NormalizerTest, EmptyInput) {
  Normalizer normalizer;
  NormalizerResult input = NormalizerResult(u8"");
  NormalizerResult expected_result = NormalizerResult(u8"", {});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, NoNormalization) {
  BertNormalizer normalizer(false, false, false, false);
  NormalizerResult input = NormalizerResult(u8"Hello, World!");
  NormalizerResult expected_result =
      NormalizerResult(u8"Hello, World!", {{0, 1},
                                           {1, 2},
                                           {2, 3},
                                           {3, 4},
                                           {4, 5},
                                           {5, 6},
                                           {6, 7},
                                           {7, 8},
                                           {8, 9},
                                           {9, 10},
                                           {10, 11},
                                           {11, 12},
                                           {12, 13}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, CleanText) {
  BertNormalizer normalizer(true, false, false, false);
  NormalizerResult input =
      NormalizerResult(u8"He\u200Bl\uFFFDl\to\n \rWo\tr\nl\rd");
  NormalizerResult expected_result =
      NormalizerResult(u8"Hell o   Wo r l d", {{0, 1},
                                               {1, 2},
                                               {3, 4},
                                               {5, 6},
                                               {6, 7},
                                               {7, 8},
                                               {8, 9},
                                               {9, 10},
                                               {10, 11},
                                               {11, 12},
                                               {12, 13},
                                               {13, 14},
                                               {14, 15},
                                               {15, 16},
                                               {16, 17},
                                               {17, 18},
                                               {18, 19}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, HandleChineseChars) {
  BertNormalizer normalizer(false, true, false, false);
  NormalizerResult input = NormalizerResult(u8"习近平访问了纽约。");
  NormalizerResult expected_result = NormalizerResult(
      u8" 习  近  平  访  问  了  纽  约 。",
      {{0, 1}, {0, 1}, {0, 1}, {1, 2}, {1, 2}, {1, 2}, {2, 3}, {2, 3}, {2, 3},
       {3, 4}, {3, 4}, {3, 4}, {4, 5}, {4, 5}, {4, 5}, {5, 6}, {5, 6}, {5, 6},
       {6, 7}, {6, 7}, {6, 7}, {7, 8}, {7, 8}, {7, 8}, {8, 9}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, StripAccents) {
  BertNormalizer normalizer(false, false, true, false);
  NormalizerResult input = NormalizerResult(u8"café naïve são élève");
  NormalizerResult expected_result = NormalizerResult(
      u8"cafe naive sao eleve",
      {{0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},   {6, 7},
       {7, 8},   {8, 9},   {9, 10},  {10, 11}, {11, 12}, {12, 13}, {13, 14},
       {14, 15}, {15, 16}, {16, 17}, {17, 18}, {18, 19}, {19, 20}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, Lowercase) {
  BertNormalizer normalizer(false, false, false, true);
  NormalizerResult input = NormalizerResult(u8"HELLO WORLD");
  NormalizerResult expected_result =
      NormalizerResult(u8"hello world", {{0, 1},
                                         {1, 2},
                                         {2, 3},
                                         {3, 4},
                                         {4, 5},
                                         {5, 6},
                                         {6, 7},
                                         {7, 8},
                                         {8, 9},
                                         {9, 10},
                                         {10, 11}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, AllOptions) {
  BertNormalizer normalizer(true, true, true, true);
  NormalizerResult input = NormalizerResult(u8"Café 中文");
  NormalizerResult expected_result =
      NormalizerResult(u8"cafe  中  文 ", {{0, 1},
                                           {1, 2},
                                           {2, 3},
                                           {3, 4},
                                           {4, 5},
                                           {5, 6},
                                           {5, 6},
                                           {5, 6},
                                           {6, 7},
                                           {6, 7},
                                           {6, 7}});
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(NormalizerHelpersTest, IsControl) {
  EXPECT_TRUE(isControl(U'\x00'));
  EXPECT_TRUE(isControl(U'\x1F'));
  EXPECT_TRUE(isControl(U'\x7F'));
  EXPECT_FALSE(isControl(U' '));
}

TEST(NormalizerHelpersTest, IsWhitespace) {
  EXPECT_TRUE(isWhitespace(U' '));
  EXPECT_TRUE(isWhitespace(U'\t'));
  EXPECT_FALSE(isWhitespace(U'A'));
}

TEST(NormalizerHelpersTest, IsChineseChar) {
  EXPECT_TRUE(isChineseChar(U'中'));
  EXPECT_FALSE(isChineseChar(U'A'));
}
