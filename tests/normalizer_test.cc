// Copyright 2025 Omkar Prabhu
#include "tokenizers/normalizer.h"

#include <gtest/gtest.h>

#include <string>

using tokenizers::normalizers::BertNormalizer;
using tokenizers::normalizers::isChineseChar;
using tokenizers::normalizers::isControl;
using tokenizers::normalizers::isWhitespace;
using tokenizers::normalizers::NFCNormalizer;
using tokenizers::normalizers::Normalizer;
using tokenizers::normalizers::NormalizerResult;

void assertNormalizerValues(const NormalizerResult& got,
                            const NormalizerResult& expected) {
  std::string got_str, expected_str;
  got.normalized.toUTF8String(got_str);
  expected.normalized.toUTF8String(expected_str);
  ASSERT_EQ(got_str, expected_str);
}

TEST(NormalizerTest, EmptyInput) {
  Normalizer normalizer;
  NormalizerResult input = NormalizerResult(u8"");
  NormalizerResult expected_result = NormalizerResult(u8"");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(NFCNormalizerTest, Normalization) {
  NFCNormalizer normalizer;
  NormalizerResult input = NormalizerResult(u8"Cafe\u0301");
  NormalizerResult expected_result = NormalizerResult(u8"Café");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, NoNormalization) {
  BertNormalizer normalizer(false, false, false, false);
  NormalizerResult input = NormalizerResult(u8"Hello, World!");
  NormalizerResult expected_result = NormalizerResult(u8"Hello, World!");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, CleanText) {
  BertNormalizer normalizer(true, false, false, false);
  NormalizerResult input =
      NormalizerResult(u8"He\u200Bl\uFFFDl\to\n \rWo\tr\nl\rd");
  NormalizerResult expected_result = NormalizerResult(u8"Hell o   Wo r l d");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, HandleChineseChars) {
  BertNormalizer normalizer(false, true, false, false);
  NormalizerResult input = NormalizerResult(u8"习近平访问了纽约。");
  NormalizerResult expected_result =
      NormalizerResult(u8" 习  近  平  访  问  了  纽  约 。");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, StripAccents) {
  BertNormalizer normalizer(false, false, true, false);
  NormalizerResult input = NormalizerResult(u8"café naïve são élève");
  NormalizerResult expected_result = NormalizerResult(u8"cafe naive sao eleve");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, Lowercase) {
  BertNormalizer normalizer(false, false, false, true);
  NormalizerResult input = NormalizerResult(u8"HELLO WORLD");
  NormalizerResult expected_result = NormalizerResult(u8"hello world");
  assertNormalizerValues(normalizer.Normalize(input), expected_result);
}

TEST(BertNormalizerTest, AllOptions) {
  BertNormalizer normalizer(true, true, true, true);
  NormalizerResult input = NormalizerResult(u8"Café 中文");
  NormalizerResult expected_result = NormalizerResult(u8"cafe  中  文 ");
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
