// Copyright 2025 Omkar Prabhu
#include "tokenizers/pre_tokenizer.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

using tokenizers::pre_tokenizers::BertPreTokenizer;
using tokenizers::pre_tokenizers::PreTokenizer;
using tokenizers::pre_tokenizers::PreTokenizerResult;
using tokenizers::pre_tokenizers::SplitDelimiterBehavior;

void assertPreTokenizerValues(const PreTokenizerResult& got,
                              const PreTokenizerResult& expected) {
  ASSERT_EQ(got.pre_tokenized.size(), expected.pre_tokenized.size());
  ASSERT_EQ(got.offsets.size(), expected.offsets.size());
  for (int i = 0; i < got.pre_tokenized.size(); i++) {
    std::string got_str, expected_str;
    got.pre_tokenized[i].toUTF8String(got_str);
    expected.pre_tokenized[i].toUTF8String(expected_str);
    ASSERT_EQ(got_str, expected_str);
  }
  for (int i = 0; i < got.offsets.size(); i++) {
    ASSERT_EQ(got.offsets[i], expected.offsets[i]);
  }
}

TEST(PreTokenizerTest, SplitRemoved) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kRemoved);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the"),
                          icu::UnicodeString::fromUTF8(u8"final"),
                          icu::UnicodeString::fromUTF8(u8"countdown")},
                         {{0, 3}, {4, 9}, {11, 20}});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitIsolated) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kIsolated);
  PreTokenizerResult expected = PreTokenizerResult(
      {icu::UnicodeString::fromUTF8(u8"the"),
       icu::UnicodeString::fromUTF8(u8"-"),
       icu::UnicodeString::fromUTF8(u8"final"),
       icu::UnicodeString::fromUTF8(u8"-"), icu::UnicodeString::fromUTF8(u8"-"),
       icu::UnicodeString::fromUTF8(u8"countdown")},
      {{0, 3}, {3, 4}, {4, 9}, {9, 10}, {10, 11}, {11, 20}});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitMergedWithPrevious) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kMergedWithPrevious);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the-"),
                          icu::UnicodeString::fromUTF8(u8"final-"),
                          icu::UnicodeString::fromUTF8(u8"-"),
                          icu::UnicodeString::fromUTF8(u8"countdown")},
                         {{0, 4}, {4, 10}, {10, 11}, {11, 20}});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitMergedWithNext) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kMergedWithNext);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the"),
                          icu::UnicodeString::fromUTF8(u8"-final"),
                          icu::UnicodeString::fromUTF8(u8"-"),
                          icu::UnicodeString::fromUTF8(u8"-countdown")},
                         {{0, 3}, {3, 9}, {9, 10}, {10, 20}});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, EmptyInput) {
  PreTokenizer pre_tokenizer;
  PreTokenizerResult input =
      PreTokenizerResult(icu::UnicodeString::fromUTF8(u8""));
  PreTokenizerResult expected_result = input;
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}

TEST(BertPreTokenizerTest, WhitespaceChars) {
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"Hey friend!     How are you?!?"));
  PreTokenizerResult expected_result = PreTokenizerResult(
      {icu::UnicodeString::fromUTF8(u8"Hey"),
       icu::UnicodeString::fromUTF8(u8"friend"),
       icu::UnicodeString::fromUTF8(u8"!"),
       icu::UnicodeString::fromUTF8(u8"How"),
       icu::UnicodeString::fromUTF8(u8"are"),
       icu::UnicodeString::fromUTF8(u8"you"),
       icu::UnicodeString::fromUTF8(u8"?"), icu::UnicodeString::fromUTF8(u8"!"),
       icu::UnicodeString::fromUTF8(u8"?")},
      {{0, 3},
       {4, 10},
       {10, 11},
       {16, 19},
       {20, 23},
       {24, 27},
       {27, 28},
       {28, 29},
       {29, 30}});
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}

TEST(BertPreTokenizerTest, ChineseChars) {
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8" 野  口  里  佳  Noguchi Rika"));
  PreTokenizerResult expected_result = PreTokenizerResult(
      {icu::UnicodeString::fromUTF8(u8"野"),
       icu::UnicodeString::fromUTF8(u8"口"),
       icu::UnicodeString::fromUTF8(u8"里"),
       icu::UnicodeString::fromUTF8(u8"佳"),
       icu::UnicodeString::fromUTF8(u8"Noguchi"),
       icu::UnicodeString::fromUTF8(u8"Rika")},
      {{1, 2}, {4, 5}, {7, 8}, {10, 11}, {13, 20}, {21, 25}});
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}

TEST(BertPreTokenizerTest, AllOptions) {
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(icu::UnicodeString::fromUTF8(
      u8"Hey friend!  野  口  里  佳  Noguchi Rika"));
  PreTokenizerResult expected_result =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"Hey"),
                          icu::UnicodeString::fromUTF8(u8"friend"),
                          icu::UnicodeString::fromUTF8(u8"!"),
                          icu::UnicodeString::fromUTF8(u8"野"),
                          icu::UnicodeString::fromUTF8(u8"口"),
                          icu::UnicodeString::fromUTF8(u8"里"),
                          icu::UnicodeString::fromUTF8(u8"佳"),
                          icu::UnicodeString::fromUTF8(u8"Noguchi"),
                          icu::UnicodeString::fromUTF8(u8"Rika")},
                         {{0, 3},
                          {4, 10},
                          {10, 11},
                          {13, 14},
                          {16, 17},
                          {19, 20},
                          {22, 23},
                          {25, 32},
                          {33, 37}});
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}
