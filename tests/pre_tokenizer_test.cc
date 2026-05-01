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
  for (int i = 0; i < got.pre_tokenized.size(); i++) {
    std::string got_str, expected_str;
    got.pre_tokenized[i].toUTF8String(got_str);
    expected.pre_tokenized[i].toUTF8String(expected_str);
    ASSERT_EQ(got_str, expected_str);
  }
}

TEST(PreTokenizerTest, SplitRemoved) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = tokenizers::pre_tokenizers::split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kRemoved);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the"),
                          icu::UnicodeString::fromUTF8(u8"final"),
                          icu::UnicodeString::fromUTF8(u8"countdown")});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitIsolated) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = tokenizers::pre_tokenizers::split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kIsolated);
  PreTokenizerResult expected = PreTokenizerResult(
      {icu::UnicodeString::fromUTF8(u8"the"),
       icu::UnicodeString::fromUTF8(u8"-"),
       icu::UnicodeString::fromUTF8(u8"final"),
       icu::UnicodeString::fromUTF8(u8"-"), icu::UnicodeString::fromUTF8(u8"-"),
       icu::UnicodeString::fromUTF8(u8"countdown")});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitMergedWithPrevious) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = tokenizers::pre_tokenizers::split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kMergedWithPrevious);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the-"),
                          icu::UnicodeString::fromUTF8(u8"final-"),
                          icu::UnicodeString::fromUTF8(u8"-"),
                          icu::UnicodeString::fromUTF8(u8"countdown")});
  assertPreTokenizerValues(result, expected);
}

TEST(PreTokenizerTest, SplitMergedWithNext) {
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  PreTokenizerResult result = tokenizers::pre_tokenizers::split(
      input, [](UChar32 c) { return c == '-'; },
      SplitDelimiterBehavior::kMergedWithNext);
  PreTokenizerResult expected =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"the"),
                          icu::UnicodeString::fromUTF8(u8"-final"),
                          icu::UnicodeString::fromUTF8(u8"-"),
                          icu::UnicodeString::fromUTF8(u8"-countdown")});
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
       icu::UnicodeString::fromUTF8(u8"?")});
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}

TEST(BertPreTokenizerTest, ChineseChars) {
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8" 野  口  里  佳  Noguchi Rika"));
  PreTokenizerResult expected_result =
      PreTokenizerResult({icu::UnicodeString::fromUTF8(u8"野"),
                          icu::UnicodeString::fromUTF8(u8"口"),
                          icu::UnicodeString::fromUTF8(u8"里"),
                          icu::UnicodeString::fromUTF8(u8"佳"),
                          icu::UnicodeString::fromUTF8(u8"Noguchi"),
                          icu::UnicodeString::fromUTF8(u8"Rika")});
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
                          icu::UnicodeString::fromUTF8(u8"Rika")});
  assertPreTokenizerValues(pre_tokenizer.PreTokenize(input), expected_result);
}
