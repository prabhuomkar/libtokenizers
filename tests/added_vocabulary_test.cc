
// Copyright 2025 Omkar Prabhu
#include "tokenizers/added_vocabulary.h"

#include <gtest/gtest.h>
#include <unicode/unistr.h>

#include <string>
#include <vector>

#include "tokenizers/normalizer.h"

using tokenizers::AddedToken;
using tokenizers::AddedVocabulary;
using tokenizers::normalizers::NormalizerResult;

void assertAddedVocabularyValues(
    const std::vector<NormalizerResult>& got,
    const std::vector<NormalizerResult>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    std::string got_str, expected_str;
    got[i].normalized.toUTF8String(got_str);
    expected[i].normalized.toUTF8String(expected_str);
    ASSERT_EQ(got_str, expected_str);
    ASSERT_EQ(got[i].offsets.size(), expected[i].offsets.size());
    for (int j = 0; j < got[i].offsets.size(); j++) {
      ASSERT_EQ(got[i].offsets[j], expected[i].offsets[j]);
    }
    ASSERT_EQ(got[i].pre_normalized, expected[i].pre_normalized);
  }
}

std::vector<std::pair<int, int>> getVec(int start, int end) {
  std::vector<std::pair<int, int>> vec;
  for (int i = start; i < end; i++) {
    vec.emplace_back(i, i + 1);
  }
  return vec;
}

void assertSplits(const std::vector<std::pair<int, int>>& got,
                  const std::vector<std::pair<int, int>>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    ASSERT_EQ(got[i], expected[i]);
  }
}

TEST(AddedVocabularyIsSpecialToken, True) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[UNK]", false, false, false, false, true)});
  ASSERT_EQ(added_vocabulary.IsSpecialToken("[UNK]"), true);
}

TEST(AddedVocabularyIsSpecialToken, False) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[UNK]", false, false, false, false, true)});
  ASSERT_EQ(added_vocabulary.IsSpecialToken("[CLS]"), false);
}

TEST(AddedVocabularyFindSplits, SpecialToken) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[MASK]", false, false, false, false, true)});
  std::vector<NormalizerResult> expected_result = {
      NormalizerResult(icu::UnicodeString("Capital of India is "),
                       getVec(0, 20)),
      NormalizerResult(icu::UnicodeString("[MASK]"), getVec(20, 26), true)};
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  std::vector<NormalizerResult> got_result = added_vocabulary.FindSplits(input);
  assertAddedVocabularyValues(got_result, expected_result);
}

TEST(AddedVocabularyFindSplits, SingleWord) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", true, false, false, false, false)});
  std::vector<NormalizerResult> expected_result = {
      NormalizerResult(icu::UnicodeString("Capital of "), getVec(0, 11)),
      NormalizerResult(icu::UnicodeString("India"), getVec(11, 16)),
      NormalizerResult(icu::UnicodeString(" is [MASK]"), getVec(16, 26))};
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  std::vector<NormalizerResult> got_result = added_vocabulary.FindSplits(input);
  assertAddedVocabularyValues(got_result, expected_result);

  added_vocabulary = AddedVocabulary(
      {AddedToken(0, "India", true, false, false, false, false)});
  expected_result = {NormalizerResult(
      icu::UnicodeString("Capital of MyIndia is [MASK]"), getVec(0, 28))};
  got_result = added_vocabulary.FindSplits(
      NormalizerResult(icu::UnicodeString("Capital of MyIndia is [MASK]")));
  assertAddedVocabularyValues(got_result, expected_result);

  expected_result = {NormalizerResult(
      icu::UnicodeString("Capital of Indias is [MASK]"), getVec(0, 27))};
  got_result = added_vocabulary.FindSplits(
      NormalizerResult(icu::UnicodeString("Capital of Indias is [MASK]")));
  assertAddedVocabularyValues(got_result, expected_result);
}

TEST(AddedVocabularyFindSplits, LStrip) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, true, false, false, false)});
  std::vector<NormalizerResult> expected_result = {
      NormalizerResult(icu::UnicodeString("Capital of"), getVec(0, 10)),
      NormalizerResult(icu::UnicodeString(" India"), getVec(10, 16)),
      NormalizerResult(icu::UnicodeString(" is [MASK]"), getVec(16, 26))};
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  std::vector<NormalizerResult> got_result = added_vocabulary.FindSplits(input);
  assertAddedVocabularyValues(got_result, expected_result);
}

TEST(AddedVocabularyFindSplits, RStrip) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, false, true, false, false)});
  std::vector<NormalizerResult> expected_result = {
      NormalizerResult(icu::UnicodeString("Capital of "), getVec(0, 11)),
      NormalizerResult(icu::UnicodeString("India "), getVec(11, 17)),
      NormalizerResult(icu::UnicodeString("is [MASK]"), getVec(17, 26))};
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  std::vector<NormalizerResult> got_result = added_vocabulary.FindSplits(input);
  assertAddedVocabularyValues(got_result, expected_result);
}

TEST(AddedVocabularyFindSplits, LStripAndRStrip) {
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, true, true, false, false)});
  std::vector<NormalizerResult> expected_result = {
      NormalizerResult(icu::UnicodeString("Capital of"), getVec(0, 10)),
      NormalizerResult(icu::UnicodeString(" India "), getVec(10, 17)),
      NormalizerResult(icu::UnicodeString("is [MASK]"), getVec(17, 26))};
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  std::vector<NormalizerResult> got_result = added_vocabulary.FindSplits(input);
  assertAddedVocabularyValues(got_result, expected_result);
}
