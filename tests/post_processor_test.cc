// Copyright 2025 Omkar Prabhu
#include "tokenizers/post_processor.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers/common.h"

using tokenizers::Encoding;
using tokenizers::post_processors::PostProcessor;
using tokenizers::post_processors::TemplateProcessing;
using tokenizers::post_processors::TemplateProcessor;

void assertPostProcessorValues(const std::vector<Encoding>& got,
                               const std::vector<Encoding>& expected) {
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

TEST(PostProcessorTest, EmptyInput) {
  PostProcessor post_processor;
  assertPostProcessorValues(post_processor.ProcessEncodings({}),
                            std::vector<Encoding>{});
}

TEST(TemplateProcessingTest, Single) {
  TemplateProcessing post_processor(
      {TemplateProcessor("SpecialToken", 0, "[CLS]"),
       TemplateProcessor("Sequence", 0, "A"),
       TemplateProcessor("SpecialToken", 0, "[SEP]")},
      {
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      },
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  std::vector<Encoding> input = {
      Encoding({200, 201}, {0, 0}, {"hello", "world"})};
  std::vector<Encoding> expected_encodings = {
      Encoding({100}, {0}, {"[CLS]"}),
      Encoding({200, 201}, {0, 0}, {"hello", "world"}),
      Encoding({101}, {0}, {"[SEP]"})};
  std::vector<Encoding> got_encodings = post_processor.ProcessEncodings(input);
  assertPostProcessorValues(got_encodings, expected_encodings);
}

TEST(TemplateProcessingTest, Pair) {
  TemplateProcessing post_processor(
      {TemplateProcessor("SpecialToken", 0, "[CLS]"),
       TemplateProcessor("Sequence", 0, "A"),
       TemplateProcessor("SpecialToken", 0, "[SEP]")},
      {
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      },
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  std::vector<Encoding> input = {
      Encoding({200, 201}, {0, 0}, {"hello", "world"}),
      Encoding({300, 301}, {1, 1}, {"martin", "garrix"})};
  std::vector<Encoding> expected_encodings = {
      Encoding({100}, {0}, {"[CLS]"}),
      Encoding({200, 201}, {0, 0}, {"hello", "world"}),
      Encoding({101}, {0}, {"[SEP]"}),
      Encoding({300, 301}, {1, 1}, {"martin", "garrix"}),
      Encoding({101}, {1}, {"[SEP]"})};
  std::vector<Encoding> got_encodings = post_processor.ProcessEncodings(input);
  assertPostProcessorValues(got_encodings, expected_encodings);
}
