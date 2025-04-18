
// Copyright 2025 Omkar Prabhu
#include "tokenizers/decoder.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using tokenizers::decoders::Decoder;
using tokenizers::decoders::WordPieceDecoder;

void assertDecoderValues(const std::vector<std::string>& got,
                         const std::vector<std::string>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    ASSERT_EQ(got[i], expected[i]);
  }
}

TEST(DecoderTest, EmptyInput) {
  Decoder decoder;
  assertDecoderValues(decoder.DecodeChain({}), std::vector<std::string>{});
}

TEST(WordPieceDecoderTest, AllOptions) {
  WordPieceDecoder decoder;
  std::vector<std::string> input = {"##uelo", "Ara", "##új",
                                    "##o",    "No",  "##guera"};
  std::vector<std::string> expected_tokens = {"##uelo", " Ara", "új",
                                              "o",      " No",  "guera"};
  std::vector<std::string> got_tokens = decoder.DecodeChain(input);
  assertDecoderValues(got_tokens, expected_tokens);
}
