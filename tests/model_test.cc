// Copyright 2025 Omkar Prabhu
#include "tokenizers/model.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "tokenizers/common.h"

using tokenizers::Token;
using tokenizers::models::Model;
using tokenizers::models::WordPiece;

void assertModelValues(const std::vector<Token>& got,
                       const std::vector<Token>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (int i = 0; i < got.size(); i++) {
    ASSERT_EQ(got[i].value, expected[i].value);
    ASSERT_EQ(got[i].id, expected[i].id);
    ASSERT_EQ(got[i].is_continuing_subword, expected[i].is_continuing_subword);
  }
}

TEST(ModelTest, EmptyInput) {
  Model model;
  assertModelValues(model.TokenizeString(u8""), std::vector<Token>{});
}

TEST(WordPieceTest, IsBad) {
  WordPiece model({{u8"[UNK]", 1}}, u8"[UNK]", u8"##", 100);
  std::string input = u8"tokenization is important!";
  std::vector<Token> expected_tokens = {Token(u8"[UNK]", 1, false)};
  std::vector<Token> got_tokens = model.TokenizeString(input);
  assertModelValues(got_tokens, expected_tokens);
}

TEST(WordPieceTest, IsFound) {
  WordPiece model(
      {{u8"[UNK]", 1}, {u8"token", 2}, {u8"##izat", 3}, {u8"##ion", 4}},
      u8"[UNK]", u8"##", 100);
  std::string input = u8"tokenization";
  std::vector<Token> expected_tokens = {Token(u8"token", 2, false),
                                        Token(u8"##izat", 3, true),
                                        Token(u8"##ion", 4, true)};
  std::vector<Token> got_tokens = model.TokenizeString(input);
  assertModelValues(got_tokens, expected_tokens);
}

TEST(WordPieceTest, UnkToken) {
  WordPiece model({{u8"hello", 1}, {u8"world", 2}, {u8"[UNK]", 3}}, u8"[UNK]",
                  u8"##", 100);
  std::string input = u8"helloqwerty";
  std::vector<Token> expected_tokens = {Token(u8"hello", 1, false),
                                        Token(u8"[UNK]", 3, false)};
  std::vector<Token> got_tokens = model.TokenizeString(input);
  assertModelValues(got_tokens, expected_tokens);
}

TEST(WordPieceTest, MaxInputCharsPerWord) {
  WordPiece model({{u8"[UNK]", 1}}, u8"[UNK]", u8"##", 5);
  std::string input = u8"tokenization is important!";
  std::vector<Token> expected_tokens = {Token(u8"[UNK]", 1, false)};
  std::vector<Token> got_tokens = model.TokenizeString(input);
  assertModelValues(got_tokens, expected_tokens);
}
