// Copyright 2025 Omkar Prabhu
#include "tokenizers/model.h"

#include <unicode/unistr.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenizers {

namespace models {

Model::Model() {}

std::vector<Token> Model::Tokenize(const icu::UnicodeString& input,
                                   const std::pair<int, int>& offset) {
  return {};
}

std::vector<Token> Model::Tokenize(const icu::UnicodeString& input) {
  return {};
}

std::vector<Token> Model::TokenizeString(const std::string& input) {
  return {};
}

std::optional<std::string> Model::IdToToken(int id) { return std::nullopt; }

std::optional<int> Model::TokenToId(const std::string& token) {
  return std::nullopt;
}

WordPiece::WordPiece(const std::unordered_map<std::string, int>& vocab,
                     const std::string& unk_token,
                     const std::string& continuing_subword_prefix,
                     int max_input_chars_per_word)
    : vocab_(vocab),
      unk_token_(unk_token),
      continuing_subword_prefix_(continuing_subword_prefix),
      max_input_chars_per_word_(max_input_chars_per_word) {
  for (const auto& pair : vocab_) {
    rvocab_[pair.second] = pair.first;
  }
}

std::vector<Token> WordPiece::Tokenize(const icu::UnicodeString& input,
                                       const std::pair<int, int>& offset) {
  int input_len = input.countChar32();
  if (input_len > max_input_chars_per_word_) {
    return {Token(unk_token_, vocab_.at(unk_token_), offset, false)};
  }

  std::vector<Token> tokens;
  int start = 0;
  bool is_bad = false;

  while (start < input_len) {
    int end = input_len;
    bool found = false;

    while (start < end) {
      icu::UnicodeString input_substr = input.tempSubStringBetween(start, end);
      std::string input_substr_str;
      input_substr.toUTF8String(input_substr_str);

      if (start > 0) {
        input_substr_str = continuing_subword_prefix_ + input_substr_str;
      }
      if (vocab_.find(input_substr_str) != vocab_.end()) {
        tokens.emplace_back(Token(input_substr_str, vocab_.at(input_substr_str),
                                  {offset.first + start, offset.first + end},
                                  start > 0 ? true : false));
        found = true;
        break;
      }

      end -= 1;
    }

    if (!found) {
      is_bad = true;
      break;
    }
    start = end;
  }

  if (is_bad) {
    tokens.emplace_back(Token(unk_token_, vocab_.at(unk_token_),
                              {offset.first + start, offset.first + input_len},
                              false));
  }

  return tokens;
}

std::vector<Token> WordPiece::Tokenize(const icu::UnicodeString& input) {
  return Tokenize(input, {0, input.countChar32()});
}

std::vector<Token> WordPiece::TokenizeString(const std::string& input) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  return Tokenize(unicode_input);
}

std::optional<std::string> WordPiece::IdToToken(int id) {
  auto it = rvocab_.find(id);
  if (it != rvocab_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<int> WordPiece::TokenToId(const std::string& token) {
  auto it = vocab_.find(token);
  if (it != vocab_.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace models

} // namespace tokenizers
