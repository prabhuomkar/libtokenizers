// Copyright 2025 Omkar Prabhu
#pragma once

#include <unicode/unistr.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

namespace tokenizers {

namespace models {

class Model {
 public:
  Model();
  virtual std::vector<Token> Tokenize(const icu::UnicodeString& input,
                                      const std::pair<int, int>& offset);
  virtual std::vector<Token> Tokenize(const icu::UnicodeString& input);
  virtual std::vector<Token> TokenizeString(const std::string& input);
};

// WordPiece
class WordPiece : public Model {
 public:
  explicit WordPiece(const std::unordered_map<std::string, int>& vocab,
                     const std::string& unk_token = "[UNK]",
                     const std::string& continuing_subword_prefix = "##",
                     int max_input_chars_per_word = 100);
  std::vector<Token> Tokenize(const icu::UnicodeString& input,
                              const std::pair<int, int>& offset) override;
  std::vector<Token> Tokenize(const icu::UnicodeString& input) override;
  std::vector<Token> TokenizeString(const std::string& input) override;

 private:
  std::unordered_map<std::string, int> vocab_;
  std::string unk_token_;
  std::string continuing_subword_prefix_;
  int max_input_chars_per_word_;
};

} // namespace models

} // namespace tokenizers
