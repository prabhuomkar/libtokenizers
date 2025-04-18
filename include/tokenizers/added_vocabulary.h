// Copyright 2025 Omkar Prabhu
#pragma once

#include <unicode/unistr.h>

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers/normalizer.h"

using tokenizers::normalizers::NormalizerResult;

namespace tokenizers {

class AddedToken {
 public:
  AddedToken();
  AddedToken(int id, const std::string &content, bool single_word,
             bool lstrip = false, bool rstrip = false, bool normalized = false,
             bool special_token = true);
  int id;
  std::string content;
  bool single_word;
  bool lstrip;
  bool rstrip;
  bool normalized;
  bool special_token;
};

class AddedVocabulary {
 public:
  AddedVocabulary();
  explicit AddedVocabulary(const std::vector<AddedToken> &tokens);
  bool IsSpecialToken(const std::string &token);
  std::vector<NormalizerResult> FindSplits(const NormalizerResult &input);

 private:
  std::unordered_map<std::string, int> added_tokens_map_;
  std::unordered_map<int, std::string> added_tokens_map_r_;
  std::set<std::string> special_tokens_;
  std::unordered_map<int, AddedToken> tokens_;
  std::vector<icu::UnicodeString> patterns_;
};

} // namespace tokenizers
