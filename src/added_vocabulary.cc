// Copyright 2025 Omkar Prabhu
#include "tokenizers/added_vocabulary.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {

AddedToken::AddedToken()
    : id(0),
      content(""),
      single_word(false),
      lstrip(false),
      rstrip(false),
      normalized(false),
      special_token(false) {}

AddedToken::AddedToken(int id, const std::string& content, bool single_word,
                       bool lstrip, bool rstrip, bool normalized,
                       bool special_token)
    : id(id),
      content(content),
      single_word(single_word),
      lstrip(lstrip),
      rstrip(rstrip),
      normalized(normalized),
      special_token(special_token) {}

AddedVocabulary::AddedVocabulary() {}

AddedVocabulary::AddedVocabulary(const std::vector<AddedToken>& tokens)
    : tokens_(tokens) {
  for (const auto& token : tokens_) {
    added_tokens_map_[token.content] = token.id;
    added_tokens_map_r_[token.id] = token.content;
    if (token.special_token) {
      special_tokens_.insert(token.content);
    }
  }
}

bool AddedVocabulary::IsSpecialToken(const std::string& token) {
  return special_tokens_.find(token) != special_tokens_.end();
}

} // namespace tokenizers
