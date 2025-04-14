// Copyright 2025 Omkar Prabhu
#include "tokenizers/common.h"

#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

Encoding::Encoding()
    : ids({}),
      type_ids({}),
      tokens({}),
      offsets({}),
      word_ids({}),
      special_tokens_mask({}),
      attention_mask({}) {}

Encoding::Encoding(const std::vector<int>& ids,
                   const std::vector<int>& type_ids,
                   const std::vector<std::string>& tokens,
                   const std::vector<std::pair<int, int>>& offsets,
                   const std::vector<std::optional<int>>& word_ids,
                   const std::vector<int>& special_tokens_mask,
                   const std::vector<int>& attention_mask)
    : ids(std::move(ids)),
      type_ids(std::move(type_ids)),
      tokens(std::move(tokens)),
      offsets(std::move(offsets)),
      word_ids(std::move(word_ids)),
      special_tokens_mask(std::move(special_tokens_mask)),
      attention_mask(std::move(attention_mask)) {}

Token::Token()
    : value(""), id(0), offsets({0, 0}), is_continuing_subword(false) {}

Token::Token(const std::string& value, int id,
             const std::pair<int, int>& offsets,
             bool is_continuing_subword = false)
    : value(value),
      id(id),
      offsets(offsets),
      is_continuing_subword(is_continuing_subword) {}

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

} // namespace tokenizers
