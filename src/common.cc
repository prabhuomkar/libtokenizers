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

Encoding::Encoding(std::vector<int> ids,
                   std::vector<int> type_ids,
                   std::vector<std::string> tokens,
                   std::vector<std::pair<int, int>> offsets,
                   std::vector<std::optional<int>> word_ids,
                   std::vector<int> special_tokens_mask,
                   std::vector<int> attention_mask)
    : ids(std::move(ids)),
      type_ids(std::move(type_ids)),
      tokens(std::move(tokens)),
      offsets(std::move(offsets)),
      word_ids(std::move(word_ids)),
      special_tokens_mask(std::move(special_tokens_mask)),
      attention_mask(std::move(attention_mask)) {}

Token::Token()
    : value(""), id(0), offsets({0, 0}), is_continuing_subword(false) {}

Token::Token(std::string value, int id,
            std::pair<int, int> offsets,
             bool is_continuing_subword = false)
    : value(std::move(value)),
      id(id),
      offsets(offsets),
      is_continuing_subword(is_continuing_subword) {}

} // namespace tokenizers
