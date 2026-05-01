// Copyright 2025 Omkar Prabhu
#include "tokenizers/common.h"

#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

Encoding::Encoding() : ids({}), type_ids({}), tokens({}) {}

Encoding::Encoding(std::vector<int> ids, std::vector<int> type_ids,
                   std::vector<std::string> tokens)
    : ids(std::move(ids)),
      type_ids(std::move(type_ids)),
      tokens(std::move(tokens)) {}

Token::Token() : value(""), id(0), is_continuing_subword(false) {}

Token::Token(std::string value, int id, bool is_continuing_subword = false)
    : value(std::move(value)),
      id(id),
      is_continuing_subword(is_continuing_subword) {}

} // namespace tokenizers
