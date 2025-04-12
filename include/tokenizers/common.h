// Copyright 2025 Omkar Prabhu
#pragma once

#include <simdjson.h>

#include <codecvt>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenizers {

class Encoding {
 public:
  Encoding();
  Encoding(const std::vector<int> &ids, const std::vector<int> &type_ids,
           const std::vector<std::string> &tokens,
           const std::vector<std::pair<int, int>> &offsets,
           const std::vector<std::optional<int>> &word_ids,
           const std::vector<int> &special_tokens_mask,
           const std::vector<int> &attention_mask);

  std::vector<int> ids;
  std::vector<int> type_ids;
  std::vector<std::string> tokens;
  std::vector<std::pair<int, int>> offsets;
  std::vector<std::optional<int>> word_ids;
  std::vector<int> special_tokens_mask;
  std::vector<int> attention_mask;
  std::vector<Encoding> overflowing;
};

class Token {
 public:
  Token();
  Token(const std::string &value, int id, const std::pair<int, int> &offsets,
        bool is_continuing_subword);
  std::string value;
  int id;
  std::pair<int, int> offsets;
  bool is_continuing_subword;
};

inline std::string get_string_or_default(simdjson::ondemand::value &&val,
                                         std::string_view key,
                                         std::string_view def = "") {
  auto result = val[key].get_string();
  return result.error() == simdjson::SUCCESS ? std::string(result.value())
                                             : std::string(def);
}

inline bool get_bool_or_default(simdjson::ondemand::value &&val,
                                std::string_view key, bool def = false) {
  auto result = val[key].get_bool();
  return result.error() == simdjson::SUCCESS ? result.value() : def;
}

inline int64_t get_int64_or_default(simdjson::ondemand::value &&val,
                                    std::string_view key, int64_t def = 0) {
  auto result = val[key].get_int64();
  return result.error() == simdjson::SUCCESS ? result.value() : def;
}

} // namespace tokenizers
