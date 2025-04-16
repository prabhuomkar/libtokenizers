// Copyright 2025 Omkar Prabhu
#include "tokenizers/added_vocabulary.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/normalizer.h"
#include "tokenizers/utils.h"

using tokenizers::normalizers::NormalizerResult;

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

AddedVocabulary::AddedVocabulary(const std::vector<AddedToken>& tokens) {
  for (const auto& token : tokens) {
    tokens_[token.id] = token;
    added_tokens_map_[token.content] = token.id;
    added_tokens_map_r_[token.id] = token.content;
    if (token.special_token) {
      special_tokens_.insert(token.content);
    }
    patterns_.emplace_back(icu::UnicodeString::fromUTF8(token.content));
  }
}

bool AddedVocabulary::IsSpecialToken(const std::string& token) {
  return special_tokens_.find(token) != special_tokens_.end();
}

std::vector<NormalizerResult> AddedVocabulary::FindSplits(
    const NormalizerResult& input) {
  std::vector<NormalizerResult> splits;
  const icu::UnicodeString& input_normalized = input.normalized;
  const std::vector<std::pair<int, int>>& input_offsets = input.offsets;

  std::vector<std::pair<int, int>> matches =
      FindMatches(input_normalized, patterns_);

  int total_len = input_normalized.countChar32();
  int start_offset = 0;
  for (const std::pair<int, int>& match : matches) {
    int start = match.first;
    int stop = match.second;
    icu::UnicodeString matched =
        input_normalized.tempSubStringBetween(start, stop);
    std::string matched_str;
    matched.toUTF8String(matched_str);

    auto it = added_tokens_map_.find(matched_str);
    if (it == added_tokens_map_.end())
      continue;

    const AddedToken& token = tokens_[it->second];

    if (token.single_word) {
      bool start_space =
          start == 0 || input_normalized.charAt(start - 1) == ' ';
      bool stop_space =
          stop == total_len || input_normalized.charAt(stop) == ' ';
      if (!start_space || !stop_space) {
        continue;
      }
    }

    if (token.lstrip) {
      while (start > 0 && u_isUWhiteSpace(input_normalized.charAt(start - 1))) {
        --start;
      }
    }

    if (token.rstrip) {
      while (stop < total_len &&
             u_isUWhiteSpace(input_normalized.charAt(stop))) {
        ++stop;
      }
    }

    if (start_offset < start) {
      splits.emplace_back(NormalizerResult(
          input_normalized.tempSubStringBetween(start_offset, start),
          std::vector<std::pair<int, int>>(input_offsets.begin() + start_offset,
                                           input_offsets.begin() + start)));
    }
    splits.emplace_back(NormalizerResult(
        input_normalized.tempSubStringBetween(start, stop),
        std::vector<std::pair<int, int>>(input_offsets.begin() + start,
                                         input_offsets.begin() + stop),
        token.special_token));
    start_offset = stop;
  }

  if (start_offset < total_len) {
    splits.emplace_back(NormalizerResult(
        input_normalized.tempSubStringBetween(start_offset, total_len),
        std::vector<std::pair<int, int>>(input_offsets.begin() + start_offset,
                                         input_offsets.end())));
  }

  return splits;
}

} // namespace tokenizers
