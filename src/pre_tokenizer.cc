// Copyright 2025 Omkar Prabhu
#include "tokenizers/pre_tokenizer.h"

#include <unicode/schriter.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

namespace pre_tokenizers {

PreTokenizerResult::PreTokenizerResult() : pre_tokenized({}), offsets({}) {}

PreTokenizerResult::PreTokenizerResult(const icu::UnicodeString& pre_tokenized)
    : pre_tokenized({pre_tokenized}) {
  char_offsets.reserve(1);
  icu::StringCharacterIterator it(pre_tokenized);
  for (it.first(); it.hasNext();) {
    int start = it.getIndex();
    it.next32PostInc();
    int end = it.getIndex();
    char_offsets[0].emplace_back(start, end);
  }
  offsets.emplace_back(0, pre_tokenized.countChar32());
}

PreTokenizerResult::PreTokenizerResult(
    const std::vector<icu::UnicodeString>& pre_tokenized,
    const std::vector<std::vector<std::pair<int, int>>>& char_offsets)
    : pre_tokenized(pre_tokenized), char_offsets(char_offsets) {
  offsets.reserve(pre_tokenized.size());
  std::transform(char_offsets.begin(), char_offsets.end(),
                 std::back_inserter(offsets),
                 [](const std::vector<std::pair<int, int>>& vec) {
                   return std::make_pair(vec.front().first, vec.back().second);
                 });
}

PreTokenizerResult::PreTokenizerResult(
    const std::vector<icu::UnicodeString>& pre_tokenized,
    const std::vector<std::pair<int, int>>& offsets)
    : pre_tokenized(pre_tokenized), char_offsets({}), offsets(offsets) {}

// When splitting on '-' for example, with input "the-final--countdown":
// Removed => [ "the", "", "final", "", "", "countdown" ]
// Isolated => [ "the", "-", "final", "-", "-", "countdown" ]
// MergedWithPrevious => [ "the-", "final-", "-", "countdown" ]
// MergedWithNext => [ "the", "-final", "-", "-countdown" ]
PreTokenizerResult split(const PreTokenizerResult& input,
                         std::function<bool(UChar32)> should_split,
                         SplitDelimiterBehavior behavior) {
  PreTokenizerResult result;
  result.pre_tokenized.reserve(input.pre_tokenized.size() * 2);
  result.offsets.reserve(input.offsets.size() * 2);
  for (int i = 0; i < input.pre_tokenized.size(); i++) {
    const icu::UnicodeString& token = input.pre_tokenized[i];
    const std::pair<int, int>& token_offset = input.offsets[i];
    const std::vector<std::pair<int, int>>& char_offsets =
        input.char_offsets[i];
    icu::UnicodeString current;
    icu::StringCharacterIterator it(token);
    int token_idx = 0;
    for (it.first(); it.hasNext();) {
      int char_start = it.getIndex();
      UChar32 c = it.next32PostInc();
      int char_end = it.getIndex();
      if (should_split(c)) {
        switch (behavior) {
          case SplitDelimiterBehavior::kRemoved:
            if (!current.isEmpty()) {
              result.pre_tokenized.emplace_back(current);
              result.offsets.emplace_back(char_offsets[token_idx].first,
                                          char_offsets[char_start - 1].second);
              std::vector<std::pair<int, int>> current_char_offsets;
              for (int j = token_idx; j < char_start; j++) {
                current_char_offsets.emplace_back(char_offsets[j].first,
                                                  char_offsets[j].second);
              }
              result.char_offsets.emplace_back(current_char_offsets);
              current.remove();
            }
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kIsolated:
            if (!current.isEmpty()) {
              result.pre_tokenized.emplace_back(current);
              result.offsets.emplace_back(char_offsets[token_idx].first,
                                          char_offsets[char_start - 1].second);
              std::vector<std::pair<int, int>> current_char_offsets;
              for (int j = token_idx; j < char_start; j++) {
                current_char_offsets.emplace_back(char_offsets[j].first,
                                                  char_offsets[j].second);
              }
              result.char_offsets.emplace_back(current_char_offsets);
              current.remove();
            }
            result.pre_tokenized.emplace_back(icu::UnicodeString(c));
            result.offsets.emplace_back(char_offsets[char_start].first,
                                        char_offsets[char_end - 1].second);
            {
              std::vector<std::pair<int, int>> append_char_offsets;
              for (int j = char_start; j < char_end; j++) {
                append_char_offsets.emplace_back(char_offsets[j].first,
                                                 char_offsets[j].second);
              }
              result.char_offsets.emplace_back(append_char_offsets);
            }
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kMergedWithPrevious:
            current.append(c);
            result.pre_tokenized.emplace_back(current);
            result.offsets.emplace_back(char_offsets[token_idx].first,
                                        char_offsets[char_end - 1].second);
            {
              std::vector<std::pair<int, int>> current_char_offsets;
              for (int j = token_idx; j < char_end; j++) {
                current_char_offsets.emplace_back(char_offsets[j].first,
                                                  char_offsets[j].second);
              }
              result.char_offsets.emplace_back(current_char_offsets);
            }
            current.remove();
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kMergedWithNext:
            if (!current.isEmpty()) {
              result.pre_tokenized.emplace_back(current);
              result.offsets.emplace_back(char_offsets[token_idx].first,
                                          char_offsets[char_start - 1].second);
              std::vector<std::pair<int, int>> current_char_offsets;
              for (int j = token_idx; j < char_start; j++) {
                current_char_offsets.emplace_back(char_offsets[j].first,
                                                  char_offsets[j].second);
              }
              current.remove();
            }
            token_idx = char_start;
            current.append(c);
            break;
        }
      } else {
        current.append(c);
      }
    }
    if (!current.isEmpty()) {
      result.pre_tokenized.emplace_back(current);
      result.offsets.emplace_back(char_offsets[token_idx].first,
                                  char_offsets.back().second);
      std::vector<std::pair<int, int>> current_char_offsets;
      for (int j = token_idx; j < char_offsets.size(); j++) {
        current_char_offsets.emplace_back(char_offsets[j].first,
                                          char_offsets[j].second);
      }
      result.char_offsets.emplace_back(current_char_offsets);
    }
  }
  return result;
}

PreTokenizer::PreTokenizer() {}

std::vector<std::pair<std::string, std::pair<int, int>>>
PreTokenizer::PreTokenizeString(const std::string& input) {
  return {};
}

PreTokenizerResult PreTokenizer::PreTokenize(const PreTokenizerResult& input) {
  return input;
}

BertPreTokenizer::BertPreTokenizer() {}

PreTokenizerResult BertPreTokenizer::PreTokenize(
    const PreTokenizerResult& input) {
  PreTokenizerResult input_pre_tokenized = split(
      input, [](UChar32 c) { return u_isWhitespace(c); },
      SplitDelimiterBehavior::kRemoved);
  input_pre_tokenized = split(
      input_pre_tokenized, [](UChar32 c) { return u_ispunct(c); },
      SplitDelimiterBehavior::kIsolated);
  return input_pre_tokenized;
}

std::vector<std::pair<std::string, std::pair<int, int>>>
BertPreTokenizer::PreTokenizeString(const std::string& input) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  PreTokenizerResult pre_tokenized = PreTokenizerResult(unicode_input);
  pre_tokenized = PreTokenize(pre_tokenized);
  std::vector<std::pair<std::string, std::pair<int, int>>> result;
  result.reserve(pre_tokenized.pre_tokenized.size());
  for (int i = 0; i < pre_tokenized.pre_tokenized.size(); i++) {
    std::string str;
    pre_tokenized.pre_tokenized[i].toUTF8String(str);
    result.emplace_back(str, pre_tokenized.offsets[i]);
  }
  return result;
}

} // namespace pre_tokenizers

} // namespace tokenizers
