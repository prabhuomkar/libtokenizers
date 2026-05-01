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

PreTokenizerResult::PreTokenizerResult() : pre_tokenized({}) {}

PreTokenizerResult::PreTokenizerResult(const icu::UnicodeString& pre_tokenized)
    : pre_tokenized({pre_tokenized}) {}

PreTokenizerResult::PreTokenizerResult(
    const std::vector<icu::UnicodeString>& pre_tokenized)
    : pre_tokenized(pre_tokenized) {}

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
  for (int i = 0; i < input.pre_tokenized.size(); i++) {
    const icu::UnicodeString& token = input.pre_tokenized[i];
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
              current.remove();
            }
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kIsolated:
            if (!current.isEmpty()) {
              result.pre_tokenized.emplace_back(current);
              current.remove();
            }
            result.pre_tokenized.emplace_back(icu::UnicodeString(c));
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kMergedWithPrevious:
            current.append(c);
            result.pre_tokenized.emplace_back(current);
            current.remove();
            token_idx = char_end;
            break;
          case SplitDelimiterBehavior::kMergedWithNext:
            if (!current.isEmpty()) {
              result.pre_tokenized.emplace_back(current);
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
    }
  }
  return result;
}

PreTokenizer::PreTokenizer() {}

std::vector<std::string> PreTokenizer::PreTokenizeString(
    const std::string& input) {
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

std::vector<std::string> BertPreTokenizer::PreTokenizeString(
    const std::string& input) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  PreTokenizerResult pre_tokenized = PreTokenizerResult(unicode_input);
  pre_tokenized = PreTokenize(pre_tokenized);
  std::vector<std::string> result;
  result.reserve(pre_tokenized.pre_tokenized.size());
  for (int i = 0; i < pre_tokenized.pre_tokenized.size(); i++) {
    std::string str;
    pre_tokenized.pre_tokenized[i].toUTF8String(str);
    result.emplace_back(str);
  }
  return result;
}

} // namespace pre_tokenizers

} // namespace tokenizers
