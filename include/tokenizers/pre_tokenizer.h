// Copyright 2025 Omkar Prabhu
#pragma once

#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

namespace pre_tokenizers {

class PreTokenizerResult {
 public:
  PreTokenizerResult();
  explicit PreTokenizerResult(const icu::UnicodeString& pre_tokenized);
  PreTokenizerResult(
      const std::vector<icu::UnicodeString>& pre_tokenized,
      const std::vector<std::vector<std::pair<int, int>>>& char_offsets);
  PreTokenizerResult(const std::vector<icu::UnicodeString>& pre_tokenized,
                     const std::vector<std::pair<int, int>>& offsets);
  std::vector<icu::UnicodeString> pre_tokenized;
  std::vector<std::vector<std::pair<int, int>>> char_offsets;
  std::vector<std::pair<int, int>> offsets;
  bool pre_pre_tokenized;
};

enum class SplitDelimiterBehavior {
  kRemoved,
  kIsolated,
  kMergedWithPrevious,
  kMergedWithNext
};

PreTokenizerResult split(const PreTokenizerResult& input,
                         std::function<bool(UChar32)> should_split,
                         SplitDelimiterBehavior behavior);

class PreTokenizer {
 public:
  PreTokenizer();
  virtual PreTokenizerResult PreTokenize(const PreTokenizerResult& input);
  virtual std::vector<std::pair<std::string, std::pair<int, int>>>
  PreTokenizeString(const std::string& input);
};

// BertPreTokenizer
class BertPreTokenizer : public PreTokenizer {
 public:
  explicit BertPreTokenizer();
  PreTokenizerResult PreTokenize(const PreTokenizerResult& input) override;
  std::vector<std::pair<std::string, std::pair<int, int>>> PreTokenizeString(
      const std::string& input) override;
};

} // namespace pre_tokenizers

} // namespace tokenizers
