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
  explicit PreTokenizerResult(
      const std::vector<icu::UnicodeString>& pre_tokenized);
  PreTokenizerResult(const icu::UnicodeString& pre_tokenized,
                     bool pre_pre_tokenized);
  std::vector<icu::UnicodeString> pre_tokenized;
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
  virtual std::vector<std::string> PreTokenizeString(const std::string& input);
};

// BertPreTokenizer
class BertPreTokenizer : public PreTokenizer {
 public:
  explicit BertPreTokenizer();
  PreTokenizerResult PreTokenize(const PreTokenizerResult& input) override;
  std::vector<std::string> PreTokenizeString(const std::string& input) override;
};

} // namespace pre_tokenizers

} // namespace tokenizers
