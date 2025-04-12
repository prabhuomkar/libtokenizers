// Copyright 2025 Omkar Prabhu
#pragma once

#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

namespace normalizers {

class NormalizerResult {
 public:
  explicit NormalizerResult(const icu::UnicodeString& normalized);
  NormalizerResult(const icu::UnicodeString& normalized,
                   const std::vector<std::pair<int, int>>& offsets);
  icu::UnicodeString normalized;
  std::vector<std::pair<int, int>> offsets;
};

void transform_offsets(NormalizerResult* input,
                       const std::vector<std::pair<int, int>>& ops);

class Normalizer {
 public:
  Normalizer();
  virtual NormalizerResult Normalize(NormalizerResult input);
  virtual std::string NormalizeString(std::string input);
};

// BertNormalizer
class BertNormalizer : public Normalizer {
 public:
  explicit BertNormalizer(bool clean_text = true,
                          bool handle_chinese_chars = true,
                          bool strip_accents = true, bool lowercase = true);
  NormalizerResult Normalize(NormalizerResult input) override;
  std::string NormalizeString(std::string input) override;

 private:
  bool clean_text_;
  bool handle_chinese_chars_;
  bool strip_accents_;
  bool lowercase_;
};

void doCleanText(NormalizerResult* input);

void doHandleChineseChars(NormalizerResult* input);

void doStripAccents(NormalizerResult* input);

void doLowercase(NormalizerResult* input);

bool isControl(UChar32 c);

bool isWhitespace(UChar32 c);

bool isChineseChar(UChar32 c);

} // namespace normalizers

} // namespace tokenizers
