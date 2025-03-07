// Copyright 2025 Omkar Prabhu
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

class PostProcessor {
 public:
  PostProcessor();
  virtual std::vector<Encoding> ProcessEncodings(
      const std::vector<Encoding>& encodings);

 private:
};

class TemplateProcessor {
 public:
  TemplateProcessor();
  TemplateProcessor(const std::string& category, int type_id,
                    const std::string& id);
  std::string category; // SpecialToken, Sequence
  int type_id;          // 0, 1
  std::string
      id; // [CLS], [SEP], A, B, <s>, </s>, <|startoftext|>, <|endoftext|>
};

// TemplateProcessing
class TemplateProcessing : public PostProcessor {
 public:
  TemplateProcessing();
  TemplateProcessing(
      const std::vector<TemplateProcessor>& single,
      const std::vector<TemplateProcessor>& pair,
      const std::unordered_map<std::string, int>& special_tokens);
  std::vector<Encoding> ProcessEncodings(
      const std::vector<Encoding>& encodings) override;

 private:
  std::vector<TemplateProcessor> single_;
  std::vector<TemplateProcessor> pair_;
  std::unordered_map<std::string, int> special_tokens_;
};
