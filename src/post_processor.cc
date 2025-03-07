// Copyright 2025 Omkar Prabhu
#include "tokenizers/post_processor.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

PostProcessor::PostProcessor() {}

std::vector<Encoding> PostProcessor::ProcessEncodings(
    const std::vector<Encoding>& encodings) {
  return {};
}

TemplateProcessor::TemplateProcessor() : category(""), type_id(0), id("") {}

TemplateProcessor::TemplateProcessor(const std::string& category, int type_id,
                                     const std::string& id)
    : category(category), type_id(type_id), id(id) {}

TemplateProcessing::TemplateProcessing() {}

TemplateProcessing::TemplateProcessing(
    const std::vector<TemplateProcessor>& single,
    const std::vector<TemplateProcessor>& pair,
    const std::unordered_map<std::string, int>& special_tokens)
    : single_(single), pair_(pair), special_tokens_(special_tokens) {}

std::vector<Encoding> TemplateProcessing::ProcessEncodings(
    const std::vector<Encoding>& encodings) {
  const std::vector<TemplateProcessor>& seq_processor =
      encodings.size() == 1 ? single_ : pair_;
  std::vector<Encoding> result;
  result.reserve(seq_processor.size());
  int seq_id = 0;
  for (const TemplateProcessor& processor : seq_processor) {
    if (processor.category == "SpecialToken") {
      auto it = special_tokens_.find(processor.id);
      if (it != special_tokens_.end()) {
        result.emplace_back(Encoding({it->second}, {processor.type_id},
                                     {processor.id}, {{0, 0}}, {std::nullopt},
                                     {1}, {1}));
      }
    } else if (processor.category == "Sequence") {
      result.emplace_back(std::move(encodings[seq_id]));
      seq_id++;
    }
  }
  return result;
}
