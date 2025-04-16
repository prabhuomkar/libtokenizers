// Copyright 2025 Omkar Prabhu
#pragma once

#include <simdjson.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/added_vocabulary.h"
#include "tokenizers/common.h"
#include "tokenizers/decoder.h"
#include "tokenizers/model.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/post_processor.h"
#include "tokenizers/pre_tokenizer.h"
#include "tokenizers/utils.h"

namespace tokenizers {

std::string parseVersion(simdjson::ondemand::document &config);
std::shared_ptr<AddedVocabulary> parseAddedVocabulary(
    simdjson::ondemand::value &config);
std::shared_ptr<tokenizers::normalizers::Normalizer> parseNormalizer(
    simdjson::ondemand::value &config);
std::shared_ptr<tokenizers::pre_tokenizers::PreTokenizer> parsePreTokenizer(
    simdjson::ondemand::value &config);
std::shared_ptr<tokenizers::models::Model> parseModel(
    simdjson::ondemand::value &config);
std::shared_ptr<tokenizers::post_processors::PostProcessor> parsePostProcessor(
    simdjson::ondemand::value &config);
std::shared_ptr<tokenizers::decoders::Decoder> parseDecoder(
    simdjson::ondemand::value &config);

class Tokenizer {
 public:
  Tokenizer();
  explicit Tokenizer(const std::string &json_config);

  Encoding Encode(const std::string &input, bool add_special_tokens = true);
  Encoding Encode(const std::pair<std::string, std::string> &input,
                  bool add_special_tokens = true);
  std::string Decode(const std::vector<int> &ids,
                     bool skip_special_tokens = true);

  std::shared_ptr<tokenizers::normalizers::Normalizer> normalizer;
  std::shared_ptr<tokenizers::pre_tokenizers::PreTokenizer> pre_tokenizer;
  std::shared_ptr<tokenizers::models::Model> model;
  std::shared_ptr<tokenizers::post_processors::PostProcessor> post_processor;
  std::shared_ptr<tokenizers::decoders::Decoder> decoder;
  std::shared_ptr<Truncation> truncation;
  std::shared_ptr<AddedVocabulary> added_vocabulary;
  std::shared_ptr<Padding> padding;
  std::string version;

 private:
  Encoding EncodeSingleSequence(icu::UnicodeString *unicode_input, int type_id);
};

} // namespace tokenizers
