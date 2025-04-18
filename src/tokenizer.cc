// Copyright 2025 Omkar Prabhu
#include "tokenizers/tokenizer.h"

#include <simdjson.h>
#include <unicode/unistr.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

namespace tokenizers {

std::string parseVersion(simdjson::ondemand::document& config) {
  auto version_result = config["version"].get_string();
  return version_result.error() == simdjson::SUCCESS
             ? std::string(version_result.value())
             : "";
}

std::shared_ptr<AddedVocabulary> parseAddedVocabulary(
    simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  auto added_tokens_result = config.get_array();
  if (added_tokens_result.error() == simdjson::SUCCESS) {
    std::vector<AddedToken> added_tokens;
    for (auto element : added_tokens_result) {
      int id = get_int64_or_default(std::move(element), "id");
      std::string content =
          get_string_or_default(std::move(element), "content");
      bool single_word =
          get_bool_or_default(std::move(element), "single_word", false);
      bool lstrip = get_bool_or_default(std::move(element), "lstrip", false);
      bool rstrip = get_bool_or_default(std::move(element), "rstrip", false);
      bool normalized =
          get_bool_or_default(std::move(element), "normalized", false);
      bool special_token =
          get_bool_or_default(std::move(element), "special_token", true);
      added_tokens.emplace_back(id, content, single_word, lstrip, rstrip,
                                normalized, special_token);
    }
    return std::make_shared<AddedVocabulary>(added_tokens);
  }

  return nullptr;
}

std::shared_ptr<normalizers::Normalizer> parseNormalizer(
    simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  std::string type = get_string_or_default(std::move(config), "type");

  if (type == "BertNormalizer") {
    return std::make_shared<normalizers::BertNormalizer>(
        get_bool_or_default(std::move(config), "clean_text"),
        get_bool_or_default(std::move(config), "handle_chinese_chars"),
        get_bool_or_default(std::move(config), "strip_accents", true),
        get_bool_or_default(std::move(config), "lowercase"));
  }

  return nullptr;
}

std::shared_ptr<pre_tokenizers::PreTokenizer> parsePreTokenizer(
    simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  std::string type = get_string_or_default(std::move(config), "type");

  if (type == "BertPreTokenizer") {
    return std::make_shared<pre_tokenizers::BertPreTokenizer>();
  }

  return nullptr;
}

std::shared_ptr<models::Model> parseModel(simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  std::string type = get_string_or_default(std::move(config), "type");

  if (type == "WordPiece") {
    std::unordered_map<std::string, int> vocab;
    for (auto element : config["vocab"].get_object()) {
      vocab[std::string(element.unescaped_key().value())] =
          static_cast<int>(static_cast<int64_t>(element.value()));
    }

    return std::make_shared<models::WordPiece>(
        vocab, get_string_or_default(std::move(config), "unk_token"),
        get_string_or_default(std::move(config), "continuing_subword_prefix"),
        get_int64_or_default(std::move(config), "max_input_chars_per_word", 0));
  }

  return nullptr;
}

std::shared_ptr<post_processors::PostProcessor> parsePostProcessor(
    simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  std::string type = get_string_or_default(std::move(config), "type");

  if (type == "TemplateProcessing") {
    std::vector<post_processors::TemplateProcessor> single;
    if (auto single_result = config["single"].get_array();
        single_result.error() == simdjson::SUCCESS) {
      for (auto element : single_result) {
        for (auto item : element.get_object()) {
          std::string category = std::string(item.unescaped_key().value());
          auto item_object = item.value().get_object();
          int type_id =
              static_cast<int>(static_cast<int64_t>(item_object["type_id"]));
          std::string id = std::string(item_object["id"].get_string().value());
          single.emplace_back(
              post_processors::TemplateProcessor(category, type_id, id));
        }
      }
    }

    std::vector<post_processors::TemplateProcessor> pair;
    if (auto pair_result = config["pair"].get_array();
        pair_result.error() == simdjson::SUCCESS) {
      for (auto element : pair_result) {
        for (auto item : element.get_object()) {
          std::string category = std::string(item.unescaped_key().value());
          auto item_object = item.value().get_object();
          int type_id =
              static_cast<int>(static_cast<int64_t>(item_object["type_id"]));
          std::string id = std::string(item_object["id"].get_string().value());
          pair.emplace_back(
              post_processors::TemplateProcessor(category, type_id, id));
        }
      }
    }

    std::unordered_map<std::string, int> special_tokens;
    auto special_tokens_json = config["special_tokens"];
    for (auto element : special_tokens_json.get_object()) {
      std::string key = std::string(element.unescaped_key().value());
      auto element_object = element.value().get_object();
      auto ids = element_object["ids"].get_array().value();
      int value =
          static_cast<int>(static_cast<int64_t>(ids.at(0).get_int64().value()));
      special_tokens[key] = value;
    }

    return std::make_shared<post_processors::TemplateProcessing>(
        single, pair, special_tokens);
  }

  return nullptr;
}

std::shared_ptr<decoders::Decoder> parseDecoder(
    simdjson::ondemand::value& config) {
  if (config.is_null())
    return nullptr;

  std::string type = get_string_or_default(std::move(config), "type");

  if (type == "WordPiece") {
    return std::make_shared<decoders::WordPieceDecoder>(
        get_string_or_default(std::move(config), "prefix", "##"),
        get_bool_or_default(std::move(config), "cleanup", true));
  }

  return nullptr;
}

Tokenizer::Tokenizer() : version("") {}

Tokenizer::Tokenizer(const std::string& json_config) {
  if (json_config.length() == 0) {
    throw std::invalid_argument(
        "json config is required for initializing a tokenizer");
  }
  simdjson::ondemand::parser parser;
  simdjson::padded_string padded_config = simdjson::padded_string(json_config);
  simdjson::ondemand::document config = parser.iterate(padded_config);
  version = parseVersion(config);
  simdjson::ondemand::value added_tokens_config = config["added_tokens"];
  added_vocabulary = parseAddedVocabulary(added_tokens_config);
  simdjson::ondemand::value normalizer_config = config["normalizer"];
  normalizer = parseNormalizer(normalizer_config);
  simdjson::ondemand::value pre_tokenizer_config = config["pre_tokenizer"];
  pre_tokenizer = parsePreTokenizer(pre_tokenizer_config);
  simdjson::ondemand::value model_config = config["model"];
  model = parseModel(model_config);
  simdjson::ondemand::value post_processor_config = config["post_processor"];
  post_processor = parsePostProcessor(post_processor_config);
  simdjson::ondemand::value decoder_config = config["decoder"];
  decoder = parseDecoder(decoder_config);
}

Encoding Tokenizer::Encode(const std::string& input, bool add_special_tokens) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  std::vector<Encoding> encodings = {EncodeSingleSequence(&unicode_input, 0)};
  if (truncation.get() != nullptr) {
    truncation->TruncateEncodings(encodings);
  }
  if (add_special_tokens && post_processor.get() != nullptr) {
    encodings = post_processor->ProcessEncodings(encodings);
  }
  if (padding.get() != nullptr) {
    padding->PadEncodings(encodings);
  }
  Encoding encoding;
  for (const Encoding& enc : encodings) {
    encoding.ids.insert(encoding.ids.end(), enc.ids.begin(), enc.ids.end());
    encoding.tokens.insert(encoding.tokens.end(), enc.tokens.begin(),
                           enc.tokens.end());
    encoding.type_ids.insert(encoding.type_ids.end(), enc.type_ids.begin(),
                             enc.type_ids.end());
    encoding.offsets.insert(encoding.offsets.end(), enc.offsets.begin(),
                            enc.offsets.end());
    encoding.word_ids.insert(encoding.word_ids.end(), enc.word_ids.begin(),
                             enc.word_ids.end());
    encoding.special_tokens_mask.insert(encoding.special_tokens_mask.end(),
                                        enc.special_tokens_mask.begin(),
                                        enc.special_tokens_mask.end());
    encoding.attention_mask.insert(encoding.attention_mask.end(),
                                   enc.attention_mask.begin(),
                                   enc.attention_mask.end());
  }
  return encoding;
}

Encoding Tokenizer::Encode(const std::pair<std::string, std::string>& input,
                           bool add_special_tokens) {
  std::pair<icu::UnicodeString, icu::UnicodeString> unicode_input = {
      icu::UnicodeString::fromUTF8(input.first),
      icu::UnicodeString::fromUTF8(input.second)};
  std::vector<Encoding> encodings = {
      EncodeSingleSequence(&unicode_input.first, 0),
      EncodeSingleSequence(&unicode_input.second, 1)};
  if (truncation.get() != nullptr) {
    truncation->TruncateEncodings(encodings);
  }
  if (add_special_tokens && post_processor.get() != nullptr) {
    encodings = post_processor->ProcessEncodings(encodings);
  }
  if (padding.get() != nullptr) {
    padding->PadEncodings(encodings);
  }
  Encoding encoding;
  for (const Encoding& enc : encodings) {
    encoding.ids.insert(encoding.ids.end(), enc.ids.begin(), enc.ids.end());
    encoding.tokens.insert(encoding.tokens.end(), enc.tokens.begin(),
                           enc.tokens.end());
    encoding.type_ids.insert(encoding.type_ids.end(), enc.type_ids.begin(),
                             enc.type_ids.end());
    encoding.offsets.insert(encoding.offsets.end(), enc.offsets.begin(),
                            enc.offsets.end());
    encoding.word_ids.insert(encoding.word_ids.end(), enc.word_ids.begin(),
                             enc.word_ids.end());
    encoding.special_tokens_mask.insert(encoding.special_tokens_mask.end(),
                                        enc.special_tokens_mask.begin(),
                                        enc.special_tokens_mask.end());
    encoding.attention_mask.insert(encoding.attention_mask.end(),
                                   enc.attention_mask.begin(),
                                   enc.attention_mask.end());
  }
  return encoding;
}

std::string Tokenizer::Decode(const std::vector<int>& ids,
                              bool skip_special_tokens) {
  std::vector<std::string> tokens;
  for (const int id : ids) {
    std::optional<std::string> opt_token = model->IdToToken(id);
    if (!opt_token.has_value())
      continue;
    if (!skip_special_tokens ||
        !added_vocabulary->IsSpecialToken(opt_token.value())) {
      tokens.emplace_back(opt_token.value());
    }
  }
  std::string result = "";
  tokens = decoder->DecodeChain(tokens);
  for (const std::string& token : tokens) {
    result += token;
  }
  return result;
}

Encoding Tokenizer::EncodeSingleSequence(icu::UnicodeString* unicode_input,
                                         int type_id) {
  normalizers::NormalizerResult normalized =
      normalizers::NormalizerResult(*unicode_input);
  std::vector<normalizers::NormalizerResult> splits = {normalized};
  if (added_vocabulary.get() != nullptr) {
    splits = added_vocabulary->FindSplits(normalized);
  }
  if (normalizer.get() != nullptr) {
    for (normalizers::NormalizerResult& split : splits) {
      if (!split.pre_normalized) {
        split = normalizer->Normalize(normalized);
      }
    }
  }
  std::vector<icu::UnicodeString> normalized_strings;
  std::vector<std::vector<std::pair<int, int>>> normalized_offsets;
  for (const normalizers::NormalizerResult& split : splits) {
    normalized_strings.emplace_back(split.normalized);
    normalized_offsets.emplace_back(split.offsets);
  }
  pre_tokenizers::PreTokenizerResult pre_tokenized =
      pre_tokenizers::PreTokenizerResult(normalized_strings,
                                         normalized_offsets);
  if (pre_tokenizer.get() != nullptr) {
    pre_tokenized = pre_tokenizer->PreTokenize(pre_tokenized);
  }
  Encoding encoding;
  if (model.get() != nullptr) {
    int word_id = -1;
    for (int i = 0; i < pre_tokenized.pre_tokenized.size(); i++) {
      std::vector<Token> tokens = model->Tokenize(
          pre_tokenized.pre_tokenized[i], pre_tokenized.offsets[i]);
      for (const Token& token : tokens) {
        encoding.ids.emplace_back(token.id);
        encoding.tokens.emplace_back(token.value);
        encoding.type_ids.emplace_back(type_id);
        encoding.offsets.emplace_back(token.offsets);
        encoding.word_ids.emplace_back(token.is_continuing_subword ? word_id
                                                                   : ++word_id);
        encoding.special_tokens_mask.emplace_back(0);
        encoding.attention_mask.emplace_back(1);
      }
    }
  }
  return encoding;
}

} // namespace tokenizers
