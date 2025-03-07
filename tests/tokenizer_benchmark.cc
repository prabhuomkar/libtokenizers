// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>
#include <unicode/unistr.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/tokenizer.h"

std::string read_json_for_benchmark(const std::string& filepath) {
  std::ifstream file(filepath);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

static void BM_TokenizerNoOp(benchmark::State& state) { // NOLINT
  Tokenizer tokenizer;
  std::string input = u8"Hello, World!";
  for (auto _ : state) {
    Encoding output = tokenizer.Encode(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerAllOpsSingle(benchmark::State& state) { // NOLINT
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1},    {"world", 2},
          {"!", 3},      {"i", 4},        {"'", 5},
          {"m", 6},      {"learning", 7}, {"bert", 8},
          {"-", 9},      {"based", 10},   {"nlp", 11},
          {"with", 12},  {"un", 13},      {"##affordable", 14},
          {"costs", 15}, {"in", 16},      {"sao", 17},
          {"paulo", 18}, {",", 19},       {"北", 20},
          {"京", 21},    {"大", 22},      {"学", 23},
          {"and", 24},   {"python", 25},  {"是", 26},
          {"一", 27},    {"种", 28},      {"编", 29},
          {"程", 30},    {"语", 31},      {"言", 32},
          {"❤️", 33},     {".", 34}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  std::string input =
      u8"Hello world! I'm learning BERT-based NLP with "
      u8"unaffordable costs in "
      u8"São Paulo, 北京大学, and Python是一种编程语言 ❤️.";
  for (auto _ : state) {
    Encoding output = tokenizer.Encode(input, true);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerAllOpsPair(benchmark::State& state) { // NOLINT
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1}, {"world", 2},   {"!", 3},
          {"i", 4},      {"'", 5},     {"m", 6},       {"learning", 7},
          {"bert", 8},   {"-", 9},     {"based", 10},  {"nlp", 11},
          {"we", 12},    {"have", 13}, {"un", 14},     {"##affordable", 15},
          {"costs", 16}, {"in", 17},   {"sao", 18},    {"paulo", 19},
          {",", 20},     {"北", 21},   {"京", 22},     {"大", 23},
          {"学", 24},    {"and", 25},  {"python", 26}, {"是", 27},
          {"一", 28},    {"种", 29},   {"编", 30},     {"程", 31},
          {"语", 32},    {"言", 33},   {"❤️", 34},      {".", 35}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  for (auto _ : state) {
    Encoding output = tokenizer.Encode(
        std::make_pair(
            u8"Hello world! I'm learning BERT-based NLP.",
            u8"We have unaffordable costs in São Paulo, 北京大学, and "
            u8"Python是一种编程语言 ❤️."),
        true);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerInitFromConfig(benchmark::State& state) { // NOLINT
  std::string config =
      read_json_for_benchmark("../scripts/tokenizers/bert-base-uncased.json");
  for (auto _ : state) {
    Tokenizer tokenizer = Tokenizer(config);
    benchmark::DoNotOptimize(tokenizer);
  }
}

static void BM_TokenizerEncodeSingleFromConfig(
    benchmark::State& state) { // NOLINT
  std::string config =
      read_json_for_benchmark("../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::string input =
      u8"Hello world! I'm learning BERT-based NLP with "
      u8"unaffordable costs in "
      u8"São Paulo, 北京大学, and Python是一种编程语言.";
  for (auto _ : state) {
    Encoding output = tokenizer.Encode(input, true);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerEncodePairFromConfig(
    benchmark::State& state) { // NOLINT
  std::string config =
      read_json_for_benchmark("../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::pair<std::string, std::string> input =
      std::make_pair(u8"Hello world! I'm learning BERT-based NLP.",
                     u8"We have unaffordable costs in São Paulo, 北京大学, and "
                     u8"Python是一种编程语言.");
  for (auto _ : state) {
    Encoding output = tokenizer.Encode(input, true);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_TokenizerNoOp)->ThreadPerCpu();
BENCHMARK(BM_TokenizerAllOpsSingle)->ThreadPerCpu();
BENCHMARK(BM_TokenizerAllOpsPair)->ThreadPerCpu();
BENCHMARK(BM_TokenizerInitFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodeSingleFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodePairFromConfig)->ThreadPerCpu();
