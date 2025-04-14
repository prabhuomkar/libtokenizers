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

#include "tokenizers/common.h"
#include "tokenizers/decoder.h"
#include "tokenizers/model.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/post_processor.h"
#include "tokenizers/pre_tokenizer.h"
#include "tokenizers/tokenizer.h"
#include "tokenizers/utils.h"

using tokenizers::Encoding;
using tokenizers::Tokenizer;
using tokenizers::models::WordPiece;
using tokenizers::normalizers::BertNormalizer;
using tokenizers::post_processors::TemplateProcessing;
using tokenizers::post_processors::TemplateProcessor;
using tokenizers::pre_tokenizers::BertPreTokenizer;

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

static void BM_TokenizerEncodeSingle(benchmark::State& state) { // NOLINT
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

static void BM_TokenizerEncodePair(benchmark::State& state) { // NOLINT
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

static void BM_TokenizerDecodeSingle(benchmark::State& state) { // NOLINT
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
  tokenizer.special_tokens =
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}});
  tokenizer.decoder =
      std::make_shared<tokenizers::decoders::WordPieceDecoder>();
  std::vector<int> input = {100, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13,  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 19, 24,
                            25,  26, 27, 28, 29, 30, 31, 32, 0,  34, 101};
  for (auto _ : state) {
    std::string output = tokenizer.Decode(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerDecodePair(benchmark::State& state) { // NOLINT
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
  tokenizer.special_tokens =
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}});
  tokenizer.decoder =
      std::make_shared<tokenizers::decoders::WordPieceDecoder>();
  std::vector<int> input = {100, 1,  2,  3,   4,  5,  6,  7,  8,  9,
                            10,  11, 35, 101, 12, 13, 14, 15, 16, 17,
                            18,  19, 20, 21,  22, 23, 24, 20, 25, 26,
                            27,  28, 29, 30,  31, 32, 33, 0,  35, 101};
  for (auto _ : state) {
    std::string output = tokenizer.Decode(input);
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

static void BM_TokenizerDecodeSingleFromConfig(
    benchmark::State& state) { // NOLINT
  std::string config =
      read_json_for_benchmark("../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::vector<int> input = {101,   7592, 2088, 999,   1045, 1005,  1049,  4083,
                            14324, 1011, 2241, 17953, 2361, 2007,  14477, 4246,
                            8551,  3085, 5366, 1999,  7509, 9094,  1010,  1781,
                            1755,  1810, 1817, 1010,  1998, 18750, 100,   1740,
                            100,   100,  100,  100,   100,  1012,  102};
  for (auto _ : state) {
    std::string output = tokenizer.Decode(input, false);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TokenizerDecodePairFromConfig(
    benchmark::State& state) { // NOLINT
  std::string config =
      read_json_for_benchmark("../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::vector<int> input = {
      101,   7592, 2088, 999,  1045, 1005, 1049,  4083, 14324, 1011, 2241,
      17953, 2361, 1012, 102,  2057, 2031, 14477, 4246, 8551,  3085, 5366,
      1999,  7509, 9094, 1010, 1781, 1755, 1810,  1817, 1010,  1998, 18750,
      100,   1740, 100,  100,  100,  100,  100,   1012, 102};
  for (auto _ : state) {
    std::string output = tokenizer.Decode(input, false);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_TokenizerNoOp)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodeSingle)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodePair)->ThreadPerCpu();
BENCHMARK(BM_TokenizerDecodeSingle)->ThreadPerCpu();
BENCHMARK(BM_TokenizerDecodePair)->ThreadPerCpu();
BENCHMARK(BM_TokenizerInitFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodeSingleFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerEncodePairFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerDecodeSingleFromConfig)->ThreadPerCpu();
BENCHMARK(BM_TokenizerDecodePairFromConfig)->ThreadPerCpu();
