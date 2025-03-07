// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>
#include <unicode/unistr.h>

#include <string>
#include <vector>

#include "tokenizers/common.h"
#include "tokenizers/model.h"

static void BM_WordPieceModelIsBad(benchmark::State& state) { // NOLINT
  WordPiece model = WordPiece({{u8"[UNK]", 1}}, u8"[UNK]", u8"##", 100);
  icu::UnicodeString input =
      icu::UnicodeString::fromUTF8(u8"tokenization is important!");
  for (auto _ : state) {
    std::vector<Token> output = model.Tokenize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_WordPieceModelIsFound(benchmark::State& state) { // NOLINT
  WordPiece model(
      {{u8"[UNK]", 1}, {u8"token", 2}, {u8"##izat", 3}, {u8"##ion", 4}},
      u8"[UNK]", u8"##", 100);
  std::string input = u8"tokenization";
  for (auto _ : state) {
    std::vector<Token> output = model.TokenizeString(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_WordPieceUnkToken(benchmark::State& state) { // NOLINT
  WordPiece model({{u8"hello", 1}, {u8"world", 2}, {u8"[UNK]", 3}}, u8"[UNK]",
                  u8"##", 100);
  std::string input = u8"helloqwerty";
  for (auto _ : state) {
    std::vector<Token> output = model.TokenizeString(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_WordPieceModelMaxInputCharsPerWord(
    benchmark::State& state) { // NOLINT
  WordPiece model = WordPiece({{u8"[UNK]", 1}}, u8"[UNK]", u8"##", 5);
  icu::UnicodeString input =
      icu::UnicodeString::fromUTF8(u8"tokenization is important!");
  for (auto _ : state) {
    std::vector<Token> output = model.Tokenize(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_WordPieceModelIsBad)->ThreadPerCpu();
BENCHMARK(BM_WordPieceModelIsFound)->ThreadPerCpu();
BENCHMARK(BM_WordPieceUnkToken)->ThreadPerCpu();
BENCHMARK(BM_WordPieceModelMaxInputCharsPerWord)->ThreadPerCpu();
