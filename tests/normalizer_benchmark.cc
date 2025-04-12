// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>
#include <unicode/unistr.h>

#include <string>

#include "tokenizers/normalizer.h"

using tokenizers::normalizers::BertNormalizer;
using tokenizers::normalizers::isChineseChar;
using tokenizers::normalizers::isControl;
using tokenizers::normalizers::isWhitespace;
using tokenizers::normalizers::Normalizer;
using tokenizers::normalizers::NormalizerResult;

static void BM_BertNormalizerNoOp(benchmark::State& state) { // NOLINT
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString::fromUTF8(u8"Hello, World!"));
  BertNormalizer normalizer(false, false, false, false);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerCleanText(benchmark::State& state) { // NOLINT
  NormalizerResult input = NormalizerResult(
      icu::UnicodeString::fromUTF8(u8"He\u200Bl\uFFFDl\to\n \rWo\tr\nl\rd"));
  BertNormalizer normalizer(true, false, false, false);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerHandleChineseChars(
    benchmark::State& state) { // NOLINT
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString::fromUTF8(u8"习近平访问了纽约。"));
  BertNormalizer normalizer(false, true, false, false);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerStripAccents(benchmark::State& state) { // NOLINT
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString::fromUTF8(u8"café naïve são élève"));
  BertNormalizer normalizer(false, false, true, false);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerLowercase(benchmark::State& state) { // NOLINT
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString::fromUTF8(u8"HELLO WORLD"));
  BertNormalizer normalizer(false, false, false, true);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerAllOps(benchmark::State& state) { // NOLINT
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString::fromUTF8(u8"Café 中文"));
  BertNormalizer normalizer(true, true, true, true);
  for (auto _ : state) {
    NormalizerResult output = normalizer.Normalize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertNormalizerAllOpsString(benchmark::State& state) { // NOLINT
  std::string input = u8"Café 中文";
  BertNormalizer normalizer(true, true, true, true);
  for (auto _ : state) {
    std::string output = normalizer.NormalizeString(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_BertNormalizerNoOp)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerCleanText)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerHandleChineseChars)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerStripAccents)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerLowercase)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerAllOps)->ThreadPerCpu();
BENCHMARK(BM_BertNormalizerAllOpsString)->ThreadPerCpu();
BENCHMARK_MAIN();
