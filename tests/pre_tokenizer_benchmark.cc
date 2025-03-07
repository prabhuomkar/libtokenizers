// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>
#include <unicode/unistr.h>

#include <string>
#include <utility>
#include <vector>

#include "tokenizers/pre_tokenizer.h"

static void BM_PreTokenizerSplitRemoved(benchmark::State& state) { // NOLINT
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  for (auto _ : state) {
    PreTokenizerResult result = split(
        input, [](UChar32 c) { return c == '-'; },
        SplitDelimiterBehavior::kRemoved);
    benchmark::DoNotOptimize(result);
  }
}

static void BM_PreTokenizerSplitIsolated(benchmark::State& state) { // NOLINT
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  for (auto _ : state) {
    PreTokenizerResult result = split(
        input, [](UChar32 c) { return c == '-'; },
        SplitDelimiterBehavior::kIsolated);
    benchmark::DoNotOptimize(result);
  }
}

static void BM_PreTokenizerSplitMergedWithPrevious(
    benchmark::State& state) { // NOLINT
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  for (auto _ : state) {
    PreTokenizerResult result = split(
        input, [](UChar32 c) { return c == '-'; },
        SplitDelimiterBehavior::kMergedWithPrevious);
    benchmark::DoNotOptimize(result);
  }
}

static void BM_PreTokenizerSplitMergedWithNext(
    benchmark::State& state) { // NOLINT
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"the-final--countdown"));
  for (auto _ : state) {
    PreTokenizerResult result = split(
        input, [](UChar32 c) { return c == '-'; },
        SplitDelimiterBehavior::kMergedWithNext);
    benchmark::DoNotOptimize(result);
  }
}

static void BM_BertPreTokenizerWhitespaceChars(
    benchmark::State& state) { // NOLINT
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8"Hey friend!     How are you?!?"));
  for (auto _ : state) {
    PreTokenizerResult output = pre_tokenizer.PreTokenize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertPreTokenizerChineseChars(benchmark::State& state) { // NOLINT
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(
      icu::UnicodeString::fromUTF8(u8" 野  口  里  佳  Noguchi Rika"));
  for (auto _ : state) {
    PreTokenizerResult output = pre_tokenizer.PreTokenize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertPreTokenizerAllOps(benchmark::State& state) { // NOLINT
  BertPreTokenizer pre_tokenizer;
  PreTokenizerResult input = PreTokenizerResult(icu::UnicodeString::fromUTF8(
      u8"Hey friend!  野  口  里  佳  Noguchi Rika"));
  for (auto _ : state) {
    PreTokenizerResult output = pre_tokenizer.PreTokenize(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_BertPreTokenizerAllOpsString(benchmark::State& state) { // NOLINT
  BertPreTokenizer pre_tokenizer;
  std::string input = u8"Hey friend!  野  口  里  佳  Noguchi Rika";
  for (auto _ : state) {
    std::vector<std::pair<std::string, std::pair<int, int>>> output =
        pre_tokenizer.PreTokenizeString(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_PreTokenizerSplitRemoved)->ThreadPerCpu();
BENCHMARK(BM_PreTokenizerSplitIsolated)->ThreadPerCpu();
BENCHMARK(BM_PreTokenizerSplitMergedWithPrevious)->ThreadPerCpu();
BENCHMARK(BM_PreTokenizerSplitMergedWithNext)->ThreadPerCpu();
BENCHMARK(BM_BertPreTokenizerWhitespaceChars)->ThreadPerCpu();
BENCHMARK(BM_BertPreTokenizerChineseChars)->ThreadPerCpu();
BENCHMARK(BM_BertPreTokenizerAllOps)->ThreadPerCpu();
BENCHMARK(BM_BertPreTokenizerAllOpsString)->ThreadPerCpu();
