// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>
#include <unicode/unistr.h>

#include <string>
#include <vector>

#include "tokenizers/added_vocabulary.h"
#include "tokenizers/normalizer.h"

using tokenizers::AddedToken;
using tokenizers::AddedVocabulary;
using tokenizers::normalizers::NormalizerResult;

static void BM_AddedVocabularyIsSpecialTokenTrue(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[UNK]", false, false, false, false, true)});
  for (auto _ : state) {
    bool output = added_vocabulary.IsSpecialToken("[UNK]");
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyIsSpecialTokenFalse(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[UNK]", false, false, false, false, true)});
  for (auto _ : state) {
    bool output = added_vocabulary.IsSpecialToken("[CLS]");
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyFindSplitsSpecialToken(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "[MASK]", false, false, false, false, true)});
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  for (auto _ : state) {
    std::vector<NormalizerResult> output = added_vocabulary.FindSplits(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyFindSplitsSingleWord(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", true, false, false, false, false)});
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  for (auto _ : state) {
    std::vector<NormalizerResult> output = added_vocabulary.FindSplits(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyFindSplitsLStrip(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, true, false, false, false)});
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  for (auto _ : state) {
    std::vector<NormalizerResult> output = added_vocabulary.FindSplits(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyFindSplitsRStrip(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, false, true, false, false)});
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  for (auto _ : state) {
    std::vector<NormalizerResult> output = added_vocabulary.FindSplits(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_AddedVocabularyFindSplitsLStripAndRStrip(
    benchmark::State& state) { // NOLINT
  AddedVocabulary added_vocabulary(
      {AddedToken(0, "India", false, true, true, false, false)});
  NormalizerResult input =
      NormalizerResult(icu::UnicodeString("Capital of India is [MASK]"));
  for (auto _ : state) {
    std::vector<NormalizerResult> output = added_vocabulary.FindSplits(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_AddedVocabularyIsSpecialTokenTrue)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyIsSpecialTokenFalse)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyFindSplitsSpecialToken)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyFindSplitsSingleWord)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyFindSplitsLStrip)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyFindSplitsRStrip)->ThreadPerCpu();
BENCHMARK(BM_AddedVocabularyFindSplitsLStripAndRStrip)->ThreadPerCpu();
