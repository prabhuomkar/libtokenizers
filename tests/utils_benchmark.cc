// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "tokenizers/common.h"
#include "tokenizers/utils.h"

using tokenizers::Encoding;
using tokenizers::Padding;
using tokenizers::PaddingDirection;
using tokenizers::PaddingStrategy;
using tokenizers::PadEncoding;
using tokenizers::TruncateEncoding;
using tokenizers::Truncation;
using tokenizers::TruncationDirection;
using tokenizers::TruncationStrategy;

static void BM_TruncateEncodingGreaterMaxLength(
    benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  for (auto _ : state) {
    TruncateEncoding(&input, 10, 2, TruncationDirection::kRight);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_TruncateEncodingMaxLengthZero(
    benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  for (auto _ : state) {
    TruncateEncoding(&input, 0, 2, TruncationDirection::kRight);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_TruncateEncodingTruncateRight(
    benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  for (auto _ : state) {
    benchmark::DoNotOptimize(input);
  }
}

static void BM_TruncateEncodingTruncateLeft(benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
                 {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
                 {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1});
  for (auto _ : state) {
    TruncateEncoding(&input, 3, 2, TruncationDirection::kLeft);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_TruncationMaxLengthZero(benchmark::State& state) { // NOLINT
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 0, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {}, {0, 0, 0, 0, 0},
               {1, 1, 1, 1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = truncation.TruncateEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TruncationStrategyLongestFirst(
    benchmark::State& state) { // NOLINT
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kLongestFirst, 6, 2);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5}, {0, 0, 0, 0, 0}, {"a", "b", "c", "d", "e"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}, {0, 1, 2, 3, 4},
               {0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}),
      Encoding({6, 7, 8, 9}, {0, 0, 0, 0}, {"f", "g", "h", "i"},
               {{10, 11}, {12, 13}, {14, 15}, {16, 17}}, {5, 6, 7, 8},
               {0, 0, 0, 0}, {1, 1, 1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = truncation.TruncateEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TruncationStrategyOnlyFirst(benchmark::State& state) { // NOLINT
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlyFirst, 6, 0);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3, 4, 5, 6, 7}, {0, 0, 0, 0, 0, 0, 0},
               {"a", "b", "c", "d", "e", "f", "g"},
               {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}},
               {0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0},
               {1, 1, 1, 1, 1, 1, 1}),
      Encoding({8, 9, 10}, {0, 0, 0}, {"h", "i", "j"},
               {{14, 15}, {16, 17}, {18, 19}}, {7, 8, 9}, {0, 0, 0},
               {1, 1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = truncation.TruncateEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TruncationStrategyOnlySecond(benchmark::State& state) { // NOLINT
  Truncation truncation(TruncationDirection::kRight,
                        TruncationStrategy::kOnlySecond, 8, 3);
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding(
          {4, 5, 6, 7, 8, 9, 10}, {0, 0, 0, 0, 0, 0, 0},
          {"d", "e", "f", "g", "h", "i", "j"},
          {{6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19}},
          {3, 4, 5, 6, 7, 8, 9}, {0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = truncation.TruncateEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_PadEncodingGreaterTargetLength(
    benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  for (auto _ : state) {
    PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_PadEncodingPadLeft(benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  for (auto _ : state) {
    PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kLeft);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_PadEncodingPadRight(benchmark::State& state) { // NOLINT
  Encoding input({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"},
                 {{0, 1}, {2, 3}, {4, 5}}, {0, 1, 2}, {0, 0, 0}, {1, 1, 1});
  for (auto _ : state) {
    PadEncoding(&input, 5, 0, 1, "[PAD]", PaddingDirection::kRight);
    benchmark::DoNotOptimize(input);
  }
}

static void BM_PaddingStrategyBatchLongest(benchmark::State& state) { // NOLINT
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kBatchLongest, 0,
                  0, 0, 0, "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}}, {3, 4, 5, 6}, {0, 0, 0, 0},
               {1, 1, 1, 1}),
      Encoding({8, 9}, {0, 0}, {"h", "i"}, {{14, 15}, {16, 17}}, {7, 8}, {0, 0},
               {1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = padding.PadEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_PaddingStrategyFixed(benchmark::State& state) { // NOLINT
  Padding padding(PaddingDirection::kRight, PaddingStrategy::kFixed, 5, 0, 0, 0,
                  "[PAD]");
  std::vector<Encoding> input = {
      Encoding({1, 2, 3}, {0, 0, 0}, {"a", "b", "c"}, {{0, 1}, {2, 3}, {4, 5}},
               {0, 1, 2}, {0, 0, 0}, {1, 1, 1}),
      Encoding({4, 5, 6, 7}, {0, 0, 0, 0}, {"d", "e", "f", "g"},
               {{6, 7}, {8, 9}, {10, 11}, {12, 13}}, {3, 4, 5, 6}, {0, 0, 0, 0},
               {1, 1, 1, 1}),
      Encoding({8, 9}, {0, 0}, {"h", "i"}, {{14, 15}, {16, 17}}, {7, 8}, {0, 0},
               {1, 1})};
  for (auto _ : state) {
    std::vector<Encoding> output = padding.PadEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_TruncateEncodingGreaterMaxLength)->ThreadPerCpu();
BENCHMARK(BM_TruncateEncodingMaxLengthZero)->ThreadPerCpu();
BENCHMARK(BM_TruncateEncodingTruncateLeft)->ThreadPerCpu();
BENCHMARK(BM_TruncateEncodingTruncateRight)->ThreadPerCpu();
BENCHMARK(BM_TruncationMaxLengthZero)->ThreadPerCpu();
BENCHMARK(BM_TruncationStrategyLongestFirst)->ThreadPerCpu();
BENCHMARK(BM_TruncationStrategyOnlyFirst)->ThreadPerCpu();
BENCHMARK(BM_TruncationStrategyOnlySecond)->ThreadPerCpu();
BENCHMARK(BM_PadEncodingGreaterTargetLength)->ThreadPerCpu();
BENCHMARK(BM_PadEncodingPadLeft)->ThreadPerCpu();
BENCHMARK(BM_PadEncodingPadRight)->ThreadPerCpu();
BENCHMARK(BM_PaddingStrategyBatchLongest)->ThreadPerCpu();
BENCHMARK(BM_PaddingStrategyFixed)->ThreadPerCpu();
