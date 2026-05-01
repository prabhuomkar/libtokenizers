// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizers/post_processor.h"

using tokenizers::Encoding;
using tokenizers::post_processors::PostProcessor;
using tokenizers::post_processors::TemplateProcessing;
using tokenizers::post_processors::TemplateProcessor;

static void BM_TemplateProcessorSingle(benchmark::State& state) { // NOLINT
  TemplateProcessing post_processor(
      {TemplateProcessor("SpecialToken", 0, "[CLS]"),
       TemplateProcessor("Sequence", 0, "A"),
       TemplateProcessor("SpecialToken", 0, "[SEP]")},
      {
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      },
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  std::vector<Encoding> input = {
      Encoding({200, 201}, {0, 0}, {"hello", "world"})};
  for (auto _ : state) {
    std::vector<Encoding> output = post_processor.ProcessEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

static void BM_TemplateProcessorPair(benchmark::State& state) { // NOLINT
  TemplateProcessing post_processor(
      {TemplateProcessor("SpecialToken", 0, "[CLS]"),
       TemplateProcessor("Sequence", 0, "A"),
       TemplateProcessor("SpecialToken", 0, "[SEP]")},
      {
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      },
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  std::vector<Encoding> input = {
      Encoding({200, 201}, {0, 0}, {"hello", "world"}),
      Encoding({300, 301}, {1, 1}, {"martin", "garrix"})};
  for (auto _ : state) {
    std::vector<Encoding> output = post_processor.ProcessEncodings(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_TemplateProcessorSingle)->ThreadPerCpu();
BENCHMARK(BM_TemplateProcessorPair)->ThreadPerCpu();
