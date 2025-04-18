// Copyright 2025 Omkar Prabhu
#include <benchmark/benchmark.h>

#include <string>
#include <vector>

#include "tokenizers/decoder.h"

using tokenizers::decoders::Decoder;
using tokenizers::decoders::WordPieceDecoder;

static void BM_WordPieceDecoderAllOps(benchmark::State& state) { // NOLINT
  WordPieceDecoder decoder;
  std::vector<std::string> input = {"##uelo", "Ara", "##új",
                                    "##o",    "No",  "##guera"};
  for (auto _ : state) {
    std::vector<std::string> output = decoder.DecodeChain(input);
    benchmark::DoNotOptimize(output);
  }
}

BENCHMARK(BM_WordPieceDecoderAllOps)->ThreadPerCpu();
