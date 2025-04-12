// Copyright 2025 Omkar Prabhu
#include "tokenizers/decoder.h"

#include <string>
#include <vector>

namespace tokenizers {

namespace decoders {

Decoder::Decoder() {}

std::vector<std::string> Decoder::DecodeChain(
    const std::vector<std::string> tokens) {
  return {};
}

WordPieceDecoder::WordPieceDecoder(const std::string& prefix, bool cleanup)
    : prefix_(prefix), cleanup_(cleanup) {}

void replace(std::string& input, const std::string& from,
             const std::string& to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = input.find(from, start_pos)) != std::string::npos) {
    input.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

void doCleanup(std::string* input) {
  replace(*input, " .", ".");
  replace(*input, " ?", "?");
  replace(*input, " !", "!");
  replace(*input, " ,", ",");
  replace(*input, " ' ", "'");
  replace(*input, " n't", "n't");
  replace(*input, " 'm", "'m");
  replace(*input, " do not", "don't");
  replace(*input, " 's", "'s");
  replace(*input, " 've", "'ve");
  replace(*input, " 're", "'re");
}

std::vector<std::string> WordPieceDecoder::DecodeChain(
    std::vector<std::string> tokens) {
  for (int i = 0; i < tokens.size(); i++) {
    std::string& token = tokens[i];
    if (i != 0) {
      if (token.rfind(prefix_, 0) == 0) {
        token = token.substr(prefix_.size());
      } else {
        token = " " + token;
      }
    }
    if (cleanup_) {
      doCleanup(&token);
    }
  }
  return tokens;
}

} // namespace decoders

} // namespace tokenizers
