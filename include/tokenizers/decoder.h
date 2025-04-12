// Copyright 2025 Omkar Prabhu
#pragma once

#include <string>
#include <vector>

namespace tokenizers {

namespace decoders {

class Decoder {
 public:
  Decoder();
  virtual std::vector<std::string> DecodeChain(std::vector<std::string> tokens);
};

// WordPieceDecoder
class WordPieceDecoder : public Decoder {
 public:
  explicit WordPieceDecoder(const std::string& prefix = "##",
                            bool cleanup = true);
  std::vector<std::string> DecodeChain(
      std::vector<std::string> tokens) override;

 private:
  std::string prefix_;
  bool cleanup_;
};

void doCleanup(std::string* input);

} // namespace decoders

} // namespace tokenizers
