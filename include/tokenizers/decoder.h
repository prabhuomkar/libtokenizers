// Copyright 2025 Omkar Prabhu
#pragma once

namespace tokenizers {

namespace decoders {

class Decoder {
 public:
  Decoder();

 private:
};

// WordPieceDecoder
class WordPieceDecoder : public Decoder {
 public:
  WordPieceDecoder();
};

} // namespace decoders

} // namespace tokenizers
