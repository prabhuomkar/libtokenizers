// Copyright 2025 Omkar Prabhu
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

enum class TruncationDirection { kLeft, kRight };

enum class TruncationStrategy { kLongestFirst, kOnlyFirst, kOnlySecond };

class Truncation {
 public:
  Truncation();
  Truncation(const TruncationDirection &direction,
             const TruncationStrategy &strategy, int max_length, int stride);
  std::vector<Encoding> TruncateEncodings(
      const std::vector<Encoding> &encodings);

 private:
  TruncationDirection direction_;
  TruncationStrategy strategy_;
  int max_length_;
  int stride_;
};

void TruncateEncoding(Encoding *encoding, int max_length, int stride,
                      TruncationDirection direction);

enum class PaddingDirection { kLeft, kRight };

enum class PaddingStrategy { kBatchLongest, kFixed };

class Padding {
 public:
  Padding();
  Padding(const PaddingDirection &direction, const PaddingStrategy &strategy,
          int strategy_size, int pad_to_multiple_of, int pad_id,
          int pad_type_id, const std::string &pad_token);
  std::vector<Encoding> PadEncodings(const std::vector<Encoding> &encodings);

 private:
  PaddingDirection direction_;
  PaddingStrategy strategy_;
  int strategy_size_;
  int pad_to_multiple_of_;
  int pad_id_;
  int pad_type_id_;
  std::string pad_token_;
};

void PadEncoding(Encoding *encoding, int target_length, int pad_id,
                 int pad_type_id, const std::string &pad_token,
                 PaddingDirection direction);
