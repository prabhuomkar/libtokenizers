// Copyright 2025 Omkar Prabhu
#include "tokenizers/utils.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "tokenizers/common.h"

Truncation::Truncation()
    : direction_(TruncationDirection::kRight),
      strategy_(TruncationStrategy::kLongestFirst),
      max_length_(512),
      stride_(0) {}

Truncation::Truncation(const TruncationDirection& direction,
                       const TruncationStrategy& strategy, int max_length,
                       int stride)
    : direction_(direction),
      strategy_(strategy),
      max_length_(max_length),
      stride_(stride) {}

void TruncateEncoding(Encoding* encoding, int max_length, int stride,
                      TruncationDirection direction) {
  int encoding_len = encoding->ids.size();

  if (max_length >= encoding_len) {
    return;
  }

  if (max_length == 0) {
    encoding->overflowing.emplace_back(*encoding);
    encoding->ids.clear();
    encoding->type_ids.clear();
    encoding->tokens.clear();
    encoding->offsets.clear();
    encoding->word_ids.clear();
    encoding->special_tokens_mask.clear();
    encoding->attention_mask.clear();
    return;
  }

  std::vector<std::pair<int, int>> ranges;
  int offset = max_length - stride;

  if (direction == TruncationDirection::kLeft) {
    for (int stop = encoding_len; stop > 0; stop -= offset) {
      int start = std::max(0, stop - max_length);
      ranges.emplace_back(start, stop);
      if (start == 0) {
        break;
      }
    }
  } else if (direction == TruncationDirection::kRight) {
    for (int start = 0; start < encoding_len; start += offset) {
      int stop = std::min(start + max_length, encoding_len);
      ranges.emplace_back(start, stop);
      if (stop == encoding_len) {
        break;
      }
    }
  }

  Encoding new_encoding;
  new_encoding.ids.assign(encoding->ids.begin() + ranges[0].first,
                          encoding->ids.begin() + ranges[0].second);
  new_encoding.type_ids.assign(encoding->type_ids.begin() + ranges[0].first,
                               encoding->type_ids.begin() + ranges[0].second);
  new_encoding.tokens.assign(encoding->tokens.begin() + ranges[0].first,
                             encoding->tokens.begin() + ranges[0].second);
  new_encoding.offsets.assign(encoding->offsets.begin() + ranges[0].first,
                              encoding->offsets.begin() + ranges[0].second);
  new_encoding.word_ids.assign(encoding->word_ids.begin() + ranges[0].first,
                               encoding->word_ids.begin() + ranges[0].second);
  new_encoding.special_tokens_mask.assign(
      encoding->special_tokens_mask.begin() + ranges[0].first,
      encoding->special_tokens_mask.begin() + ranges[0].second);
  new_encoding.attention_mask.assign(
      encoding->attention_mask.begin() + ranges[0].first,
      encoding->attention_mask.begin() + ranges[0].second);

  new_encoding.overflowing.reserve(ranges.size() - 1);
  for (int i = 1; i < ranges.size(); i++) {
    new_encoding.overflowing.emplace_back(Encoding(
        std::vector<int>(encoding->ids.begin() + ranges[i].first,
                         encoding->ids.begin() + ranges[i].second),
        std::vector<int>(encoding->type_ids.begin() + ranges[i].first,
                         encoding->type_ids.begin() + ranges[i].second),
        std::vector<std::string>(encoding->tokens.begin() + ranges[i].first,
                                 encoding->tokens.begin() + ranges[i].second),
        std::vector<std::pair<int, int>>(
            encoding->offsets.begin() + ranges[i].first,
            encoding->offsets.begin() + ranges[i].second),
        std::vector<std::optional<int>>(
            encoding->word_ids.begin() + ranges[i].first,
            encoding->word_ids.begin() + ranges[i].second),
        std::vector<int>(
            encoding->special_tokens_mask.begin() + ranges[i].first,
            encoding->special_tokens_mask.begin() + ranges[i].second),
        std::vector<int>(encoding->attention_mask.begin() + ranges[i].first,
                         encoding->attention_mask.begin() + ranges[i].second)));
  }
  *encoding = new_encoding;
}

std::vector<Encoding> Truncation::TruncateEncodings(
    const std::vector<Encoding>& encodings) {
  std::vector<Encoding> result = encodings;
  if (max_length_ == 0) {
    for (int i = 0; i < result.size(); i++) {
      TruncateEncoding(&result[i], max_length_, stride_, direction_);
    }
    return result;
  }

  int total_length =
      result[0].ids.size() + (result.size() > 1 ? result[1].ids.size() : 0);
  if (total_length <= max_length_) {
    return result;
  }
  int to_remove = total_length - max_length_;

  if (strategy_ == TruncationStrategy::kLongestFirst) {
    if (result.size() > 1) {
      int n1 = result[0].ids.size();
      int n2 = result[1].ids.size();
      int swap = false;
      if (n1 > n2) {
        swap = true;
        std::swap(n1, n2);
      }
      if (n1 > max_length_) {
        n2 = n1;
      } else {
        n2 = std::max(n1, max_length_ - n1);
      }
      if (n1 + n2 > max_length_) {
        n1 = max_length_ / 2;
        n2 = n1 + max_length_ % 2;
      }
      if (swap) {
        std::swap(n1, n2);
      }
      TruncateEncoding(&result[0], n1, stride_, direction_);
      TruncateEncoding(&result[1], n2, stride_, direction_);
    } else {
      TruncateEncoding(&result[0], total_length - to_remove, stride_,
                       direction_);
    }
  } else if (strategy_ == TruncationStrategy::kOnlyFirst ||
             strategy_ == TruncationStrategy::kOnlySecond) {
    int target_length = strategy_ == TruncationStrategy::kOnlyFirst
                            ? result[0].ids.size()
                            : result[1].ids.size();
    if (target_length > to_remove) {
      TruncateEncoding(
          strategy_ == TruncationStrategy::kOnlyFirst ? &result[0] : &result[1],
          target_length - to_remove, stride_, direction_);
    }
  }
  return result;
}

Padding::Padding()
    : direction_(PaddingDirection::kRight),
      strategy_(PaddingStrategy::kBatchLongest),
      strategy_size_(0),
      pad_to_multiple_of_(0),
      pad_id_(0),
      pad_type_id_(0),
      pad_token_("[PAD]") {}

Padding::Padding(const PaddingDirection& direction,
                 const PaddingStrategy& strategy, int strategy_size,
                 int pad_to_multiple_of, int pad_id, int pad_type_id,
                 const std::string& pad_token)
    : direction_(direction),
      strategy_(strategy),
      strategy_size_(strategy_size),
      pad_to_multiple_of_(pad_to_multiple_of),
      pad_id_(pad_id),
      pad_type_id_(pad_type_id),
      pad_token_(pad_token) {}

void PadEncoding(Encoding* encoding, int target_length, int pad_id,
                 int pad_type_id, const std::string& pad_token,
                 PaddingDirection direction) {
  for (Encoding& overflow_encoding : encoding->overflowing) {
    PadEncoding(&overflow_encoding, target_length, pad_id, pad_type_id,
                pad_token, direction);
  }

  if (encoding->ids.size() >= target_length) {
    return;
  }

  int pad_length = target_length - encoding->ids.size();

  if (direction == PaddingDirection::kLeft) {
    encoding->ids.insert(encoding->ids.begin(), pad_length, pad_id);
    encoding->type_ids.insert(encoding->type_ids.begin(), pad_length,
                              pad_type_id);
    encoding->tokens.insert(encoding->tokens.begin(), pad_length, pad_token);
    encoding->offsets.insert(encoding->offsets.begin(), pad_length,
                             std::make_pair(0, 0));
    encoding->word_ids.insert(encoding->word_ids.begin(), pad_length,
                              std::nullopt);
    encoding->special_tokens_mask.insert(encoding->special_tokens_mask.begin(),
                                         pad_length, 1);
    encoding->attention_mask.insert(encoding->attention_mask.begin(),
                                    pad_length, 0);
  } else {
    encoding->ids.insert(encoding->ids.end(), pad_length, pad_id);
    encoding->type_ids.insert(encoding->type_ids.end(), pad_length,
                              pad_type_id);
    encoding->tokens.insert(encoding->tokens.end(), pad_length, pad_token);
    encoding->offsets.insert(encoding->offsets.end(), pad_length,
                             std::make_pair(0, 0));
    encoding->word_ids.insert(encoding->word_ids.end(), pad_length,
                              std::nullopt);
    encoding->special_tokens_mask.insert(encoding->special_tokens_mask.end(),
                                         pad_length, 1);
    encoding->attention_mask.insert(encoding->attention_mask.end(), pad_length,
                                    0);
  }
}

std::vector<Encoding> Padding::PadEncodings(
    const std::vector<Encoding>& encodings) {
  if (encodings.empty()) {
    return {};
  }

  int pad_length =
      strategy_ == PaddingStrategy::kFixed
          ? strategy_size_
          : std::max_element(encodings.begin(), encodings.end(),
                             [](const Encoding& a, const Encoding& b) {
                               return a.ids.size() < b.ids.size();
                             })
                ->ids.size();

  if (pad_to_multiple_of_ > 0 && pad_length % pad_to_multiple_of_ > 0) {
    pad_length += pad_to_multiple_of_ - pad_length % pad_to_multiple_of_;
  }

  std::vector<Encoding> result = encodings;
  for (int i = 0; i < result.size(); i++) {
    PadEncoding(&result[i], pad_length, pad_id_, pad_type_id_, pad_token_,
                direction_);
  }

  return result;
}
