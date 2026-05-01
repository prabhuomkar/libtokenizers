// Copyright 2025 Omkar Prabhu
#include "tokenizers/normalizer.h"

#include <unicode/normalizer2.h>
#include <unicode/schriter.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/ustring.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace tokenizers {

namespace normalizers {

NormalizerResult::NormalizerResult(const icu::UnicodeString& normalized,
                                   bool pre_normalized)
    : normalized(normalized), pre_normalized(pre_normalized) {}

Normalizer::Normalizer() {}

std::string Normalizer::NormalizeString(std::string input) { return ""; }

NormalizerResult Normalizer::Normalize(NormalizerResult input) { return input; }

NFCNormalizer::NFCNormalizer() {}

NormalizerResult NFCNormalizer::Normalize(NormalizerResult input) {
  UErrorCode error_code = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer =
      icu::Normalizer2::getNFCInstance(error_code);
  if (U_FAILURE(error_code) || !normalizer) {
    throw std::runtime_error(
        std::string("failed to get normalizer instance: ") +
        u_errorName(error_code));
  }

  icu::UnicodeString normalized;
  normalizer->normalize(input.normalized, normalized, error_code);
  if (U_FAILURE(error_code)) {
    throw std::runtime_error(std::string("failed to normalize string input: ") +
                             u_errorName(error_code));
  }

  input.normalized = normalized;
  return input;
}

std::string NFCNormalizer::NormalizeString(std::string input) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  NormalizerResult normalized = NormalizerResult(unicode_input);
  normalized = Normalize(normalized);
  std::string result;
  normalized.normalized.toUTF8String(result);
  return result;
}

BertNormalizer::BertNormalizer(bool clean_text, bool handle_chinese_chars,
                               bool strip_accents, bool lowercase)
    : clean_text_(clean_text),
      handle_chinese_chars_(handle_chinese_chars),
      strip_accents_(strip_accents),
      lowercase_(lowercase) {}

NormalizerResult BertNormalizer::Normalize(NormalizerResult input) {
  if (clean_text_) {
    doCleanText(&input);
  }
  if (handle_chinese_chars_) {
    doHandleChineseChars(&input);
  }
  if (strip_accents_) {
    doStripAccents(&input);
  }
  if (lowercase_) {
    doLowercase(&input);
  }
  return input;
}

std::string BertNormalizer::NormalizeString(std::string input) {
  icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
  NormalizerResult normalized = NormalizerResult(unicode_input);
  normalized = Normalize(normalized);
  std::string result;
  normalized.normalized.toUTF8String(result);
  return result;
}

void doCleanText(NormalizerResult* input) {
  icu::UnicodeString result;
  icu::StringCharacterIterator it(input->normalized);
  for (it.first(); it.hasNext();) {
    UChar32 c = it.next32PostInc();
    if (c == 0x0000 || c == 0xFFFD || isControl(c)) {
      continue;
    }
    result.append(isWhitespace(c) ? ' ' : c);
  }
  input->normalized = result;
}

void doHandleChineseChars(NormalizerResult* input) {
  icu::UnicodeString result;
  icu::StringCharacterIterator it(input->normalized);
  for (it.first(); it.hasNext();) {
    UChar32 c = it.next32PostInc();
    if (isChineseChar(c)) {
      result.append(' ');
      result.append(c);
      result.append(' ');
    } else {
      result.append(c);
    }
  }
  input->normalized = result;
}

void doStripAccents(NormalizerResult* input) {
  UErrorCode error_code = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer =
      icu::Normalizer2::getNFDInstance(error_code);
  if (U_FAILURE(error_code) || !normalizer) {
    throw std::runtime_error(
        std::string("failed to get normalizer instance: ") +
        u_errorName(error_code));
  }

  icu::UnicodeString normalized;
  normalizer->normalize(input->normalized, normalized, error_code);
  if (U_FAILURE(error_code)) {
    throw std::runtime_error(std::string("failed to normalize string input: ") +
                             u_errorName(error_code));
  }

  icu::UnicodeString result;
  icu::StringCharacterIterator remove_it(normalized);
  for (remove_it.first(); remove_it.hasNext();) {
    UChar32 c = remove_it.next32PostInc();
    if (u_charType(c) != U_NON_SPACING_MARK) {
      result.append(c);
    }
  }

  input->normalized = result;
}

void doLowercase(NormalizerResult* input) { input->normalized.toLower(); }

bool isControl(UChar32 c) {
  if (c == '\t' || c == '\n' || c == '\r')
    return false;
  int char_type = u_charType(c);
  return char_type == U_CONTROL_CHAR ||   // Cc
         char_type == U_FORMAT_CHAR ||    // Cf
         char_type == U_UNASSIGNED ||     // Cn
         char_type == U_PRIVATE_USE_CHAR; // Co
}

bool isWhitespace(UChar32 c) { return u_isWhitespace(c); }

bool isChineseChar(UChar32 c) {
  UBlockCode block = ublock_getCode(c);
  return (block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F ||
          block == UBLOCK_CJK_UNIFIED_IDEOGRAPHS_EXTENSION_G);
}

} // namespace normalizers

} // namespace tokenizers
