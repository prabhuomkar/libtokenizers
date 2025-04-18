// Copyright 2025 Omkar Prabhu
#include "tokenizers/tokenizer.h"

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizers/common.h"
#include "tokenizers/decoder.h"
#include "tokenizers/model.h"
#include "tokenizers/normalizer.h"
#include "tokenizers/post_processor.h"
#include "tokenizers/pre_tokenizer.h"
#include "tokenizers/utils.h"

using tokenizers::Encoding;
using tokenizers::Tokenizer;
using tokenizers::decoders::WordPieceDecoder;
using tokenizers::models::WordPiece;
using tokenizers::normalizers::BertNormalizer;
using tokenizers::post_processors::TemplateProcessing;
using tokenizers::post_processors::TemplateProcessor;
using tokenizers::pre_tokenizers::BertPreTokenizer;

std::string read_json_for_test(const std::string& filepath) {
  std::ifstream file(filepath);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void assertTokenizerValues(const Encoding& got, const Encoding& expected) {
  ASSERT_EQ(got.ids.size(), expected.ids.size());
  ASSERT_EQ(got.type_ids.size(), expected.type_ids.size());
  ASSERT_EQ(got.tokens.size(), expected.tokens.size());
  ASSERT_EQ(got.offsets.size(), expected.offsets.size());
  ASSERT_EQ(got.word_ids.size(), expected.word_ids.size());
  ASSERT_EQ(got.special_tokens_mask.size(),
            expected.special_tokens_mask.size());
  ASSERT_EQ(got.attention_mask.size(), expected.attention_mask.size());
  for (int i = 0; i < got.ids.size(); i++) {
    ASSERT_EQ(got.ids[i], expected.ids[i]);
    ASSERT_EQ(got.type_ids[i], expected.type_ids[i]);
    ASSERT_EQ(got.tokens[i], expected.tokens[i]);
    ASSERT_EQ(got.offsets[i], expected.offsets[i]);
    ASSERT_EQ(got.word_ids[i], expected.word_ids[i]);
    ASSERT_EQ(got.special_tokens_mask[i], expected.special_tokens_mask[i]);
    ASSERT_EQ(got.attention_mask[i], expected.attention_mask[i]);
  }
}

TEST(TokenizerTest, Error) {
  EXPECT_THROW({ auto tokenizer = Tokenizer(""); }, std::invalid_argument);
}

TEST(TokenizerTest, EncodeSingle) {
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1},    {"world", 2},
          {"!", 3},      {"i", 4},        {"'", 5},
          {"m", 6},      {"learning", 7}, {"bert", 8},
          {"-", 9},      {"based", 10},   {"nlp", 11},
          {"with", 12},  {"un", 13},      {"##affordable", 14},
          {"costs", 15}, {"in", 16},      {"sao", 17},
          {"paulo", 18}, {",", 19},       {"北", 20},
          {"京", 21},    {"大", 22},      {"学", 23},
          {"and", 24},   {"python", 25},  {"是", 26},
          {"一", 27},    {"种", 28},      {"编", 29},
          {"程", 30},    {"语", 31},      {"言", 32},
          {"❤️", 33},     {".", 34}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  Encoding expected_encoding = Encoding(
      {100, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
       13,  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 19, 24,
       25,  26, 27, 28, 29, 30, 31, 32, 0,  34, 101},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {"[CLS]", "hello",    "world",        "!",     "i",     "'",
       "m",     "learning", "bert",         "-",     "based", "nlp",
       "with",  "un",       "##affordable", "costs", "in",    "sao",
       "paulo", ",",        "北",           "京",    "大",    "学",
       ",",     "and",      "python",       "是",    "一",    "种",
       "编",    "程",       "语",           "言",    "[UNK]", ".",
       "[SEP]"},
      {{0, 0},   {0, 5},    {6, 11},    {11, 12},   {13, 14},   {14, 15},
       {15, 16}, {17, 25},  {26, 30},   {30, 31},   {31, 36},   {37, 40},
       {41, 45}, {46, 48},  {48, 58},   {59, 64},   {65, 67},   {68, 71},
       {72, 77}, {77, 78},  {79, 80},   {80, 81},   {81, 82},   {82, 83},
       {83, 84}, {85, 88},  {89, 95},   {95, 96},   {96, 97},   {97, 98},
       {98, 99}, {99, 100}, {100, 101}, {101, 102}, {103, 104}, {104, 105},
       {0, 0}},
      {std::nullopt, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,           10, 11,
       12,           12, 13, 14, 15, 16, 17, 18, 19, 20, 21,          22, 23,
       24,           25, 26, 27, 28, 29, 30, 31, 32, 33, std::nullopt},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      u8"Hello world! I'm learning BERT-based NLP with unaffordable costs in "
      u8"São Paulo, 北京大学, and Python是一种编程语言 ❤️.",
      true);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, EncodePair) {
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1}, {"world", 2},   {"!", 3},
          {"i", 4},      {"'", 5},     {"m", 6},       {"learning", 7},
          {"bert", 8},   {"-", 9},     {"based", 10},  {"nlp", 11},
          {"we", 12},    {"have", 13}, {"un", 14},     {"##affordable", 15},
          {"costs", 16}, {"in", 17},   {"sao", 18},    {"paulo", 19},
          {",", 20},     {"北", 21},   {"京", 22},     {"大", 23},
          {"学", 24},    {"and", 25},  {"python", 26}, {"是", 27},
          {"一", 28},    {"种", 29},   {"编", 30},     {"程", 31},
          {"语", 32},    {"言", 33},   {"❤️", 34},      {".", 35}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  Encoding expected_encoding = Encoding(
      {100, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 35, 101,
       12,  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 20,
       25,  26, 27, 28, 29, 30, 31, 32, 33, 0,  35, 101},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"[CLS]", "hello",    "world", "!",     "i",     "'",
       "m",     "learning", "bert",  "-",     "based", "nlp",
       ".",     "[SEP]",    "we",    "have",  "un",    "##affordable",
       "costs", "in",       "sao",   "paulo", ",",     "北",
       "京",    "大",       "学",    ",",     "and",   "python",
       "是",    "一",       "种",    "编",    "程",    "语",
       "言",    "[UNK]",    ".",     "[SEP]"},
      {{0, 0},   {0, 5},   {6, 11},  {11, 12}, {13, 14}, {14, 15}, {15, 16},
       {17, 25}, {26, 30}, {30, 31}, {31, 36}, {37, 40}, {40, 41}, {0, 0},
       {0, 2},   {3, 7},   {8, 10},  {10, 20}, {21, 26}, {27, 29}, {30, 33},
       {34, 39}, {39, 40}, {41, 42}, {42, 43}, {43, 44}, {44, 45}, {45, 46},
       {47, 50}, {51, 57}, {57, 58}, {58, 59}, {59, 60}, {60, 61}, {61, 62},
       {62, 63}, {63, 64}, {65, 66}, {66, 67}, {0, 0}},
      {std::nullopt, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
       std::nullopt, 0,  1,  2,  2,  3,  4,  5,  6,  7,  8,  9,  10,
       11,           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
       std::nullopt},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      std::make_pair(u8"Hello world! I'm learning BERT-based NLP.",
                     u8"We have unaffordable costs in São Paulo, 北京大学, and "
                     u8"Python是一种编程语言 ❤️."),
      true);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, DecodeSingle) {
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1},    {"world", 2},
          {"!", 3},      {"i", 4},        {"'", 5},
          {"m", 6},      {"learning", 7}, {"bert", 8},
          {"-", 9},      {"based", 10},   {"nlp", 11},
          {"with", 12},  {"un", 13},      {"##affordable", 14},
          {"costs", 15}, {"in", 16},      {"sao", 17},
          {"paulo", 18}, {",", 19},       {"北", 20},
          {"京", 21},    {"大", 22},      {"学", 23},
          {"and", 24},   {"python", 25},  {"是", 26},
          {"一", 27},    {"种", 28},      {"编", 29},
          {"程", 30},    {"语", 31},      {"言", 32},
          {"❤️", 33},     {".", 34}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  tokenizer.added_vocabulary = std::make_shared<tokenizers::AddedVocabulary>(
      std::vector<tokenizers::AddedToken>(
          {tokenizers::AddedToken(100, "[CLS]", true, false, false, true),
           tokenizers::AddedToken(101, "[SEP]", true, false, false, true)}));
  tokenizer.decoder = std::make_shared<WordPieceDecoder>();
  std::string expected_result =
      "hello world! i ' m learning bert - based nlp with unaffordable costs "
      "in sao paulo, 北 京 大 学, and python 是 一 种 编 程 语 言 [UNK].";
  std::string got_result =
      tokenizer.Decode({100, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13,  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 19, 24,
                        25,  26, 27, 28, 29, 30, 31, 32, 0,  34, 101});
  ASSERT_EQ(got_result, expected_result);
}

TEST(TokenizerTest, DecodePair) {
  Tokenizer tokenizer;
  tokenizer.normalizer =
      std::make_shared<BertNormalizer>(true, true, true, true);
  tokenizer.pre_tokenizer = std::make_shared<BertPreTokenizer>();
  tokenizer.model = std::make_shared<WordPiece>(
      std::unordered_map<std::string, int>{
          {"[UNK]", 0},  {"hello", 1}, {"world", 2},   {"!", 3},
          {"i", 4},      {"'", 5},     {"m", 6},       {"learning", 7},
          {"bert", 8},   {"-", 9},     {"based", 10},  {"nlp", 11},
          {"we", 12},    {"have", 13}, {"un", 14},     {"##affordable", 15},
          {"costs", 16}, {"in", 17},   {"sao", 18},    {"paulo", 19},
          {",", 20},     {"北", 21},   {"京", 22},     {"大", 23},
          {"学", 24},    {"and", 25},  {"python", 26}, {"是", 27},
          {"一", 28},    {"种", 29},   {"编", 30},     {"程", 31},
          {"语", 32},    {"言", 33},   {"❤️", 34},      {".", 35}},
      u8"[UNK]", u8"##", 100);
  tokenizer.post_processor = std::make_shared<TemplateProcessing>(
      std::vector<TemplateProcessor>(
          {TemplateProcessor("SpecialToken", 0, "[CLS]"),
           TemplateProcessor("Sequence", 0, "A"),
           TemplateProcessor("SpecialToken", 0, "[SEP]")}),
      std::vector<TemplateProcessor>({
          TemplateProcessor("SpecialToken", 0, "[CLS]"),
          TemplateProcessor("Sequence", 0, "A"),
          TemplateProcessor("SpecialToken", 0, "[SEP]"),
          TemplateProcessor("Sequence", 1, "B"),
          TemplateProcessor("SpecialToken", 1, "[SEP]"),
      }),
      std::unordered_map<std::string, int>({{"[CLS]", 100}, {"[SEP]", 101}}));
  tokenizer.added_vocabulary = std::make_shared<tokenizers::AddedVocabulary>(
      std::vector<tokenizers::AddedToken>(
          {tokenizers::AddedToken(100, "[CLS]", true, false, false, true),
           tokenizers::AddedToken(101, "[SEP]", true, false, false, true)}));
  tokenizer.decoder = std::make_shared<WordPieceDecoder>();
  std::string expected_result =
      "hello world! i ' m learning bert - based nlp. we have unaffordable "
      "costs in sao paulo, 北 京 大 学, and python 是 一 种 编 程 语 言 [UNK].";
  std::string got_result = tokenizer.Decode(
      {100, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 35, 101,
       12,  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 20,
       25,  26, 27, 28, 29, 30, 31, 32, 33, 0,  35, 101});
  ASSERT_EQ(got_result, expected_result);
}

TEST(TokenizerTest, InitFromConfig) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer(config);
  ASSERT_EQ(tokenizer.version, "1.0");
  ASSERT_TRUE(tokenizer.normalizer != nullptr);
  auto normalizer =
      std::dynamic_pointer_cast<BertNormalizer>(tokenizer.normalizer);
  ASSERT_TRUE(normalizer != nullptr);
  ASSERT_TRUE(tokenizer.pre_tokenizer != nullptr);
  auto pre_tokenizer =
      std::dynamic_pointer_cast<BertPreTokenizer>(tokenizer.pre_tokenizer);
  ASSERT_TRUE(pre_tokenizer != nullptr);
  ASSERT_TRUE(tokenizer.model != nullptr);
  auto model = std::dynamic_pointer_cast<WordPiece>(tokenizer.model);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(tokenizer.post_processor != nullptr);
  auto post_processor =
      std::dynamic_pointer_cast<TemplateProcessing>(tokenizer.post_processor);
  ASSERT_TRUE(post_processor != nullptr);
  auto decoder = std::dynamic_pointer_cast<WordPieceDecoder>(tokenizer.decoder);
  ASSERT_TRUE(decoder != nullptr);
}

TEST(TokenizerTest, EncodeSingleFromConfigAddSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  Encoding expected_encoding = Encoding(
      {101,  7592,  2088, 999,  1045,  1005, 1049, 4083, 14324, 1011,
       2241, 17953, 2361, 2007, 14477, 4246, 8551, 3085, 5366,  1999,
       7509, 9094,  1010, 1781, 1755,  1810, 1817, 1010, 1998,  18750,
       100,  1740,  100,  100,  100,   100,  100,  1012, 102},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {"[CLS]",    "hello",  "world", "!",      "i",     "'",     "m",
       "learning", "bert",   "-",     "based",  "nl",    "##p",   "with",
       "una",      "##ff",   "##ord", "##able", "costs", "in",    "sao",
       "paulo",    ",",      "北",    "京",     "大",    "学",    ",",
       "and",      "python", "[UNK]", "一",     "[UNK]", "[UNK]", "[UNK]",
       "[UNK]",    "[UNK]",  ".",     "[SEP]"},
      {{0, 0},     {0, 5},     {6, 11},  {11, 12}, {13, 14},  {14, 15},
       {15, 16},   {17, 25},   {26, 30}, {30, 31}, {31, 36},  {37, 39},
       {39, 40},   {41, 45},   {46, 49}, {49, 51}, {51, 54},  {54, 58},
       {59, 64},   {65, 67},   {68, 71}, {72, 77}, {77, 78},  {79, 80},
       {80, 81},   {81, 82},   {82, 83}, {83, 84}, {85, 88},  {89, 95},
       {95, 96},   {96, 97},   {97, 98}, {98, 99}, {99, 100}, {100, 101},
       {101, 102}, {102, 103}, {0, 0}},
      {std::nullopt, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10,
       11,           12, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21,           22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, std::nullopt},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      u8"Hello world! I'm learning BERT-based NLP with unaffordable costs in "
      u8"São Paulo, 北京大学, and Python是一种编程语言.",
      true);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, EncodePairFromConfigAddSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  Encoding expected_encoding = Encoding(
      {101,   7592, 2088, 999,  1045, 1005, 1049,  4083, 14324, 1011, 2241,
       17953, 2361, 1012, 102,  2057, 2031, 14477, 4246, 8551,  3085, 5366,
       1999,  7509, 9094, 1010, 1781, 1755, 1810,  1817, 1010,  1998, 18750,
       100,   1740, 100,  100,  100,  100,  100,   1012, 102},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"[CLS]",    "hello", "world", "!",     "i",      "'",     "m",
       "learning", "bert",  "-",     "based", "nl",     "##p",   ".",
       "[SEP]",    "we",    "have",  "una",   "##ff",   "##ord", "##able",
       "costs",    "in",    "sao",   "paulo", ",",      "北",    "京",
       "大",       "学",    ",",     "and",   "python", "[UNK]", "一",
       "[UNK]",    "[UNK]", "[UNK]", "[UNK]", "[UNK]",  ".",     "[SEP]"},
      {{0, 0},   {0, 5},   {6, 11},  {11, 12}, {13, 14}, {14, 15}, {15, 16},
       {17, 25}, {26, 30}, {30, 31}, {31, 36}, {37, 39}, {39, 40}, {40, 41},
       {0, 0},   {0, 2},   {3, 7},   {8, 11},  {11, 13}, {13, 16}, {16, 20},
       {21, 26}, {27, 29}, {30, 33}, {34, 39}, {39, 40}, {41, 42}, {42, 43},
       {43, 44}, {44, 45}, {45, 46}, {47, 50}, {51, 57}, {57, 58}, {58, 59},
       {59, 60}, {60, 61}, {61, 62}, {62, 63}, {63, 64}, {64, 65}, {0, 0}},
      {std::nullopt,
       0,
       1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9,
       10,
       10,
       11,
       std::nullopt,
       0,
       1,
       2,
       2,
       2,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9,
       10,
       11,
       12,
       13,
       14,
       15,
       16,
       17,
       18,
       19,
       20,
       21,
       22,
       std::nullopt},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      std::make_pair(u8"Hello world! I'm learning BERT-based NLP.",
                     u8"We have unaffordable costs in São Paulo, 北京大学, and "
                     u8"Python是一种编程语言."),
      true);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, EncodeSingleFromConfigNoSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  Encoding expected_encoding = Encoding(
      {7592,  2088, 999,  1045,  1005, 1049, 4083, 14324, 1011,  2241,
       17953, 2361, 2007, 14477, 4246, 8551, 3085, 5366,  1999,  7509,
       9094,  1010, 1781, 1755,  1810, 1817, 1010, 1998,  18750, 100,
       1740,  100,  100,  100,   100,  100,  1012},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {"hello",  "world", "!",      "i",     "'",     "m",     "learning",
       "bert",   "-",     "based",  "nl",    "##p",   "with",  "una",
       "##ff",   "##ord", "##able", "costs", "in",    "sao",   "paulo",
       ",",      "北",    "京",     "大",    "学",    ",",     "and",
       "python", "[UNK]", "一",     "[UNK]", "[UNK]", "[UNK]", "[UNK]",
       "[UNK]",  "."},
      {{0, 5},    {6, 11},  {11, 12}, {13, 14},  {14, 15},   {15, 16},
       {17, 25},  {26, 30}, {30, 31}, {31, 36},  {37, 39},   {39, 40},
       {41, 45},  {46, 49}, {49, 51}, {51, 54},  {54, 58},   {59, 64},
       {65, 67},  {68, 71}, {72, 77}, {77, 78},  {79, 80},   {80, 81},
       {81, 82},  {82, 83}, {83, 84}, {85, 88},  {89, 95},   {95, 96},
       {96, 97},  {97, 98}, {98, 99}, {99, 100}, {100, 101}, {101, 102},
       {102, 103}},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 10, 11,
       12, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      u8"Hello world! I'm learning BERT-based NLP with unaffordable costs in "
      u8"São Paulo, 北京大学, and Python是一种编程语言.",
      false);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, EncodePairFromConfigNoSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  Encoding expected_encoding = Encoding(
      {7592,  2088, 999,  1045, 1005, 1049,  4083, 14324, 1011, 2241,
       17953, 2361, 1012, 2057, 2031, 14477, 4246, 8551,  3085, 5366,
       1999,  7509, 9094, 1010, 1781, 1755,  1810, 1817,  1010, 1998,
       18750, 100,  1740, 100,  100,  100,   100,  100,   1012},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {"hello", "world", "!",      "i",     "'",      "m",     "learning",
       "bert",  "-",     "based",  "nl",    "##p",    ".",     "we",
       "have",  "una",   "##ff",   "##ord", "##able", "costs", "in",
       "sao",   "paulo", ",",      "北",    "京",     "大",    "学",
       ",",     "and",   "python", "[UNK]", "一",     "[UNK]", "[UNK]",
       "[UNK]", "[UNK]", "[UNK]",  "."},
      {{0, 5},   {6, 11},  {11, 12}, {13, 14}, {14, 15}, {15, 16}, {17, 25},
       {26, 30}, {30, 31}, {31, 36}, {37, 39}, {39, 40}, {40, 41}, {0, 2},
       {3, 7},   {8, 11},  {11, 13}, {13, 16}, {16, 20}, {21, 26}, {27, 29},
       {30, 33}, {34, 39}, {39, 40}, {41, 42}, {42, 43}, {43, 44}, {44, 45},
       {45, 46}, {47, 50}, {51, 57}, {57, 58}, {58, 59}, {59, 60}, {60, 61},
       {61, 62}, {62, 63}, {63, 64}, {64, 65}},
      {0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 10, 11, 0,  1,  2,  2,  2,  2, 3,
       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  Encoding got_encoding = tokenizer.Encode(
      std::make_pair(u8"Hello world! I'm learning BERT-based NLP.",
                     u8"We have unaffordable costs in São Paulo, 北京大学, and "
                     u8"Python是一种编程语言."),
      false);
  assertTokenizerValues(got_encoding, expected_encoding);
}

TEST(TokenizerTest, DecodeSingleFromConfigSkipSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::string expected_result =
      "hello world! i ' m learning bert - based nlp with unaffordable costs in "
      "sao paulo, 北 京 大 学, and python 一.";
  std::string got_result = tokenizer.Decode(
      {101,  7592,  2088, 999,  1045,  1005, 1049, 4083, 14324, 1011,
       2241, 17953, 2361, 2007, 14477, 4246, 8551, 3085, 5366,  1999,
       7509, 9094,  1010, 1781, 1755,  1810, 1817, 1010, 1998,  18750,
       100,  1740,  100,  100,  100,   100,  100,  1012, 102});
  ASSERT_EQ(got_result, expected_result);
}

TEST(TokenizerTest, DecodePairFromConfigSkipSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::string expected_result =
      "hello world! i ' m learning bert - based nlp. we have unaffordable "
      "costs in sao paulo, 北 京 大 学, and python 一.";
  std::string got_result = tokenizer.Decode(
      {101,   7592, 2088, 999,  1045, 1005, 1049,  4083, 14324, 1011, 2241,
       17953, 2361, 1012, 102,  2057, 2031, 14477, 4246, 8551,  3085, 5366,
       1999,  7509, 9094, 1010, 1781, 1755, 1810,  1817, 1010,  1998, 18750,
       100,   1740, 100,  100,  100,  100,  100,   1012, 102});
  ASSERT_EQ(got_result, expected_result);
}

TEST(TokenizerTest, DecodeSingleFromConfigIncludeSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::string expected_result =
      "[CLS] hello world! i ' m learning bert - based nlp with unaffordable "
      "costs in "
      "sao paulo, 北 京 大 学, and python [UNK] 一 [UNK] [UNK] [UNK] [UNK] "
      "[UNK]. [SEP]";
  std::string got_result = tokenizer.Decode(
      {101,  7592,  2088, 999,  1045,  1005, 1049, 4083, 14324, 1011,
       2241, 17953, 2361, 2007, 14477, 4246, 8551, 3085, 5366,  1999,
       7509, 9094,  1010, 1781, 1755,  1810, 1817, 1010, 1998,  18750,
       100,  1740,  100,  100,  100,   100,  100,  1012, 102},
      false);
  ASSERT_EQ(got_result, expected_result);
}

TEST(TokenizerTest, DecodePairFromConfigIncludeSpecialTokens) {
  std::string config =
      read_json_for_test("../../scripts/tokenizers/bert-base-uncased.json");
  Tokenizer tokenizer = Tokenizer(config);
  std::string expected_result =
      "[CLS] hello world! i ' m learning bert - based nlp. [SEP] we have "
      "unaffordable "
      "costs in sao paulo, 北 京 大 学, and python [UNK] 一 [UNK] [UNK] [UNK] "
      "[UNK] [UNK]. [SEP]";
  std::string got_result = tokenizer.Decode(
      {101,   7592, 2088, 999,  1045, 1005, 1049,  4083, 14324, 1011, 2241,
       17953, 2361, 1012, 102,  2057, 2031, 14477, 4246, 8551,  3085, 5366,
       1999,  7509, 9094, 1010, 1781, 1755, 1810,  1817, 1010,  1998, 18750,
       100,   1740, 100,  100,  100,  100,  100,   1012, 102},
      false);
  ASSERT_EQ(got_result, expected_result);
}
