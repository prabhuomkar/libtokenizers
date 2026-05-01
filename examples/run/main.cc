// Copyright 2025 Omkar Prabhu
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "tokenizers/tokenizer.h"

using tokenizers::Encoding;
using tokenizers::Tokenizer;

template <typename T>
void printVec(const std::vector<T>& list) {
  std::cout << "[";
  for (size_t i = 0; i < list.size(); ++i) {
    if constexpr (std::is_same_v<T, std::string>) {
      std::cout << "'" << list[i] << "'";
    } else {
      std::cout << list[i];
    }
    if (i != list.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

std::string readTokenizerConfigJSON(std::string path) {
  std::ifstream file(path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

int main(int argc, char* argv[]) {
  Tokenizer tokenizer = Tokenizer(readTokenizerConfigJSON(argv[1]));
  std::string input = argv[2];
  Encoding encoding = tokenizer.Encode(input);

  std::cout << "ids: ";
  printVec(encoding.ids);
  std::cout << "type_ids: ";
  printVec(encoding.type_ids);
  std::cout << "tokens: ";
  printVec(encoding.tokens);

  return 0;
}
