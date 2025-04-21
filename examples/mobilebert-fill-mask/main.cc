// Copyright 2025 Omkar Prabhu
#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "tokenizers/tokenizer.h"

using tokenizers::Encoding;
using tokenizers::Tokenizer;

std::string readTokenizerConfigJSON(std::string path) {
  std::ifstream file(path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

int main(int argc, char* argv[]) {
  std::string input = "Capital of India is [MASK].";
  Tokenizer tokenizer = Tokenizer(readTokenizerConfigJSON(argv[1]));
  Encoding encoding = tokenizer.Encode(input);

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[2]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model" << std::endl;
    return -1;
  }
  std::cout << "model loaded successfully" << std::endl;

  torch::Tensor input_ids = torch::tensor(encoding.ids).unsqueeze(0);
  torch::Tensor attention_mask =
      torch::tensor(encoding.attention_mask).unsqueeze(0);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_ids);
  inputs.push_back(attention_mask);

  auto output = module.forward(inputs).toGenericDict();
  at::Tensor logits = output.at("logits").toTensor();

  int mask_index = -1;
  at::Tensor input_row = input_ids[0];
  auto input_accessor = input_row.accessor<int64_t, 1>();
  for (int i = 0; i < input_accessor.size(0); i++) {
    if (input_accessor[i] == 103) {
      mask_index = i;
      break;
    }
  }
  at::Tensor mask_logits = logits[0][mask_index];
  int top_k = 5;
  std::tuple<at::Tensor, at::Tensor> topk = torch::topk(mask_logits, top_k);
  at::Tensor topk_scores = std::get<0>(topk);
  at::Tensor topk_indices = std::get<1>(topk);
  std::cout << "Top " << top_k << " predictions for [MASK] token:" << std::endl;
  for (int i = 0; i < top_k; i++) {
    int token_id = topk_indices[i].item<int>();
    float score = topk_scores[i].item<float>();
    auto token = tokenizer.Decode({token_id});
    std::cout << "Token: " << token << ", Score: " << score << std::endl;
  }

  return 0;
}
