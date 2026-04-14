from tokenizers import Tokenizer

# Load tokenizer from JSON
with open("../../scripts/tokenizers/bert-base-uncased.json") as f:
    tokenizer = Tokenizer.from_str(f.read())

# Input text (same UTF-8 string)
text = (
    "Hello world! I'm learning BERT-based NLP with unaffordable costs in "
    "São Paulo, 北京大学, and Python是一种编程语言."
)

# Encode (true → add special tokens)
encoding = tokenizer.encode(text, add_special_tokens=True)

# Inspect output
print("Tokens:")
print(encoding.tokens)

print("\nIDs:")
print(encoding.ids)

print("\nOffsets:")
print(encoding.offsets)

print("\nType IDs:")
print(encoding.type_ids)

print("\nAttention Mask:")
print(encoding.attention_mask)