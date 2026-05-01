import sys
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(sys.argv[1])
output = tokenizer.encode(sys.argv[2])
print(f'ids: {output.ids}')
print(f'type_ids: {output.type_ids}')
print(f'tokens: {output.tokens}')
