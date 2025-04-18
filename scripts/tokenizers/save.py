import sys
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained(sys.argv[1])
tokenizer.save(sys.argv[2])
