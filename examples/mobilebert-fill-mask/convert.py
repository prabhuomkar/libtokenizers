import torch
from transformers import MobileBertForMaskedLM, AutoTokenizer


model_name = "google/mobilebert-uncased"
model = MobileBertForMaskedLM.from_pretrained(model_name, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def save_model():
    model.eval()
    inputs = tokenizer("Capital of India is [MASK].", return_tensors="pt")
    traced_script_module = torch.jit.trace(
        model, (inputs.input_ids, inputs.attention_mask),
        strict=False
    )
    traced_script_module.save("mobilebert.pt")

def save_tokenizer():
    tokenizer.save_pretrained("tokenizer")

if __name__ == "__main__":
    save_model()
    save_tokenizer()
