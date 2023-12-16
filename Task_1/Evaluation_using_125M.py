import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

# Loading tokenizer and original GPT-2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
original_model = GPT2LMHeadModel.from_pretrained("gpt2")

# I tried usingthe same hyperparameters used in my model to simulate a similar environment
custom_config = GPT2Config(
    n_head=6,
    n_layer=6,
    d_model=128,
    d_ffn=128,
    block_size=256,
)

# Creating a new instance of the GPT-2 model with custom hyperparameters
custom_model = GPT2LMHeadModel(config=custom_config)
shared_state_dict = {k: v for k, v in original_model.state_dict().items() if k in custom_model.state_dict()}
custom_model.load_state_dict(shared_state_dict, strict=False)

# Reading the content
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenizing the input text
tokenized_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

# Model generating text
generated_text = custom_model.generate(**tokenized_inputs, max_length=1025, num_beams=5, temperature=0.7, top_k=50, top_p=0.95, no_repeat_ngram_size=2, num_return_sequences=1)

# Decoding and printing the generated text
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(decoded_text)
