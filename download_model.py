from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Make sure this is the correct model path
cache = "./cache"  # Make sure this is the correct cache path

tokenizer = AutoTokenizer.from_pretrained(model_name, chache_dir=cache)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache)

# Save locally
model.save_pretrained("./Qwen2.5-Coder-32B-Instruct")
tokenizer.save_pretrained("./Qwen2.5-Coder-32B-Instruct")