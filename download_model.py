from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
cache = "./cache"

# Load tokenizer and model with optimizations
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache,
    torch_dtype=torch.float16,  # Load in half precision
    low_cpu_mem_usage=True,     # Reduce memory usage during loading
)

# Save locally
model.save_pretrained("./Qwen2.5-Coder-32B-Instruct")
tokenizer.save_pretrained("./Qwen2.5-Coder-32B-Instruct")