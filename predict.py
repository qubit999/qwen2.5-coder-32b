import os
import torch

os.getenv("TORCH_USE_CUDA_DSA", "1")

from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "./Qwen2.5-Coder-32B-Instruct"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()

    def predict(
        self,
        prompt: str = Input(description="User Prompt", default="Write a hello world program in Python."),
        system_prompt : str = Input(description="System Prompt", default="You are Qwen. You are a helpful assistant."),
        max_new_tokens : int = Input(description="Max New Tokens", default=4096),
        min_new_tokens : int = Input(description="Min New Tokens", default=1),
        temperature : float = Input(description="Temperature", default=0.7),
        top_k : int = Input(description="Top K", default=50),
        top_p : float = Input(description="Top P", default=0.9),
        repetition_penalty : float = Input(description="Repetition Penalty", default=1.0),
        do_sample : bool = Input(description="Do Sample", default=True),
    ) -> str:
        """Run a single prediction on the model"""
        if(isinstance(prompt, str)):
            prompt = prompt
        else:
            prompt = prompt.default
        if(isinstance(system_prompt, str)):
            system_prompt = system_prompt
        else:
            system_prompt = system_prompt.default
        if(isinstance(max_new_tokens, int)):
            max_new_tokens = max_new_tokens
        else:
            max_new_tokens = max_new_tokens.default
        if(isinstance(min_new_tokens, int)):
            min_new_tokens = min_new_tokens
        else:
            min_new_tokens = min_new_tokens.default
        if(isinstance(temperature, float)):
            temperature = temperature
        else:
            temperature = temperature.default
        if(isinstance(top_k, int)):
            top_k = top_k
        else:
            top_k = top_k.default
        if(isinstance(top_p, float)):
            top_p = top_p
        else:
            top_p = top_p.default
        if(isinstance(repetition_penalty, float)):
            repetition_penalty = repetition_penalty
        else:
            repetition_penalty = repetition_penalty.default
        if(isinstance(do_sample, bool)):
            do_sample = do_sample
        else:
            do_sample = do_sample.default

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def predict_code(
        self,
        prompt: str = Input(description="Code", default="def hello_world():"),
    ) -> str:
        """Run a single prediction on the model"""
        model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=4096,
            do_sample=True,
        )
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return output_text

    def cleanup(self):
        """Cleanup after each prediction to save memory"""
        if self.model is not None:
            self.model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        # Additional cleanup if necessary

if __name__ == "__main__":
    run_bool = True
    if run_bool:
        predictor = Predictor()
        predictor.setup()
        print(predictor.predict())
        predictor.cleanup()