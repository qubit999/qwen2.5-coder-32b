from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "./Qwen2.5-Coder-32B-Instruct"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
        self.tokenizer_code = AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )
        self.model_code = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
        ).eval()

    def predict(
        self,
        prompt: str = Input(description="Question", default="Write a hello world program in Python."),
    ) -> str:
        """Run a single prediction on the model"""
        prompt = prompt
        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=4096,
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
        model_inputs = self.tokenizer_code([prompt], return_tensors="pt", padding=True).to(self.model_code.device)
        generated_ids = self.model_code.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=4096,
            do_sample=True
        )[0]
        generated_ids = self.model_code.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)[0]
        output_text = self.tokenizer.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

        return output_text
        
if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    print(predictor.predict(prompt="Write a hello world program in Python"))
    print(predictor.predict_code(prompt="if True: print('Hello, World!')"))