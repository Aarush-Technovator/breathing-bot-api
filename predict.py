import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Predictor:
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("breathingbrushes_gpt")
        self.model = AutoModelForCausalLM.from_pretrained("breathingbrushes_gpt", torch_dtype=torch.float16)
        self.model.eval()

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
