import json
import csv
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def calculate_similarity(expected, generated):
    """Calculate similarity between expected and generated strings using SequenceMatcher."""
    return SequenceMatcher(None, expected, generated).ratio()

class BaseModelPredictor:
    def __init__(self, base_model="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        self.base_model = base_model
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()

    def _initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _initialize_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32 if self.device == "mps" else torch.float16,
            trust_remote_code=True,
        )
        model.to(self.device)
        model.eval()
        return model

    def generate_response(self, messages, max_new_tokens=80, do_sample=False):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode ONLY the newly generated tokens (avoids echoing the whole prompt)
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()