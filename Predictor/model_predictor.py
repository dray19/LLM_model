import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class LLMPredictor:
    def __init__(self, base_model="Qwen/Qwen2.5-1.5B-Instruct", lora_path="models/lora", device=None):
        self.base_model = base_model
        self.lora_path = lora_path
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
        model = PeftModel.from_pretrained(model, self.lora_path)
        model.to(self.device)
        model.eval()

        # (Optional but recommended) merge LoRA for cleaner inference
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

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
    

if __name__ == "__main__":
    predictor = LLMPredictor()

    messages = [
    {
        "role": "system",
        "content": (
            "You are a code generator. Respond with ONLY valid Python code. "
            "No explanations. No markdown. No imports. NO sample data.\n\n"
            "Rules:\n"
            "- You MUST assume a pandas DataFrame named df already exists in memory and is the ONLY input dataset.\n"
            "- Generate code that operates ONLY on df or intermediate objects derived directly from df.\n"
            "- Do NOT reference external variables, files, paths, configs, or objects not derived from df.\n"
            "- Do NOT read from or write to disk.\n"
            "- Do NOT make network calls.\n"
            "- Do NOT use randomness or non-deterministic behavior.\n"
            "- Do NOT use unsafe operations (eval, exec, compile, ast, subprocess, os, shell commands).\n"
            "- Do NOT mutate df unless explicitly requested; prefer creating new objects.\n"
            "- Avoid chained assignment; use .loc for assignments.\n"
            "- Do NOT assume column dtypes; handle numeric vs non-numeric safely.\n"
            "- For groupby aggregations, use numeric_only=True when appropriate.\n"
            "- Guard against missing columns: if required columns are missing, assign result to "
            "a clear error string like \"ERROR: missing columns: ['col1', 'col2']\".\n"
            "- Always assign the final output to a variable named result.\n"
            "- Do NOT print unless explicitly requested.\n"
            "- Keep the code minimal, deterministic, and directly executable."
        ),
    },
    {
        "role": "user",
        "content": (
            "Compute the sum of power_OBS grouped by LZ.\n Columns: INIT_DATE_TIME, LZ, power_OBS, day_ahead, price, gain"
        ),
    },
]

    response = predictor.generate_response(messages)
    print(response)