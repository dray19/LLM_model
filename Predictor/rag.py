import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import json


class LLMPredictorRAG:
    def __init__(
        self,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_path="models/lora",
        device=None,
        embedding_model="all-MiniLM-L6-v2"
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.device = device if device else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # LLM
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model()

        # RAG components
        self.embedder = SentenceTransformer(embedding_model)
        self.documents = []
        self.doc_embeddings = None


    # --------------------------------------------------
    # Static Method for Loading JSONL
    # --------------------------------------------------
    @staticmethod
    def load_jsonl(file_path):
        docs = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # Assuming each line has a "code" or "text" field containing the document
                    docs.append(data.get("output", "").strip())
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
        return docs

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def _initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
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

        # Optional: merge LoRA
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        return model

    # --------------------------------------------------
    # RAG setup
    # --------------------------------------------------
    def set_documents(self, documents):
        """
        documents: List[str] (Python code snippets)
        """
        self.documents = documents
        self.doc_embeddings = self.embedder.encode(
            documents,
            normalize_embeddings=True
        )

    def _retrieve(self, query, k=3):
        if self.doc_embeddings is None:
            raise ValueError("No documents set. Call set_documents() first.")

        q_emb = self.embedder.encode(
            query,
            normalize_embeddings=True
        )
        scores = np.dot(self.doc_embeddings, q_emb)
        top_idx = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_idx]

    # --------------------------------------------------
    # Standard generation (unchanged)
    # --------------------------------------------------
    def generate_response(self, messages, max_new_tokens=80, do_sample=False):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True
        ).strip()

    # --------------------------------------------------
    # RAG generation (NEW)
    # --------------------------------------------------
    def generate_response_rag(
        self,
        question,
        k=3,
        max_new_tokens=120,
        do_sample=False
    ):
        retrieved_code = self._retrieve(question, k)
        context = "\n\n---\n\n".join(retrieved_code)

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
                )
            },
            {
                "role": "user",
                "content": (
                    f"Code context:\n{context}\n\n"
                f"Question:\n{question}"
                )
            }
        ]


        return self.generate_response(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )
    
if __name__ == "__main__":
    code_docs = LLMPredictorRAG.load_jsonl("data/train.jsonl")

    llm = LLMPredictorRAG()

    llm.set_documents(code_docs)

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
                )
            },
            {
                "role": "user",
                "content": (
                    "Filter rows where gain is less than 50"
                )
            }
        ]

    answer = llm.generate_response_rag(
        messages[1]['content']
        )
    response = answer.replace("```python", "").replace("```", "").strip()
    response = response.replace("result =", "").strip()

    print(response)