import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer


class LLMPredictor:
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
                    "You are a senior Python engineer. "
                    "Answer using ONLY the provided code context."
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
    code_docs = [
        """
        def compute_mean(df, col):
            return df[col].mean()
        """,
            """
        df.groupby(["models", "LZ"])["gain"].mean().reset_index()
        """,
            """
        df["hour"] = df["INIT_DATE_TIME"].dt.hour
        """,
            """
        df.sort_values("day_ahead", ascending=False)
        """
        ]

    llm = LLMPredictor(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            lora_path="models/lora"
        )

    llm.set_documents(code_docs)

    answer = llm.generate_response_rag(
            "Compute the mean and std of power_OBS grouped by LZ and sort  by day_ahead?"
        )

    print(answer)