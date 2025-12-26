import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/train.jsonl"
OUT_DIR = "models/lora"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Qwen often has no pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# IMPORTANT: keep fp32 on MPS for stability
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if DEVICE == "mps" else torch.float16,
    trust_remote_code=True,
)
model.to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA_PATH)

MAX_LEN = 384

def format_example(ex):
    # Build the chat WITHOUT the assistant answer (ends with assistant prompt)
    messages_prompt = [
        {"role": "system", "content": "You are a helpful data analysis assistant."},
        {"role": "user", "content": f"{ex['instruction']}\n{ex['input']}"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,   # leaves assistant "slot" open
    )

    # Full text = prompt + answer (+ eos)
    answer = ex["output"].strip()
    full_text = prompt_text + answer + tokenizer.eos_token

    # Tokenize full sequence
    tok = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_attention_mask=True,
    )

    # Tokenize prompt alone to get prompt length
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    labels = tok["input_ids"].copy()

    # Mask prompt tokens so loss only trains on the assistant answer
    labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

    # Mask padding tokens too (super important)
    attn = tok["attention_mask"]
    labels = [lab if m == 1 else -100 for lab, m in zip(labels, attn)]

    tok["labels"] = labels
    return tok

train_ds = dataset["train"].map(format_example, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,

    # IMPORTANT: do NOT use fp16 on MPS Trainer
    fp16=False,
    bf16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
)

trainer.train()
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Saved LoRA to: {OUT_DIR}")

