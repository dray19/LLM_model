## LLM-Powered Electricity Trading Analytics (Python)

- This project is an LLM-powered electricity trading analytics toolkit built in Python.

- It allows users to:
	- Upload forecast datasets and trade result datasets
	- Ask questions in natural language
	- Automatically generate safe, runnable Pandas code
	- Perform fast analysis, diagnostics, and reporting on electricity trading data

- The goal is to bridge the gap between data, analysis, and decision-making, making advanced electricity trading analytics accessible without requiring deep Python or Pandas expertise.

### Training

- **training/finetune_lora.py**
	- This script fine-tunes a large language model using LoRA (Low-Rank Adaptation). Instead of updating all model weights, it trains a small set of adapter weights, making fine-tuning fast, memory-efficient, and practical on local machines
- **Base Model**
	- Qwen2.5-1.5B-Instruct 
		- 1.5-billion parameter
		- Qwen2.5-1.5B-Instruct is a lightweight, instruction-tuned large language model optimized for strong reasoning and code generation while remaining efficient enough to run locally on CPU or Apple Silicon. Its modern architecture and instruction alignment make it especially well suited for LoRA fine-tuning, domain adaptation, and building practical local AI applications.

### Work in Progress

- This repository is under active development. Planned improvements include:
	- Better error handling and guardrails for generated code
	- Performance and scalability improvements

- Expect breaking changes as the design evolves.

