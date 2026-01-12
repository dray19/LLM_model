# LLM-Powered Electricity Trading Analytics (Python)

- This project is a demo of the work I completed to design and build a custom Large Language Model (LLM) that is better aligned with a company’s specific business needs in the electricity trading domain.

- The model was then integrated into a scalable application infrastructure (similar to the repo “LLM_Fastapi”) to support real-world use, including:
	- Uploading forecast datasets and trade result datasets
	- Asking questions in natural language
	- Automatically generating safe, runnable Pandas code
	- Performing fast analysis, diagnostics, and reporting on electricity trading data

- Overall, the goal is to bridge the gap between raw data, analytics, and decision-making, making advanced electricity trading analysis accessible to users without requiring deep expertise in Python or Pandas.

## Training

- ### training/finetune_lora.py
	- This script fine-tunes a large language model using LoRA (Low-Rank Adaptation). Instead of updating all model weights, it trains a small set of adapter weights, making fine-tuning fast, memory-efficient, and practical on local machines

- ### Training Data 
	- Training data was created in **create_data/create_train_test_datasets.py**
	- The model was fine-tuned on ~850 high-quality, instruction–response pairs.
  	- Instructions were intentionally close to real pandas code patterns used in the company’s datasets, including:
		- DataFrame filtering and sorting
		- Groupby and aggregation logic
		- Feature engineering and column transformations
		- Time-series and timestamp operations
		- Common data-analysis and ETL-style workflows
	- This focused instruction set helps the model generate practical, production-style pandas code that mirrors real internal analytics tasks rather than generic examples.
	- Three datasets were used: data/train.jsonl, data/test.jsonl, and data/unseen.jsonl.
		- The training and test sets contain pandas instruction patterns derived from the company’s real datasets.
		- The unseen set follows the same pandas-style patterns but includes small variations to evaluate generalization beyond the exact 
		training examples.

- ### Base Model
	- Qwen2.5-1.5B-Instruct 
		- 1.5-billion parameter
		- Qwen2.5-1.5B-Instruct is a lightweight, instruction-tuned large language model optimized for strong reasoning and code generation while remaining efficient enough to run locally on CPU or Apple Silicon. Its modern architecture and instruction alignment make it especially well suited for LoRA fine-tuning, domain adaptation, and building practical local AI applications.

- ### Training Notes
	- Uses a memory-efficient LoRA fine-tuning setup optimized for local CPU and Apple Silicon environments
	- Trains with small per-device batches and gradient accumulation to achieve a stable effective batch size without exceeding hardware limits
	- Applies conservative learning rates and limited epochs to adapt the model to pandas-style data workflows while minimizing overfitting

## Models
- ### Four Models Examined:

	- ***predictor/basemodel.py***
		- Model Name: base
		- Just using the the base model (Qwen/Qwen2.5-1.5B-Instruct), does not use the LoRA fine-tuning 

	- ***predictor/model_predictor.py***
		- Model Name: none (file names without a model name)
	  	- Used the base model with the LoRA fine tuning

	- ***predictor/rag.py***
		- Model Name: rag
		- Used the base model with the LoRA fine tuning and added a basic RAG (Retrieval-Augmented Generation) model
		- How this RAG model works wit our fine turned model
			```
			User Question
				↓
			Sentence Embedding (MiniLM)
				↓
			Vector Similarity Search (cosine via dot product)
				↓
			Top-K Relevant Code Snippets
				↓
			Prompt Construction (Context + Question)
				↓
			LoRA-fine-tuned Qwen LLM
				↓
			Generated Python Code
			```
		- Embedding model used ***all-MiniLM-L6-v2***
			- Converts text or code into 384-dimensional semantic vectors so that similar meanings are close together, enabling fast retrieval in a RAG system.
			- In the pipeline, it retrieves the most relevant past code examples for a question and supplies them as context to the LLM, helping ground and improve the generated output.

	- ***predictor/rag_multi_query.py***
		- Model Name: rag_MQ
		- Used the base model with the LoRA fine tuning and added a RAG (Retrieval-Augmented Generation) model with Multi-query expansion + re-ranking 
		- How this RAG model works wit our fine turned model
		```
		User Question
			↓
		Multi-Query Expansion
			↓
		Semantic Retrieval (SentenceTransformer)
			↓
		Score Aggregation + Re-Ranking
			↓
		Context Injection
			↓
		LoRA-Enhanced LLM
			↓
		Valid Python Code 
		```
		- Compared to a basic RAG, this system expands the user question into multiple deterministic paraphrases and then re-ranks retrieved documents by both semantic similarity and cross-query coverage.
		- Improves recall and prioritizes evidence that is consistently relevant across interpretations rather than just top-K similarity from a single query.

		- Embedding model used ***all-mpnet-base-v2***
			- It is built on MPNet (Masked and Permuted Language Modeling), a model trained to understand the meaning of sentences by learning which words belong together, and it has seen many examples of sentence pairs, which makes it very good at telling when two pieces of text mean the same thing and at finding and ranking relevant matches.
			- Compared to lighter models (e.g., MiniLM), it produces more accurate embeddings at the cost of slightly higher compute, which is ideal for RAG systems where retrieval quality matters more than speed.

## Predictions 

- Predictions were produced with the scripts 
	- test_llm_[model_name]_predictions.py 
	- test_llm_[model_name]_predictions_unseen.py.

## Analysis

- All analysis outputs are located in the ***Error_analy/*** directory.
- The two primary notebooks used to compare model_predictor.py, rag.py, and rag_multi_query.py are:
	- **Error_analy/error_results_rag_MQ_updated.ipynb**
	- **Error_analy/error_results_rag_MQ_updated_unceen.ipynb**
- Results from both notebooks show that ***rag.py*** and ***rag_multi_query.py*** consistently perform best among the evaluated model types.
- Similarity is calculated using the following formula:
	- 2 * (number of matching characters)/ (total characters in both strings)
	


