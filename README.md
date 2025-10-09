# LiveCLKTBench 
Towards Reliable Evaluation of Cross-Lingual Knowledge Transfer in Multilingual LLMs


## Installation
Clone the repository and install the required dependencies:
```
    pip install -r requirements.txt
```


## Benchmark Generation Pipeline
LiveCLKTBench provides a fully automated, refreshable benchmark generation pipeline for evaluating cross-lingual knowledge transfer (CL-KT) in multilingual large language models (LLMs).

You can generate benchmarks for various domains (e.g., sports, music, movies) and multiple languages (e.g., en, ja, fr, es, zh).


## Step 1. Collect Valid Knowledge Entities
This step identifies valid knowledge entities within a given domain.
```
PYTHONPATH=lib python3 data_generation/0_collect_entity.py \
    --domain sports \
    --start_str 2025-04-01 \
    --end_str 2025-06-30 \
    --output_dir data/entities \
    --max_entity 30
```

Arguments:
```
--domain: Target domain (e.g., sports, music, movies)
--start_str, --end_str: Time range for entity extraction (YYYY-MM-DD)
--max_entity: Maximum number of entities to collect
```



## Step 2. Generate Training Documents
Generates context documents for each collected entity, which will serve as model pretraining data.
```
PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain sports \
    --entity_file data/entities/sports/2025-07-01_2025-09-30.json \
    --output_dir data/train_docs/sports \
    --test_languages en ja fr es zh
```



## Step 3. Generate Factual QA Pairs
Creates factual, document-grounded multiple-choice questions for each entity.
```
PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain sports \
    --training_docs_dir data/train_docs/sports/2025-07-01_2025-09-30 \
    --output_dir data/factQA/sports \
    --qlangs en ja fr es zh
```



## Step 4. Assemble the LiveCLKTBench Benchmark

Combines factual QA and document data to construct the final benchmark split into train/validation/test sets.
```
PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir data/factQA/sports/2025-07-01_2025-09-30 \
    --training_docs_dir data/train_docs/sports/2025-07-01_2025-09-30 \
    --output_dir data/benchmark/sports \
    --test_languages en ja fr es zh \
    --val_ratio 0.2
```



## 🧠 Demo Experiment

This section demonstrates how to conduct Continual Pretraining (CPT) and Cross-Lingual Evaluation using the generated benchmark.

1️⃣ Continual Pretraining (CPT)
Example: source language = English
```
PYTHONPATH=lib python3 demo_experiment/cpt.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_file data/benchmark/sports/en/train_doc.jsonl \
    --val_file data/benchmark/sports/en/train_doc.jsonl \
    --output_dir models/sports/en-ja-fr-es-zh_2025-07-01_2025-09-30/en/Qwen2.5-3B-Instruct \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --rank 16 \
    --alpha 32 \
    --dropout 0.1
```


2️⃣ Inference
Example: evaluate source-trained model (en) on test languages = ja, fr, es, zh.
```
PYTHONPATH=lib python3 demo_experiment/inference.py \
    --model_dir models/sports/en-ja-fr-es-zh_2025-07-01_2025-09-30/en/Qwen2.5-3B-Instruct \
    --test_file_path data/benchmark/sports/en/test_mc.jsonl \
    --output_dir data/inference_output/sports/en \
    --temperature 0.0 \
    --n_return 1
```



3️⃣ Evaluation
Compare model performance across languages for the same source-trained model (e.g., trained on English).
```
PYTHONPATH=lib python3 demo_experiment/eval.py \
    --pred_dir data/inference_output/sports/en \
    --output_dir data/eval_output/sports/en \
    --epnum 3
```
