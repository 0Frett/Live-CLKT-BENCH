# LiveCLKTBench 

Evaluating cross-lingual knowledge transfer in large language models (LLMs) is challenging, as correct answers in a target language may arise either from genuine transfer or from prior exposure during pre-training.
We present **LiveCLKTBench**, an automated generation pipeline specifically designed to isolate and measure **C**ross-**L**ingual **K**nowledge **T**ransfer. Our pipeline identifies self-contained, time-sensitive knowledge entities from real-world domains, filters them based on temporal occurrence, and verifies them against the model’s knowledge. The documents of these valid entities are then used to generate factual questions, which are translated into multiple languages to evaluate transferability across linguistic boundaries.

## Overview

This repository provides two main workflows:

1. **Benchmark generation**
   - collect valid entities and related documents
   - generate factual QA instances
   - 

2. **Model evaluation**
   - run model inference on generated benchmarks
   - evaluate question predictions


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
    --domain movie \
    --start_str 2026-01-01 \
    --end_str 2026-04-01 \
    --output_dir test_data/entities \
    --max_entity 20

PYTHONPATH=lib python3 data_generation/0_collect_entity.py \
    --domain sports \
    --start_str 2026-03-01 \
    --end_str 2026-04-01 \
    --output_dir test_data/entities \
    --max_entity 20
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
    --entity_file test_data/entities/sports/2026-03-01_2026-04-01.json \
    --output_dir test_data/train_docs/sports \
    --test_languages en ja fr es zh


PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain movie \
    --entity_file test_data/entities/movie/2025-01-01_2026-04-01.json \
    --output_dir test_data/train_docs/movie \
    --test_languages en ja fr es zh

PYTHONPATH=lib python3 data_generation/1_gen_train_docs.py \
    --domain music \
    --entity_file test_data/entities/music/2025-01-01_2026-04-01.json \
    --output_dir test_data/train_docs/music \
    --test_languages en ja fr es zh
```



## Step 3. Generate Factual QA Pairs
Creates factual, document-grounded multiple-choice questions for each entity.
```
PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain sports \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-01 \
    --output_dir test_data/factQA/sports \
    --qlangs en ja fr es zh

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain music \
    --training_docs_dir test_data/train_docs/music/2025-12-01_2026-04-01 \
    --output_dir test_data/factQA/music \
    --qlangs en ja fr es zh

PYTHONPATH=lib python3 data_generation/2_gen_fact_qa.py \
    --domain movie \
    --training_docs_dir test_data/train_docs/movie/2026-01-01_2026-04-01 \
    --output_dir test_data/factQA/movie \
    --qlangs en ja fr es zh
```



## Step 4. Assemble the LiveCLKTBench Benchmark

Combines factual QA and document data to construct the final benchmark split into train/validation/test sets.
```
PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir test_data/factQA/sports/2026-03-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-01 \
    --output_dir test_data/benchmark/sports \
    --test_languages en ja fr es zh \
    --val_ratio 0.2


PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir test_data/factQA/movie/2026-01-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/movie/2026-01-01_2026-04-01 \
    --output_dir test_data/benchmark/movie \
    --test_languages en ja fr es zh \
    --val_ratio 0.2


PYTHONPATH=lib python3 data_generation/3_gen_cl-kt.py \
    --factqa_dir test_data/factQA/music/2025-12-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/music/2025-12-01_2026-04-01 \
    --output_dir test_data/benchmark/music \
    --test_languages en ja fr es zh \
    --val_ratio 0.2
```

```
PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir test_data/factQA/sports/2026-03-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/sports/2026-03-01_2026-04-01 \
    --output_dir test_data/benchmark_add/sports \
    --test_languages en ja fr es zh \
    --val_ratio 0.2 \
    --eval_model Qwen/Qwen2.5-3B-Instruct \
    --domain sports \
    --tp 1 \
    --gpu_mem 0.9


PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir test_data/factQA/movie/2026-01-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/movie/2026-01-01_2026-04-01 \
    --output_dir test_data/benchmark_add/movie \
    --test_languages en ja fr es zh \
    --val_ratio 0.2 \
    --eval_model Qwen/Qwen2.5-3B-Instruct \
    --domain movie \
    --tp 1 \
    --gpu_mem 0.9


PYTHONPATH=lib python3 data_generation/3_gen_cl-kt_additional_check.py \
    --factqa_dir test_data/factQA/music/2025-12-01_2026-04-01 \
    --training_docs_dir test_data/train_docs/music/2025-12-01_2026-04-01 \
    --output_dir test_data/benchmark_add/music \
    --test_languages en ja fr es zh \
    --val_ratio 0.2 \
    --eval_model Qwen/Qwen2.5-3B-Instruct \
    --domain music \
    --tp 1 \
    --gpu_mem 0.9
```

## 🧠 Demo Experiment

1️⃣ Continual Pretraining (CPT)
Example: source language = English
```
python3 demo_experiment/cpt.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --train_file test_data/benchmark/sports/en/train_doc.jsonl \
    --output_dir test_models/sports/en/Qwen2.5-1.5B-Instruct \
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
PYTHONPATH=lib python3 demo_experiment/lora_inference.py \
    --model_dir test_models/sports/en/Qwen2.5-3B-Instruct \
    --test_file_path test_data/benchmark/sports/en/test_mc.jsonl \
    --output_dir test_data/inference_output/sports/en/finetune \
    --temperature 0.6

PYTHONPATH=lib python3 demo_experiment/vllm_inference.py \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --test_file_path test_data/benchmark/sports/en/test_mc.jsonl \
    --output_dir test_data/inference_output/sports/en/pretrain \
    --temperature 0.6 \
    --tp 1 \
    --gpu_mem 0.9
```



3️⃣ Evaluation
```
python3 demo_experiment/eval.py \
  --pred_file test_data/inference_output/sports/en/pretrain/Qwen2.5-3B-Instruct_pred.jsonl \
  --output_file test_data/eval_result/sports/en/pretrain/Qwen2.5-3B-Instruct_pred.json
```
