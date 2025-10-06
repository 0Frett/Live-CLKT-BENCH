import os
import json
import re
import argparse
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from collections import Counter
from scipy.stats import kendalltau
from openai_client import OpenAIModel_parallel


STANCES = ["supportive", "opposing", "neutral"]
MODEL = OpenAIModel_parallel('gpt-4o-mini-2024-07-18', temperature=0.8, max_tokens=9999)

MC_JUDGE_template = """
    You are given a multiple-choice QA evaluation task.

    Question: {q}
    Correct Answer: {ans_choice}. {ans_text}
    Model Response: {pred}

    Your task:
    1. Determine if the model's response clearly corresponds to **exactly one option**.
    2. Accept the response if it correctly matches either:
        - the correct option letter ({ans_choice}), OR
        - the correct answer text ("{ans_text}").
    3. Reject responses that:
        - Contain multiple option letters (e.g., "A and B"),
        - List several answer choices,

    Output one of the following JSON format:
    {{"correct": "True"}} or {{"correct": "False"}} 
"""

NLQ_JUDGE_Template = """
    You are evaluating a question–answer task.

    Question: {q}
    Correct Answer: {ans_text}
    Model Response: {pred}

    Your task:
    1. Check if the model’s response provides a clear and unambiguous answer.
    2. Mark it correct if the response expresses the same meaning as the correct answer, even if the wording or language is different.
    3. Mark it incorrect if the response:
        - Gives multiple possible answers,
        - Contradicts the correct answer,
        - Or does not clearly match the expected answer.

    Output strictly in the following JSON format:
    {{"correct": "True"}} or {{"correct": "False"}}
"""


def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


    
def mcq_eval(question: str, pred: str, ans_choice: str, ans_text: str) -> bool:
    print("-"*20)
    print(f"Expected answer: {ans_choice}. {ans_text}")
    print(f"Predicted response: {pred}")
    match = re.search(r'\b([A-D])\b', pred)  # capture A–D as standalone
    if match:
        extract_pred = match.group(1)
        print(f"Pred Choice: '{extract_pred}'")
        if extract_pred == ans_choice:
            print("Descision: True")
            return True
        else:
            print("Descision: False")
            return False

    else:
        print(f"Fail Parse Letter Choice")
        # random choose a letter
        pred = random.choice(['A', 'B', 'C', 'D'])
        if pred == ans_choice:
            print("Guess: True")
            return True
        else:
            print("Guess: False")
            return False



def nlq_eval(question: str, pred: str, ans_text: str) -> bool:
    print("-"*20)
    print(f"Expected answer: {ans_text}")
    print(f"Predicted response: {pred}")

    print("Start LLM Judge")
    response = MODEL.generate(
        prompt=NLQ_JUDGE_Template.format(
            q=question, 
            pred=pred, 
            ans_text=ans_text
        ),
        response_format={"type": "json_object"},
        num_return_sequences=1
    ).text[0].strip()
    print("LLM Judge Response:", response)
    is_correct = json.loads(response)["correct"]
    if isinstance(is_correct, str):
        is_correct = (is_correct.lower() == "true")

    if is_correct:
        print("LLM Judge --> True")
        return True
    print("LLM Judge --> False")   
    return False



def cl_kt_eval(pred_path, qa_type):
    pred_data = load_jsonl(pred_path)

    # Step 1: Cluster by qid
    qid_clusters = defaultdict(list)
    for item in pred_data:
        qid_clusters[item["qid"]].append(item)

    # Store scores per (train_lang, test_lang)
    scores_per_lang_pair = defaultdict(
        lambda: {
            "overall": [], "learn": [], "transfer": [],
            "sctc": [], "sctw": [], "swtc": [], "swtw": []
        }
    )

    for qid, items in qid_clusters.items():
        # Map language → correctness
        lang_correct = {}
        for item in items:
            if qa_type == "mc":
                is_correct = mcq_eval(
                    item['question'], item['pred'], 
                    item['answer'], item['text_answer']
                )
            elif qa_type == "nlq":
                is_correct = nlq_eval(item['question'], item['pred'], item['answer'])

            lang_correct[item['test_lang']] = is_correct

        for item in items:
            ls, lt = item['train_lang'], item['test_lang']
            src_correct = int(lang_correct.get(ls))
            tgt_correct = int(lang_correct.get(lt))

            # Overall success: correct in both source and target / all
            scores_per_lang_pair[(ls, lt)]["overall"].append(int(src_correct and tgt_correct))
            # Learn success: correct in source / all
            scores_per_lang_pair[(ls, lt)]["learn"].append(src_correct)
            # Transfer success: correct in target / correct in source
            if src_correct:
                scores_per_lang_pair[(ls, lt)]["transfer"].append(tgt_correct)

            # Stats
            scores_per_lang_pair[(ls, lt)]["sctc"].append(int(src_correct and tgt_correct))        # source correct, target correct
            scores_per_lang_pair[(ls, lt)]["sctw"].append(int(src_correct and not tgt_correct))    # source correct, target wrong
            scores_per_lang_pair[(ls, lt)]["swtc"].append(int(not src_correct and tgt_correct))    # source wrong, target correct
            scores_per_lang_pair[(ls, lt)]["swtw"].append(int(not src_correct and not tgt_correct))# source wrong, target wrong

    # Compute final metric per (train_lang, test_lang)
    final_scores = {}
    for lang_pair, metrics_eval in scores_per_lang_pair.items():
        results = {}

        for metric_name, values in metrics_eval.items():
            correct = sum(values)
            total = len(values)
            score = correct / total if total else 0
            results[metric_name] = {
                "score": score,
                "correct": correct,
                "total": total,
            }

        final_scores[str(lang_pair)] = results
    return final_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process answer and prediction files.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="test_output/movie/en-ja-zh-fr-es-2025-01-01-2025-07-31/kt",
        help="Directory of prediction JSONL file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="eval_result/FR-TW-JP-US-ES_2025-01-01_2025-07-15/",
        help="Directory of eval result"
    )
    parser.add_argument(
        "--qa_type",
        type=str,
        default="mc",
        help="Type of benchmark "
    )
    parser.add_argument(
        "--ep_num",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="finetune",
        choices=["finetune", "pretrain"],
        help="Type of model (finetune or pretrain)"
    )
    
    args = parser.parse_args()
    pred_dir = args.pred_dir
    qa_type = args.qa_type
    epnum = args.ep_num
    model_type = args.model_type

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"eval_result.json")
    if os.path.exists(output_file):
        # end
        print(f"⚠️ Eval result already exists: {output_file}. Please remove it first if you want to re-evaluate.")
        exit(0)

    if qa_type in ['mc', 'nlq']:
        eval_result = {}
        if model_type == "pretrain":
            for fn in os.listdir(pred_dir):
                ckpt_pred_path = os.path.join(pred_dir, fn)
                model_name = fn.split("_pred.jsonl")[0]
                print(f"Evaluating {ckpt_pred_path} ...")
                seq = cl_kt_eval(ckpt_pred_path, qa_type)
                eval_result[model_name] = seq
        else:
            for model in os.listdir(pred_dir):
                seq = []
                for epoch in range(1, epnum+1):
                    filename = f"checkpoint-epoch-{epoch}_pred.jsonl"
                    ckpt_pred_path = os.path.join(pred_dir, model, filename)
                    print(f"Evaluating {model} at epoch {epoch}...")
                    seq.append(cl_kt_eval(ckpt_pred_path, qa_type))
                eval_result[model] = seq

    else:
        print(f"Unsupported Question Type {qa_type}")

    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_file}")