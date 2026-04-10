import os
import json
import re
import argparse
from collections import defaultdict



def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data



    
def mcq_eval(pred: str, ans_choice: str, ans_text: str) -> bool:
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
        return False




def cl_kt_eval(pred_path):
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
            is_correct = mcq_eval(item['pred'], item['answer'], item['text_answer'])
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
            score = round(correct / total, 3) if total else 0
            results[metric_name] = {"score": score, "correct": correct, "total": total}

        final_scores[str(lang_pair)] = results
    return final_scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single prediction JSONL file.")
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="Path to a single prediction JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output evaluation result JSON file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_file}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating: {args.pred_file}")
    eval_result = cl_kt_eval(args.pred_file)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    print(json.dumps(eval_result, indent=2, ensure_ascii=False))
    print(f"Evaluation results saved to {args.output_file}")
