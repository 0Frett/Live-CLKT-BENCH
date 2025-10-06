import json
import argparse
import os
from collections import defaultdict
import random
from typing import List


def mcq_format(item):

    template = (
        "{q}\n"
        "- A. {a}\n"
        "- B. {b}\n"
        "- C. {c}\n"
        "- D. {d}\n"
    )
    return template.format(
        q=item['question'],
        a=item['options']['A'],
        b=item['options']['B'],
        c=item['options']['C'],
        d=item['options']['D'],
    )


NATURALQA_TEMPLATE = "{q}"

def shuffle_options(qa_pair, rng):
    """
    Shuffle options and return updated qa_pair with:
    - 'options': shuffled dict {A: ..., B: ..., C: ..., D: ...}
    - 'correct_option': new letter after shuffle
    """
    options = list(qa_pair["options"].items())  # [(A, textA), (B, textB), ...]
    rng.shuffle(options)

    # build new mapping A/B/C/D -> text
    letters = ["A", "B", "C", "D"]
    new_options = {}
    correct_text = qa_pair["options"][qa_pair["correct_option"]]
    new_correct = None
    for new_letter, (old_letter, text) in zip(letters, options):
        new_options[new_letter] = text
        if text == correct_text:
            new_correct = new_letter

    qa_pair["options"] = new_options
    qa_pair["correct_option"] = new_correct
    return qa_pair


def main(
    doc_dir:str,
    factqa_dir:str, 
    test_langs:List[str], 
    output_dir:str, 
    val_ratio:float=0.2, 
    seed:int=204,
):

    rng = random.Random(seed)

    for train_lang in test_langs:

        train_lang_factQA_dir = os.path.join(factqa_dir, train_lang)
        train_lang_doc_dir = os.path.join(doc_dir, train_lang)
        train_docs, test_mcqs, test_naturalqs = [], [], []

        for unit_name in os.listdir(train_lang_factQA_dir):
            #  train doc
            doc_fp = os.path.join(train_lang_doc_dir, f"{unit_name}.json")
            with open(doc_fp, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            train_docs.append({
                "text": doc_data["fact_source"],
                "source": unit_name
            })

            #  test qas
            for test_lang in test_langs:

                qa_fp = os.path.join(train_lang_factQA_dir, unit_name, f"{test_lang}QA.json")
                with open(qa_fp, "r", encoding="utf-8") as f:
                    qa_data = json.load(f)
                # print(qa_fp)
                for idx, qa_pair in enumerate(qa_data):
                    qa_pair = shuffle_options(qa_pair, rng)
                    choice = qa_pair['correct_option']
                    test_mcqs.append(
                        {
                            'question': mcq_format(qa_pair),
                            'answer': choice,
                            'text_answer': qa_pair['options'][choice],
                            'train_lang': train_lang,
                            'test_lang': test_lang,
                            'source': unit_name,
                            'qid': f"{unit_name}-{idx}"
                        }
                    )
                    test_naturalqs.append(
                        {
                            'question': NATURALQA_TEMPLATE.format(q=qa_pair['question']),
                            'answer': qa_pair['options'][choice],
                            'train_lang': train_lang,
                            'test_lang': test_lang,
                            'source': unit_name,
                            'qid': f"{unit_name}-{idx}"
                        }
                    )


        # ----- Shuffle and split at item-level -----
        # rng = random.Random(seed)

        # group by qid (for MCQ + Natural QA in sync)
        qid_groups_mcq = defaultdict(list)
        qid_groups_nat = defaultdict(list)
        for item in test_mcqs:
            qid_groups_mcq[item["qid"]].append(item)
        for item in test_naturalqs:
            qid_groups_nat[item["qid"]].append(item)

        all_qids = list(qid_groups_mcq.keys())
        rng.shuffle(all_qids)
        n_val = int(len(all_qids) * val_ratio)
        val_qids = set(all_qids[:n_val])
        print(f"[INFO] {train_lang} has {len(val_qids)} val qids out of {len(all_qids)} total qids.")

        val_mcq, test_mcq = [], []
        val_nat, test_nat = [], []
        for qid in all_qids:
            if qid in val_qids:
                val_mcq.extend(qid_groups_mcq[qid])
                val_nat.extend(qid_groups_nat[qid])
            else:
                test_mcq.extend(qid_groups_mcq[qid])
                test_nat.extend(qid_groups_nat[qid])

        print(f"[INFO] {train_lang} has {len(val_mcq)} val and {len(test_mcq)} test items (MCQ).")
        print(f"[INFO] {train_lang} has {len(val_nat)} val and {len(test_nat)} test items (Natural QA).")

        # Save everything
        save_dir = os.path.join(output_dir, train_lang)
        os.makedirs(save_dir, exist_ok=True)
        save_jsonl(train_docs, save_dir, 'train_doc')
        save_jsonl(val_mcq, save_dir, 'val_mc')
        save_jsonl(test_mcq, save_dir, 'test_mc')
        save_jsonl(val_nat, save_dir, 'val_natural')
        save_jsonl(test_nat, save_dir, 'test_natural')
        print(f"[END] Finish Generating {train_lang} CL-KT Benchmark to {save_dir}")


def save_jsonl(data, save_dir, base_name):
    file_path = os.path.join(save_dir, f"{base_name}.jsonl")
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to: {file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factqa_dir", 
        type=str, 
        default="data/movie/factQA/2025-01-01-2025-07-31",
        help="Directory of Movie QA Set"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,  
        default="data/movie/benchmarks/en-ja-fr-es-zh-2025-01-01-2025-07-31/cl-kt",
        help="Directory to save the Benchmark"
    )
    parser.add_argument(
        "--test_languages",
        type=str,
        nargs='+',
        default=["en", "ja", "fr", "es", "zh"],
        help="List of test language codes"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2
    )
    parser.add_argument(
        "--training_docs_dir", type=str, 
        default="data/movie/training_docs/2025-01-01_2025-07-31"
    )
    args = parser.parse_args()

    main(
        args.training_docs_dir,
        args.factqa_dir, 
        args.test_languages, 
        args.output_dir,
        args.val_ratio
    )
