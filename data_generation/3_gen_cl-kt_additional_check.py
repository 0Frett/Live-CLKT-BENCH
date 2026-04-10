# conservative check: filter the entity that is known by eval model

import json
import argparse
import os
from collections import defaultdict
import random
from typing import List
from openai_client import OpenAIModel
from prompts import music_genQA_prompts, movie_genQA_prompts, sports_genQA_prompts
from vllms import VLLMModel



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

    
def is_known_entity(entity_name, doc_text, lm, gpt, templates, verbose=True):
    if verbose:
        print(f"\n[Entity] {entity_name}")

    # 1. model generate
    is_known_prompt = templates.IS_KNOWN_ENTITY_PROMPT.format(entity_name=entity_name)

    if verbose:
        print("\n[Prompt to LM]")
        print(is_known_prompt)

    model_response = lm.generate(inputs=is_known_prompt, num_return_sequences=1)[0][0]

    if verbose:
        print("\n[Model Response]")
        print(model_response)

    # 2. judge
    check_prompt = templates.CHECK_TEMPLATE.format(
        entity_name=entity_name,
        doc_text=doc_text,
        response=model_response
    )

    if verbose:
        print("\n[Prompt to Judge]")
        print(check_prompt)

    out = gpt.generate(
        prompt=check_prompt,
        response_format={"type": "json_object"}
    )

    raw_output = out.text[0]

    if verbose:
        print("\n[Judge Raw Output]")
        print(raw_output)

    try:
        obj = json.loads(raw_output)
    except Exception as e:
        if verbose:
            print("[ERROR] Failed to parse JSON:", e)
        return False

    is_known = obj.get("is_known", False)

    if verbose:
        print("\n[Parsed is_known]")
        print(is_known, type(is_known))

    # normalize output
    if isinstance(is_known, bool):
        result = is_known
    elif isinstance(is_known, str):
        result = is_known.strip().lower() in ["true", "yes"]
    else:
        result = False

    if verbose:
        print(f"\n[Final Decision] -> {result}")
        print("=" * 60)

    return result



def main(
    doc_dir:str,
    factqa_dir:str, 
    test_langs:List[str], 
    output_dir:str, 
    val_ratio:float,
    eval_model:str,
    domain:str,
    tp:int,
    gpu_mem:float,
    seed:int=204,
):
    
    lm = VLLMModel(
        model=eval_model, temperature=0.6, max_tokens=4096, 
        tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem, max_model_len=6000)
    gpt = OpenAIModel('gpt-4o-mini', temperature=0.6, max_tokens=14000)

    if domain == "music":
        templates = music_genQA_prompts
    elif domain == "movie":
        templates = movie_genQA_prompts
    elif domain == "sports":
        templates = sports_genQA_prompts
    else:
        raise ValueError("domain must be 'music' or 'movie' or 'sports'")

    rng = random.Random(seed)

    for train_lang in test_langs:

        train_lang_factQA_dir = os.path.join(factqa_dir)
        train_lang_doc_dir = os.path.join(doc_dir, train_lang)
        train_docs, test_mcqs = [], []

        for unit_name in os.listdir(train_lang_factQA_dir):
            #  train doc
            doc_fp = os.path.join(train_lang_doc_dir, f"{unit_name}.json")
            with open(doc_fp, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            if is_known_entity(unit_name, doc_data["fact_source"], lm, gpt, templates):
                continue  # Skip this entity if it's known by the eval model

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


        # ----- Shuffle and split at item-level -----
        qid_groups_mcq = defaultdict(list)
        for item in test_mcqs:
            qid_groups_mcq[item["qid"]].append(item)

        all_qids = list(qid_groups_mcq.keys())
        rng.shuffle(all_qids)
        n_val = int(len(all_qids) * val_ratio)
        val_qids = set(all_qids[:n_val])
        print(f"[INFO] {train_lang} has {len(val_qids)} val qids out of {len(all_qids)} total qids.")

        val_mcq, test_mcq = [], []
        for qid in all_qids:
            if qid in val_qids:
                val_mcq.extend(qid_groups_mcq[qid])
            else:
                test_mcq.extend(qid_groups_mcq[qid])

        print(f"[INFO] {train_lang} has {len(val_mcq)} val and {len(test_mcq)} test items (MCQ).")

        # Save everything
        save_dir = os.path.join(output_dir, train_lang)
        os.makedirs(save_dir, exist_ok=True)
        save_jsonl(train_docs, save_dir, 'train_doc')
        save_jsonl(val_mcq, save_dir, 'val_mc')
        save_jsonl(test_mcq, save_dir, 'test_mc')
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
        default="data/factQA/movie/2025-01-01-2025-07-31",
        help="Directory of Movie QA Set"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,  
        default="data/benchmark/movie",
        help="Directory to save the Benchmark"
    )
    parser.add_argument(
        "--test_languages",
        type=str,
        nargs='+',
        default=["en", "ja", "fr", "es", "zh"],
        help="List of test language codes"
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--training_docs_dir", type=str, 
        default="data/train_docs/movie/2025-01-01_2025-07-31"
    )
    parser.add_argument("--eval_model", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    args = parser.parse_args()

    main(
        args.training_docs_dir,
        args.factqa_dir, 
        args.test_languages, 
        args.output_dir,
        args.val_ratio,
        args.eval_model,
        args.domain,
        args.tp,
        args.gpu_mem
    )
