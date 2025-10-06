import os
import json
import argparse
from tqdm import tqdm
import torch
import gc
from llms import LanguageModel, GPT, LanguageModelPretrained
torch.cuda.empty_cache()

def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_inference(inferencer, input_data, temperature, n, qa_type):
    """Run inference on a single model checkpoint."""
    print(f"[INFO] Running inference")
    if qa_type == "mc":
        max_tokens = 64
    if qa_type == "nlq":
        max_tokens = 256

    results = []
    for item in tqdm(input_data):
        prompt = item["question"]
        if qa_type == "mc":
            prompt += "Please output only the correct option letter followed by its text, in format: <option letter>. <option text>."
        if qa_type == "nlq":
            prompt += "Please provide a short answer in a few words only."

        response = inferencer.generate(
            prompt=prompt, max_new_tokens=max_tokens,
            temperature=temperature, num_return_sequences=n
        )
        if n == 1:
            item['pred'] = response[0]
        else:
            item['pred'] = response

        results.append(item)

        print("------------------------------", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        print(f"Response: {response}", flush=True)

    del inferencer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return results


def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference on multiple model checkpoints.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default=None,
        help="Hugging Face model ID or local path for a pretrained model. If specified, uses pretrained mode."
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default=None,
        help="Directory containing model checkpoints. Used only if --model_id is not provided."
    )
    parser.add_argument(
        "--test_file_path", 
        type=str, 
        default="data/movie/benchmarks/en-ja-fr-es-zh_2025-01-01_2025-08-31/cl-kt/en/val.jsonl", 
        help="Path to JSONL file with validation prompts."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="inference_output/movie/en-ja-fr-es-zh_2025-01-01_2025-08-31/cl-kt/en",
        help="Directory to save generated responses."
    )
    parser.add_argument("--qa_type", type=str, default="mc")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_return", type=int, default=1)
    parser.add_argument("--ep_num", type=int, default=10)
    args = parser.parse_args()

    val_data = load_jsonl(args.test_file_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------
    # Pretrained model mode
    # --------------------
    if args.model_id:
        print(f"[INFO] Using pretrained model: {args.model_id}")
        lm = LanguageModelPretrained(args.model_id)
        save_path = os.path.join(args.output_dir, f"{args.model_id.split('/')[1]}_pred.jsonl")
        if os.path.exists(save_path):
            print(f"Pass Exist File : {save_path}")
        else:
            outputs = run_inference(lm, val_data, args.temperature, args.n_return, args.qa_type)
            save_jsonl(outputs, save_path)
            print(f"[✓] Saved: {save_path}")
        return

    # --------------------
    # Checkpoint directory mode
    # --------------------
    if args.model_dir is None:
        raise ValueError("You must provide either --model_id or --model_dir")

    # Loop through checkpoints
    ckpts = [f"checkpoint-epoch-{i}" for i in range(1, args.ep_num + 1)]
    for ckpt_name in ckpts:
        ckpt_dir = os.path.join(args.model_dir, ckpt_name)
        if not os.path.isdir(ckpt_dir):
            continue  # skip missing checkpoints
        print(f"[INFO] Processing checkpoint: {ckpt_dir}")

        save_dir = os.path.join(args.output_dir, os.path.basename(args.model_dir))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{ckpt_name}_pred.jsonl")
        if os.path.exists(save_path):
            print(f"Pass Exist File : {save_path}")
            continue

        lm = LanguageModel(ckpt_dir)
        outputs = run_inference(lm, val_data, args.temperature, args.n_return, args.qa_type)
        save_jsonl(outputs, save_path)
        print(f"[✓] Saved: {save_path}")

        del lm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

