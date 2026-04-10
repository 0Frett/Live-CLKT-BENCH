import os
import json
import argparse
from tqdm import tqdm
import torch
import gc
from llms import LanguageModel
torch.cuda.empty_cache()

def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_inference(inferencer, input_data, temperature, max_tokens = 64):
    """Run inference on a single model checkpoint."""
    print(f"[INFO] Running inference")

    results = []
    for item in tqdm(input_data):
        prompt = item["question"]
        prompt += "Please output only the correct option letter followed by its text, in format: <option letter>. <option text>."
 

        response = inferencer.generate(
            prompt=prompt, max_new_tokens=max_tokens,
            temperature=temperature, num_return_sequences=1
        )[0]
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
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--test_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    val_data = load_jsonl(args.test_file_path)

    os.makedirs(args.output_dir, exist_ok=True)


    # --------------------
    # Checkpoint directory mode
    # --------------------
    if args.model_dir is None:
        raise ValueError("You must provide either --model_id or --model_dir")

    # Loop through checkpoints
    dirs = [
        d for d in os.listdir(args.model_dir)
        if os.path.isdir(os.path.join(args.model_dir, d)) and d.startswith("checkpoint-epoch-")
    ]
    ckpts = [f"checkpoint-epoch-{i}" for i in range(1, len(dirs) + 1)]
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
        outputs = run_inference(lm, val_data, args.temperature)
        save_jsonl(outputs, save_path)
        print(f"[✓] Saved: {save_path}")

        del lm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

