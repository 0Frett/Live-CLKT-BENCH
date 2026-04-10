import os
import json
import argparse
from tqdm import tqdm
from vllms import VLLMModel


def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_inference(model, input_data):
    """Run inference on a single model checkpoint."""
    print(f"[INFO] Running inference")

    results = []
    for item in tqdm(input_data):
        prompt = item["question"]
        prompt += "Please output only the correct option letter followed by its text, in format: <option letter>. <option text>."
        response = model.generate(inputs=prompt, num_return_sequences=1)[0][0]
        item['pred'] = response 
        results.append(item)

        print("------------------------------", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        print(f"Response: {response}", flush=True)


    return results


def save_jsonl(data, path):
    """Save list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference on multiple model checkpoints.")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--test_file_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=6000)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    args = parser.parse_args()

    val_data = load_jsonl(args.test_file_path)
    os.makedirs(args.output_dir, exist_ok=True)

    lm = VLLMModel(
        model=args.model_id, 
        temperature=args.temperature, 
        max_tokens=args.max_tokens, 
        tensor_parallel_size=args.tp, 
        gpu_memory_utilization=args.gpu_mem, 
        max_model_len=args.max_model_len
    )
    
    save_path = os.path.join(args.output_dir, f"{args.model_id.split('/')[1]}_pred.jsonl")
    if os.path.exists(save_path):
        print(f"Pass Exist File : {save_path}")
    else:
        outputs = run_inference(lm, val_data)
        save_jsonl(outputs, save_path)
        print(f"[✓] Saved: {save_path}")
    

if __name__ == "__main__":
    main()

