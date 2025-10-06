import os
import json
import argparse
import random

def format_clkt_file(data):
    train = []
    for item in data:
        train.append({"text": item["text"]})
    return train


def format_dl_file(data, batch_size=5, seed=204):
    train = []
    rng = random.Random(seed)

    for item in data:
        current_id = 0
        info = item.get("info", "").strip()
        comments = item.get("comments", [])
        if not isinstance(comments, list):
            comments = [comments]

        # Clean comments
        comments = [c.strip() for c in comments if c.strip()]
        if not info or not comments:
            continue

        # Shuffle comments
        rng.shuffle(comments)

        # Create batches
        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i:i+batch_size]
            batch_ids = list(range(current_id, current_id + len(batch_comments)))
            current_id += len(batch_comments)

            combined_text = f"{info}\n\nViewer Comments:\n" + "\n".join(f"Viewer_{cid}: {c}" for cid, c in zip(batch_ids, batch_comments))
            train.append({"text": combined_text})

    return train


def save_jsonl(data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to: {fp}")


def main(train_data_fp, task_type, output_fp):
    with open(train_data_fp, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    if task_type == "cl-kt":
        cp_data = format_clkt_file(data)

    elif task_type == "mono-dl" or task_type == "multi-dl":
        cp_data = format_dl_file(data, batch_size=5)

    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    save_jsonl(cp_data, output_fp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type", type=str, default="cl-kt",
        choices=["cl-kt", "mono-dl", "multi-dl"],
    )
    parser.add_argument(
        "--output_file", type=str,  
        default="data/movie/experiment/en-ja-fr-es-zh_2025-01-01_2025-08-31/cl-kt/en/cp.jsonl",
        help="Directory to save the Benchmark"
    )
    parser.add_argument(
        "--training_data", type=str, 
        default="data/movie/benchmarks/en-ja-fr-es-zh_2025-01-01_2025-08-31/cl-kt/en/train_doc.jsonl"
    )
    args = parser.parse_args()

    main(
        args.training_data,
        args.task_type, 
        args.output_file
    )
