import torch
import os
import argparse
import shutil
import wandb
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import gc
from inference import run_inference


def load_jsonl(path):
    """Load a JSONL file as a list of dictionaries."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data



def inspect_lora_modules(model):
    print("Inspecting LoRA-wrapped modules...")
    total_lora_params = 0
    for name, module in model.named_modules():
        if "lora" in name.lower():
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_lora_params += num_params
            print(f"{name} — trainable params: {num_params}")
    print(f"\nTotal trainable LoRA parameters: {total_lora_params:,}")


def preprocess_unsupervised(example, tokenizer):
    text = str(example["text"])
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    labels = input_ids.copy()
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels}


def evaluate(model, val_data, tokenizer, temperature=0.7, n=1):
    """
    Evaluate the model on MCQ validation data.
    Returns a metric (accuracy by default).
    """
    # Run inference
    model.eval()
    input_data = val_data  # val_data is a list of dicts: {"question": ..., "answer": ...}
    results = []
    correct = 0
    total = 0

    for item in input_data:
        prompt = item["question"]
        with torch.no_grad():
            output = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=512,
                temperature=temperature,
                num_return_sequences=n,
            )
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
        item['pred'] = pred_text
        results.append(item)

        # Simple MCQ accuracy: check if the correct answer text is in pred
        if "answer" in item:
            if item["answer"].strip().lower() in pred_text.lower():
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0

    # free GPU memory
    del results
    torch.cuda.empty_cache()
    gc.collect()

    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--project_name", type=str, default="supervised-finetuning"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Check if training was already completed
        ckpts = os.listdir(args.output_dir)
        dirs = [os.path.join(args.output_dir, d) for d in ckpts if os.path.isdir(os.path.join(args.output_dir, d))]

        if len(dirs) == args.num_train_epochs:
            print(f"✅ MODEL {args.model_name} ALREADY TRAINED on {args.train_file}")
            return
        else:
            print(f"⚠️ Incomplete training found for {args.model_name}. Cleaning up...")
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)

    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=f"run_{os.path.basename(args.output_dir)}"
    )
    
    if args.model_name == "microsoft/Phi-3.5-mini-instruct":
        print("Using Phi-3.5-mini-instruct model, setting LoRA config for all-linear modules.")
        lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    else:
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='eager'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    # inspect_lora_modules(model)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    val_data = load_jsonl(args.test_file_path)
    # val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    print("Using preprocess function.")
    train_dataset = train_dataset.map(lambda x: preprocess_unsupervised(x, tokenizer), remove_columns=train_dataset.column_names)
    # val_dataset = val_dataset.map(lambda x: preprocess_unsupervised(x, tokenizer), remove_columns=val_dataset.column_names)

    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch]
        labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    train_step_losses = []
    val_epoch_scores = []
    train_epoch_losses = []

    model.train()
    global_step = 0
    for epoch in range(args.num_train_epochs):
        accum_raw = 0.0
        epoch_step_losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            for k in batch:
                batch[k] = batch[k].to(model.device)

            with autocast("cuda"):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_raw += raw_loss.item()

            if step % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_raw / args.gradient_accumulation_steps
                accum_raw = 0.0

                print(f"Epoch {epoch+1} | Step {global_step} | Train loss {avg_loss:.4f}")
                wandb.log({"train/loss": avg_loss, "step": global_step})
                train_step_losses.append({"step": global_step, "loss": avg_loss})
                epoch_step_losses.append(avg_loss)

        if step % args.gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            remainder = step % args.gradient_accumulation_steps
            avg_loss = accum_raw / remainder
            print(f"Epoch {epoch+1} | Step {global_step} | Train loss {avg_loss:.4f} (final step)")
            wandb.log({"train/loss": avg_loss, "step": global_step})
            train_step_losses.append({"step": global_step, "loss": avg_loss})
            epoch_step_losses.append(avg_loss)

        epoch_avg_loss = sum(epoch_step_losses) / len(epoch_step_losses)
        train_epoch_losses.append({"epoch": epoch + 1, "loss": epoch_avg_loss})
        print(f"Epoch {epoch + 1} | Train Epoch Avg Loss: {epoch_avg_loss:.4f}")

        val_score = evaluate(model, val_data)
        print(f"Epoch {epoch + 1} | Validation loss {val_score:.4f}")
        wandb.log({"val/score": val_score, "epoch": epoch + 1})
        val_epoch_scores.append({"epoch": epoch + 1, "score": val_score})

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        # Save losses to JSON files
        with open(os.path.join(args.output_dir, "train_step_losses.json"), "w") as f:
            json.dump(train_step_losses, f, indent=2)

        with open(os.path.join(args.output_dir, "val_epoch_scores.json"), "w") as f:
            json.dump(val_epoch_scores, f, indent=2)
        
        with open(os.path.join(args.output_dir, "train_epoch_losses.json"), "w") as f:
            json.dump(train_epoch_losses, f, indent=2)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
