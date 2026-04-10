import json
import os
import argparse
from openai_client import OpenAIModel
from prompts import music_genQA_prompts, movie_genQA_prompts, sports_genQA_prompts


def verify_qa(model, meta_data: str, qa_item: dict, templates):
    """Return verdict + source sentence for logging."""
    prompt = templates.FACTQA_VERIFIER_PROMPT.format(
        meta_data=meta_data,
        qa_item=json.dumps(qa_item, ensure_ascii=False)
    )
    response = model.generate(prompt=prompt, response_format={"type": "json_object"})
    parsed = json.loads(response.text[0])
    verdict = parsed.get("Decision", "")
    source = parsed.get("SourceSentence", "")

    log_entry = {
        "qa_item": qa_item,
        "verdict": verdict,
        "source": source
    }

    print("QA Item:", json.dumps(qa_item, ensure_ascii=False))
    print(f"Verification result: {verdict}")
    print(f"Source: {source}")
    return log_entry



def gen_FactQA(model, knowledge, source_lang, langs, templates, save_dir):
    FactQA = {}
    # --- Generate Source-Language QA ---
    src_path = os.path.join(save_dir, f"{source_lang}QA.json")
    log_path = os.path.join(save_dir, f"{source_lang}_verification_log.json")

    if os.path.exists(src_path):
        print(f"Loading existing {source_lang} QA from {src_path}")
        with open(src_path, "r", encoding="utf-8") as f:
            FactQA[source_lang] = json.load(f)
    else:
        print(f"Generating {source_lang} QA...")
        response = model.generate(
            prompt=templates.GEN_FACTQA_TEMPLATE.format(
                meta_data=knowledge, lang=source_lang
            ),
            response_format={"type": "json_object"}
        )
        qa = json.loads(response.text[0])

        verified_qa = []
        verification_logs = []

        for item in qa.get("QA", []):
            log_entry = verify_qa(model, knowledge, item, templates)
            verification_logs.append(log_entry)
            if log_entry["verdict"].upper() == "SUPPORTED":
                verified_qa.append(item)

        FactQA[source_lang] = verified_qa

        # Save verified QA
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump(verified_qa, f, indent=2, ensure_ascii=False)
        print(f"Saved {source_lang} QA: {src_path}")

        # Save verification logs
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(verification_logs, f, indent=2, ensure_ascii=False)
        print(f"Saved verification log: {log_path}")

    # --- Translate ---
    qa_str = json.dumps({"QA": FactQA[source_lang]}, ensure_ascii=False)
    for lang in langs:
        if lang == source_lang:
            continue
        lang_key = "zh" if lang.startswith("zh") else lang
        save_path = os.path.join(save_dir, f"{lang_key}QA.json")

        if os.path.exists(save_path):
            print(f"Skipping {lang}, already exists: {save_path}")
            with open(save_path, "r", encoding="utf-8") as f:
                FactQA[lang] = json.load(f)
            continue

        print(f"Translating QA to {lang}")
        lang_code = "zh-tw" if lang == "zh" else lang
        response = model.generate(
            prompt=templates.FACTQA_TRANSLATE_TEMPLATE.format(
                qa=qa_str, lang_code=lang_code
            ),
            response_format={"type": "json_object"}
        )
        translated = json.loads(response.text[0])
        FactQA[lang] = translated.get("QA", [])

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(FactQA[lang], f, indent=2, ensure_ascii=False)
        print(f"Saved {lang} QA: {save_path}")
    return FactQA



def main(training_docs_dir, output_dir, source_lang, qlangs, domain):
    if domain == "music":
        templates = music_genQA_prompts
    elif domain == "movie":
        templates = movie_genQA_prompts
    elif domain == "sports":
        templates = sports_genQA_prompts
    else:
        raise ValueError("domain must be 'music' or 'movie'")

    model = OpenAIModel("gpt-4o-mini", temperature=0.8, max_tokens=15000)
    time_stamp = os.path.basename(training_docs_dir)

    lang_docs_dir = os.path.join(training_docs_dir, source_lang)
    for doc_fn in os.listdir(lang_docs_dir):
        with open(os.path.join(lang_docs_dir, doc_fn), "r", encoding="utf-8") as f:
            train_docs = json.load(f)
        knowledge_text = train_docs["fact_source"]

        save_dir = os.path.join(
            output_dir, f"{time_stamp}", 
            os.path.splitext(doc_fn)[0]
        )
        os.makedirs(save_dir, exist_ok=True)
        gen_FactQA(
            model, knowledge_text, source_lang, 
            qlangs, templates, save_dir
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="movie", choices=["movie", "music", "sports"])
    parser.add_argument("--training_docs_dir", type=str, default="data/training_docs/movie/2025-01-01_2025-07-31")
    parser.add_argument("--output_dir", type=str, default="data/factQA/movie")
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--qlangs", type=str, nargs="+", default=["en", "ja", "zh", "fr", "es"])
    args = parser.parse_args()

    main(
        args.training_docs_dir,
        args.output_dir,
        args.source_lang,
        args.qlangs,
        args.domain,
    )
