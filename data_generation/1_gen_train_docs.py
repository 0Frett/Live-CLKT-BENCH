import json
import os
import argparse
from openai_client import OpenAIModel_parallel
from prompts import music_genQA_prompts, movie_genQA_prompts, sports_genQA_prompts
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def check_lang(text, lang):
    try:
        detected_lang = detect(text)
        if lang == "zh":
            allowed = {"zh-cn", "zh-tw", "zh", "ko"}  # strange langdetect bug ko
            if detected_lang not in allowed:
                return False
        else:
            if detected_lang != lang:
                return False
        return True
    except LangDetectException:
        return False


def get_train_doc(model, unit, lang, domain, templates):
    print(f"Translating Doc to {lang}")
    target_lang = "zh-tw" if lang == "zh" else lang

    # --- Domain unit unpacking ---
    if domain == "music":
        translated_doc = templates.DOC_TEMPLATE.format(
            title=unit['title'],
            date=unit["published_time"][:10],
            description=unit['description']
        )

    elif domain == "sports":
        if target_lang in ["en", "ja", "zh-tw"]:
            #  baseball template
            original_doc = templates.build_en_ja_zh_doc(unit)
        elif target_lang in ["fr", "es"]:
            #  football template
            original_doc = templates.build_es_fr_doc(unit)
        else:
            print(f"Unsupported sport/lang combination: {sport}, {target_lang}")

        # print(original_doc)
        if target_lang == "en":
            translated_doc = original_doc
        else:
            output = model.generate(
                prompt=templates.DOC_TRANSLATE_TEMPLATE.format(
                    lang=target_lang, text=original_doc, title=unit['title']
                )
            )
            translated_doc = output.text[0]
    

    elif domain == "movie":
        if target_lang == "en":
            translated_doc = templates.DOC_TEMPLATE.format(
                title=unit['title'],
                casts=", ".join(unit.get("top5cast", [])),
                summary=" ".join(unit.get("summary", [])),
                synopsis=" ".join(unit.get("synopsis", [])),
            )

        else:
            output = model.generate(
                prompt=templates.DOC_TRANSLATE_TEMPLATE.format(
                    casts=", ".join(unit.get("top5cast", [])),
                    summary=" ".join(unit.get("summary", [])),
                    synopsis=" ".join(unit.get("synopsis", [])),
                    lang=target_lang
                ),
                response_format={"type": "json_object"}
            )
            trans = json.loads(output.text[0])["translation"]
            translated_doc = templates.DOC_TEMPLATE.format(
                title=unit['title'],
                casts=trans["Cast"],
                summary=trans["Summary"],
                synopsis=trans["Synopsis"],
            )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    print(f"{target_lang} doc : {translated_doc}")
    return translated_doc


def main(domain_units_dir, output_dir, source_lang, domain):
    if domain == "music":
        templates = music_genQA_prompts
    elif domain == "movie":
        templates = movie_genQA_prompts
    elif domain == "sports":
        templates = sports_genQA_prompts
    else:
        raise ValueError("domain must be 'music' or 'movie' or 'sports'")

    model = OpenAIModel_parallel('gpt-4o-mini', temperature=0.8, max_tokens=9999)

    time_stamp = os.path.basename(domain_units_dir)

    with open(os.path.join(domain_units_dir, f"{source_lang}.json"), 'r', encoding='utf-8') as f:
        units = json.load(f)
    
    save_dir = os.path.join(output_dir, time_stamp, source_lang)
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, unit in enumerate(units):
        title = unit.get('title')
        save_path = os.path.join(save_dir, f"{title}.json")

        if os.path.exists(save_path):
            print(f"[{idx}/{len(units)}] Skipping... Train docs already saved to {save_path}")
            continue
        else:
            print(f"[{idx+1}/{len(units)}] Processing {domain}: {title}")
            doc = get_train_doc(model, unit, source_lang, domain, templates)

        unit_train_doc = {
            "fact_source": doc,
            "opinion_source": [c["text"] for c in unit.get('comments')]
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(unit_train_doc, f, indent=2, ensure_ascii=False)
        print(f"Finished: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str,
        default="movie", choices=["movie", "music", "sports"]
    )
    parser.add_argument(
        "--domain_units_dir", type=str, 
        default="data/movie/domain_units/2025-01-01_2025-07-31"
    )
    parser.add_argument(
        "--output_dir", type=str, 
        default="data/movie/training_docs"
    )
    parser.add_argument(
        "--source_lang", type=str, default="en"
    )
    args = parser.parse_args()

    main(
        args.domain_units_dir, 
        args.output_dir,
        args.source_lang,
        args.domain,
    )
