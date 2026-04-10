import json
import os
import argparse
from openai_client import OpenAIModel
from prompts import music_genQA_prompts, movie_genQA_prompts, sports_genQA_prompts




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
        if target_lang == "en":
            translated_doc = templates.DOC_TEMPLATE.format(
                title=unit['title'],
                date=unit["published_time"][:10],
                description=unit['description']
            )

        else:
            output = model.generate(
                prompt=templates.DOC_TRANSLATE_TEMPLATE.format(
                    description=", ".join(unit.get("description", [])),
                    lang=target_lang
                ),
                response_format={"type": "json_object"}
            )
            trans = json.loads(output.text[0])
            translated_doc = templates.DOC_TEMPLATE.format(
                title=unit['title'],
                date=unit["published_time"][:10],
                description=trans["Description"]
            )


    elif domain == "sports":
        # if target_lang in ["en", "ja", "zh-tw"]:
            #  baseball template
        original_doc = templates.build_doc(unit)
        # elif target_lang in ["fr", "es"]:
        #     #  football template
        #     original_doc = templates.build_es_fr_doc(unit)

        # print(original_doc)
        if target_lang == "en":
            translated_doc = original_doc
        else:
            output = model.generate(
                prompt=templates.DOC_TRANSLATE_TEMPLATE.format(
                    lang=target_lang, text=original_doc
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

            sss = templates.DOC_TRANSLATE_TEMPLATE.format(
                    casts=", ".join(unit.get("top5cast", [])),
                    summary=" ".join(unit.get("summary", [])),
                    synopsis=" ".join(unit.get("synopsis", [])),
                    lang=target_lang
                )

            output = model.generate(
                prompt=sss,
                response_format={"type": "json_object"}
            )

            print(output.text[0])
            trans = json.loads(output.text[0])["translation"]
            translated_doc = templates.DOC_TEMPLATE.format(
                title=unit['title'],
                casts=trans["Cast"],
                summary=trans["Summary"],
                synopsis=trans["Synopsis"],
            )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    print(f"{target_lang.upper()} Document:\n{translated_doc}")
    return translated_doc



def main(entity_file, output_dir, test_languages, domain):
    if domain == "music":
        templates = music_genQA_prompts
    elif domain == "movie":
        templates = movie_genQA_prompts
    elif domain == "sports":
        templates = sports_genQA_prompts
    else:
        raise ValueError("domain must be 'music' or 'movie' or 'sports'")

  
    time_stamp = os.path.splitext(os.path.basename(entity_file))[0]

    with open(entity_file, 'r', encoding='utf-8') as f:
        units = json.load(f)

    for source_lang in test_languages:
        print(f"\n=== Generating {domain} docs for language: {source_lang} ===")
        model = OpenAIModel('gpt-4o-mini', temperature=0.8, max_tokens=14000)

        save_dir = os.path.join(output_dir, time_stamp, source_lang)
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, unit in enumerate(units):
            title = unit.get('title', f"untitled_{idx}")
            safe_title = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title)
            save_path = os.path.join(save_dir, f"{safe_title}.json")

            if os.path.exists(save_path):
                print(f"[{idx+1}/{len(units)}] Skipping existing: {safe_title}")
                continue

            print(f"[{idx+1}/{len(units)}] {domain.capitalize()} | {safe_title} | Lang={source_lang}")
            doc = get_train_doc(model, unit, source_lang, domain, templates)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({"fact_source": doc}, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str,
        default="movie", choices=["movie", "music", "sports"]
    )
    parser.add_argument(
        "--entity_file", type=str, 
        default="data/entites/movie/2025-04-01_2025-06-30.json"
    )
    parser.add_argument(
        "--output_dir", type=str, 
        default="data/train_docs/movie"
    )
    parser.add_argument(
        "--test_languages",
        type=str,
        nargs='+',
        default=["en", "ja", "fr", "es", "zh"],
        help="List of test language codes"
    )
    args = parser.parse_args()

    main(
        args.entity_file, 
        args.output_dir,
        args.test_languages,
        args.domain,
    )
