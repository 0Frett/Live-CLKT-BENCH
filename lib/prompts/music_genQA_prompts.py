
DOC_TEMPLATE = """
- Music Video Title: {title}
- Music Release Date: {date}
- Music Video Description: {description}
"""


DOC_TRANSLATE_TEMPLATE = """
Translate the following music video description into {lang}.
- Do NOT translate the field name "Description".
- Translate only the value of the description.
- If the description is already in {lang}, return it unchanged.
- Return ONLY the JSON object, no extra text.

Description: {description}

Output Format:
Return a single JSON object with this structure:
{{
    "Description": "<translated value here>"
}}
"""


GEN_FACTQA_TEMPLATE = """
You are generating high-quality multiple-choice QA pairs in {lang}, strictly grounded in the given YouTube music video information.

You will be provided with:
- Music Video Title
- Music Release Date
- Music Video Description

Task:
- Generate natural, audience-friendly questions that viewers might realistically ask.
- All questions must be written fully in {lang}, including the leading phrase (“In the music video: '<title>', …”).
- Each QA pair must be based ONLY on facts explicitly present in the input. Do not add, assume, or hallucinate.
- Use diverse aspects (title, release date, description details).

Each QA pair must include:
- **Question** in {lang}, beginning with “In the music video: '<title>', …” (do not translate title)
- **Options**:
    - Provide four options labeled A, B, C, D.
    - Exactly one option is correct.
    - Place the correct option randomly among A–D (do not always use the same position).
    - Distractors must be plausible but wrong (no random, absurd, or unrelated answers).
- **Correct Option**: Output the letter (A, B, C, or D) of the correct answer.

--------------------------
Inputs:
{meta_data}

--------------------------
Output Format (JSON):
{{
    "QA": [
        {{
            "question": "<string in {lang}>",
            "options": {{
                "A": "<option A in {lang}>",
                "B": "<option B in {lang}>",
                "C": "<option C in {lang}>",
                "D": "<option D in {lang}>"
            }},
            "correct_option": "<A | B | C | D>"
        }},
        ...
    ]
}}

--------------------------
Guidelines:
- Write everything (questions and options) only in {lang}.
- Keep all proper names (people, places, entities) unchanged.
- Ensure every correct answer can be directly verified in the input metadata.
- Distractors must be reasonable, related, and plausible.
"""


FACTQA_VERIFIER_PROMPT = """
You are verifying if QA pairs are grounded in the provided metadata.

Metadata:
{meta_data}

QA to verify:
{qa_item}

Check carefully:
- Is the correct option explicitly supported by the metadata?
- If yes, return SUPPORTED and also provide the exact sentence(s) from metadata that support it.
- If the correct option does not appear in or cannot be inferred from the metadata, return UNSUPPORTED.

Output Format:
Return a single JSON object with this structure:
{{
    "Decision": "<SUPPORTED or UNSUPPORTED>",
    "SourceSentence": "<exact sentence(s) from metadata that justify the decision, or empty string if UNSUPPORTED>"
}}
"""



FACTQA_TRANSLATE_TEMPLATE = """
You are given a JSON object containing a list of question–answer (QA) pairs for a music video.

Each QA entry contains:
A multiple-choice question with four answer options and the correct one marked

--------------------------
Example Input Format:
{{
    "QA": [
        {{
            "question": "<string>",
            "options": {{
                "A": "<option A>",
                "B": "<option B>",
                "C": "<option C>",
                "D": "<option D>"
            }},
            "correct_option": "<A | B | C | D>"
        }},
        ...
    ]
}}

--------------------------
Your task:
1. Translate only the **values** of "question" and "options" into the target language.
2. Do NOT translate or modify:
    - The field names ("QA", "question", "options", "correct_option", "A", "B", "C", "D").
    - The "correct_option" value (keep it exactly as "A", "B", "C", or "D").
    - Any proper names (people, places, songs, entities).
3. Keep the JSON structure, formatting, and order exactly the same as input.

--------------------------
Input QA JSON:
{qa}

Language: {lang_code}

--------------------------
Output:
Return the translated JSON object with the same structure and unchanged keys.
"""



IS_KNOWN_ENTITY_PROMPT = """
Do you know the music video titled "{entity_name}"?
If you do, provide the artist, singer, and release date.
If you do not know it, output exactly: "I don't know"
"""


CHECK_TEMPLATE = """
You are verifying whether the model's response shows real knowledge of the music video.

Ground-truth music video information:
{doc_text}

Model response:
{response}

Task:
Determine whether the response is factually consistent with the ground-truth information.

Judging rules:
- If the model explicitly states that it does not know the music video, return false.
- Return true only if the response correctly provides the artist, singer, or release date.
- Minor wording differences or small omissions are acceptable.
- Major factual errors, wrong artist, wrong video, or fabricated details should be judged as false.

Output strictly in JSON: 
{{"is_known": true/false, "reason": "short explanation"}}
"""

# these are in doc_text
# - Music Video Title: {title}
# - Music Release Date: {date}
# - Music Video Description: {description}
