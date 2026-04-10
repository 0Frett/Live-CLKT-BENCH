

DOC_TEMPLATE = """
- Movie Title: {title}
- Movie Cast: {casts}
- Movie Summary: {summary}
- Movie Synopsis: {synopsis}
"""


DOC_TRANSLATE_TEMPLATE = """
Translate the following movie document into {lang}.
- Do NOT translate the field names (e.g., "Movie Cast", "Movie Summary", "Movie Synopsis").
- Translate only the values after the colon.
- If the text is already in {lang}, return it unchanged.
- Return only the JSON object, no extra text.

Document:
    - Movie Cast: {casts}
    - Movie Summary: {summary}
    - Movie Synopsis: {synopsis}

Output Format:
Return a single JSON object with this structure:
{{
    "translation": {{
        "Cast": "<translated cast>",
        "Summary": "<translated summary>",
        "Synopsis": "<translated synopsis>"
    }}
}}
"""


GEN_FACTQA_TEMPLATE = """
You are generating high-quality multiple-choice QA pairs in {lang}, strictly grounded in the given movie information.

You will be provided with:
- Movie Title
- Movie Casts
- Movie Summary
- Movie Synopsis

Task:
- Generate natural, audience-friendly questions that viewers might realistically ask.
- All questions must be written fully in {lang}, including the leading phrase (“In the movie: '<title>', …”).
- Each QA pair must be based ONLY on facts explicitly present in the input. Do not add, assume, or hallucinate.
- Use diverse aspects (casts, summary, synopsis content).

Each QA pair must include:
- **Question** in {lang}, beginning with “In the movie: '<title>', …” (do not translate title)
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
You are given a JSON object containing a list of question–answer (QA) pairs for a movie.

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
    - Any proper names (people, places, characters, movie name, entities).
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
Do you know the movie titled "{entity_name}"?
If you do, provide the cast and a short summary of the movie.
If you do not know it, output exactly: "I don't know"
"""

CHECK_TEMPLATE = """
You are verifying whether the model's response shows real knowledge of the movie.

Ground-truth movie information:
{doc_text}

Model response:
{response}

Task:
Determine whether the response is factually consistent with the ground-truth information.

Judging rules:
- If the model explicitly states that it does not know the movie (e.g., "I don't know"), return false.
- Return true only if the response correctly provides the cast and a summary of the movie.
- Minor wording differences or small omissions in the summary are acceptable.
- Major factual errors, incorrect cast, wrong movie, or fabricated details should be judged as false.

Output strictly in JSON:
{{"is_known": true/false, "reason": "short explanation"}}
"""

# these are in doc_text
# - Movie Title: {title}
# - Movie Cast: {casts}
# - Movie Summary: {summary}