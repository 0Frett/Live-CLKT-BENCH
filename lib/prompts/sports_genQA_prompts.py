
DOC_TEMPLATE = """
Sports: {sports}
League: {league}

Match: {home_team} vs {away_team}
Date: {date}
Score: {home_score} - {away_score}
Venue: {venue}

Innings Breakdown:
{home_team}: {home_innings} → Hits: {home_hits}, Errors: {home_errors}
{away_team}: {away_innings} → Hits: {away_hits}, Errors: {away_errors}
"""


def build_doc(unit: dict) -> str:
    game_info = unit["game_info"]
    league = game_info.get("league", "")
    sports = game_info.get("sports", "")
    date = game_info.get("date", "")
    home_team = game_info.get("home_team", "")
    away_team = game_info.get("away_team", "")
    home_score = game_info.get("score", {}).get("home", "")
    away_score = game_info.get("score", {}).get("away", "")
    venue = game_info.get("match_details", {}).get("venue", "")

    parsed = game_info.get("match_details", {}).get("parsed_result", {})
    home_stats = parsed.get(home_team, {})
    away_stats = parsed.get(away_team, {})

    def format_innings(innings):
        if isinstance(innings, list):
            return " ".join(str(x) for x in innings)
        return str(innings)

    return DOC_TEMPLATE.format(
        sports=sports,
        league=league,
        date=date,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        venue=venue,
        home_innings=format_innings(home_stats.get("innings", [])),
        away_innings=format_innings(away_stats.get("innings", [])),
        home_hits=home_stats.get("hits", ""),
        away_hits=away_stats.get("hits", ""),
        home_errors=home_stats.get("errors", ""),
        away_errors=away_stats.get("errors", ""),
    )


DOC_TRANSLATE_TEMPLATE = """
Translate the following sports game information into {lang}.

Instructions:
- Translate both the field labels and their values.
- Preserve the exact structure, formatting, line breaks, colons, arrows, and symbols.
- Do NOT add explanations, summaries, or extra text.
- Output only the translated text.

Input:
{text}

{lang} translation of Input:
"""


GEN_FACTQA_TEMPLATE = """
You are generating high-quality multiple-choice QA pairs in {lang}, strictly grounded in the provided sports game information.

You will be provided with:
- Sports Game Title
- Date
- League
- Teams
- Score
- Game Details / Stats

Task:
- Generate natural, audience-friendly questions that game viewer might realistically ask about the game.
- All questions must be written fully in {lang}, including the leading phrase (“In the sports game: '<title>' at '<date>', …”).
- Each QA pair must be based ONLY on facts explicitly present in the input. Do not add, assume, or hallucinate.
- Use diverse aspects (teams, score, venue, stats, notable events).
- Ensure the question is clear and unambiguous.

Each QA pair must include:
- **Question** in {lang}. 
- **Options**:
    - Provide four options labeled A, B, C, D.
    - Exactly one option is correct.
    - Place the correct option randomly among the four choice.
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
- Ensure every correct answer can be directly verified in the input metadata.
- Distractors must be reasonable, related, and plausible.
"""


FACTQA_VERIFIER_PROMPT = """
You are verifying if QA pairs are grounded in the provided sports game metadata.

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
You are given a JSON object containing a list of multiple-choice QA pairs for a sports game.

Each QA entry contains:
A question with four answer options and the correct one marked

--------------------------
Your task:
1. Translate only the **values** of "question" and "options" into the target language.
2. Do NOT translate or modify:
    - The field names ("QA", "question", "options", "correct_option", "A", "B", "C", "D").
    - The "correct_option" value (keep it exactly as "A", "B", "C", or "D").
    - Any proper names (teams, players, leagues, stadiums).
3. Keep the JSON structure, formatting, and order exactly the same as input.
4. Keep question natural and fluent in the target language.

--------------------------
Input QA JSON:
{qa}

Language: {lang_code}

--------------------------
Output:
Return the translated JSON object with the same structure and unchanged keys.
"""



IS_KNOWN_ENTITY_PROMPT = """
Do you know the result of the MLB game "{entity_name}"?
If you do, provide the venue and the final score of each team.
If you do not know it, output exactly: "I don't know"
"""


CHECK_TEMPLATE = """
You are verifying whether the model's response align with real knowledge of the MLB game.

Ground-truth game information:
{doc_text}

Model response:
{response}

Task:
Determine whether the response is factually consistent with the ground-truth information.

Judging rules:
- If the model explicitly states that it does not know the game, return false.
- Return true only if the response provides the correct final score and the correct venue.
- Minor formatting differences in the score or venue name are acceptable.
- Incorrect score, wrong teams, wrong game, or fabricated details should be judged as false.
- If the response is vague or lacks concrete information, return false.

Output strictly in JSON:
{{"is_known": true/false, "reason": "short explanation"}}
"""



# DOC_TEMPLATE = """
# Sports: {sports}
# League: {league}

# Match: {home_team} vs {away_team}
# Date: {date}
# Score: {home_score} - {away_score}
# Venue: {venue}

# Innings Breakdown:
# {home_team}: {home_innings} → Hits: {home_hits}, Errors: {home_errors}
# {away_team}: {away_innings} → Hits: {away_hits}, Errors: {away_errors}
# """



ES_FR_DOC_TEMPLATE = """    
    Sports: {sports}
    League: {league}

    Match: {home_team} vs {away_team}
    Date: {date}
    Score: {home_score} - {away_score}

    Match Stats ({home_team} vs {away_team}):
    {stats_block}
"""

def build_es_fr_doc(unit: dict) -> str:
    game = unit["game_info"]
    stats = game.get("match_details")  # safe: can be None

    stat_fields = [
        ("Shots on Goal", "Shots on Goal"),
        ("Shots off Goal", "Shots off Goal"),
        ("Total Shots", "Total Shots"),
        ("Blocked Shots", "Blocked Shots"),
        ("Shots insidebox", "Shots Inside Box"),
        ("Shots outsidebox", "Shots Outside Box"),
        ("Fouls", "Fouls"),
        ("Corner Kicks", "Corner Kicks"),
        ("Offsides", "Offsides"),
        ("Ball Possession", "Ball Possession", "%"),
        ("Yellow Cards", "Yellow Cards"),
        ("Red Cards", "Red Cards"),
        ("Goalkeeper Saves", "Goalkeeper Saves"),
        ("Total passes", "Total Passes"),
        ("Passes accurate", "Passes Accurate"),
        ("Passes %", "Passes %", "%"),
        ("expected_goals", "Expected Goals (xG)"),
        ("goals_prevented", "Goals Prevented"),
    ]

    stats_lines = []
    if stats:  # only build if stats exist
        for key, label, *suffix in stat_fields:
            if key in stats:
                suf = suffix[0] if suffix else ""
                stats_lines.append(
                    f"- {label}: {stats[key]['home']}{suf} vs {stats[key]['away']}{suf}"
                )

    stats_block = "\n".join(stats_lines)

    return ES_FR_DOC_TEMPLATE.format(
        sports=unit.get("sports", "Football"),
        league=unit.get("league", "Ligue 1"),
        date=unit.get("date", ""),
        home_team=game["home_team"],
        away_team=game["away_team"],
        home_score=game["score"]["home"],
        away_score=game["score"]["away"],
        stats_block=stats_block,  # will be empty if no stats
    )

