"""Stage 1 extraction: dictation -> clause-level findings JSON via LLM."""
import json
import pandas as pd
from src.llm import chat

SYSTEM_PROMPT_FILE = "prompts/extraction_system_prompt.txt"


def load_system_prompt() -> str:
    """Load extraction system prompt text, return string."""
    with open(SYSTEM_PROMPT_FILE) as f:
        return f.read()


def extract_findings(row: pd.Series) -> list[dict]:
    """Extract clause-level findings for one row, return list of dicts."""
    if str(row["dictation"]).strip().lower() == "normal":
        return []
    system = load_system_prompt()
    user = (f"Modality: {row['modality']}\nBody part: {row['body_part']}\n"
            f"Study: {row['study_description']}\n"
            f"Template:\n{row['template_content']}\n"
            f"Dictation:\n{row['dictation']}\n"
            "Return ONLY the JSON array.")
    raw = chat(system, user, max_tokens=2000, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raw2 = chat(system + f"\nFix this JSON error: {e}\n{raw}", user,
                    max_tokens=2000, json_mode=True)
        return json.loads(raw2)
