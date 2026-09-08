"""Stage 1 extraction: dictation -> clause-level findings JSON via Anthropic."""
import json
import os
import time
import pandas as pd

SYSTEM_PROMPT_FILE = "prompts/extraction_system_prompt.txt"


def load_system_prompt() -> str:
    """Load extraction system prompt text, return string."""
    with open(SYSTEM_PROMPT_FILE) as f:
        return f.read()


def call_llm(system: str, user: str, temperature: float = 0.0,
              retries: int = 2) -> str:
    """Call Anthropic with backoff, return raw text."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for attempt in range(retries + 1):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=2000,
                temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}])
            return str(msg.content[0].text)
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(2 ** attempt)
            _ = e
    raise RuntimeError("unreachable")


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
    raw = call_llm(system, user, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raw2 = call_llm(system + f"\nFix this JSON error: {e}\n{raw}",
                         user, temperature=0.0)
        return json.loads(raw2)
