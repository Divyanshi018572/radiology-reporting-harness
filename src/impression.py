"""Stage 3 impression: changed findings -> concise summary via Anthropic."""
import os
import time

SYSTEM_PROMPT_FILE = "prompts/impression_system_prompt.txt"


def write_impression(changed_texts: list[str],
                     template_impression: str = "") -> str:
    """Summarize changed findings only, return impression string."""
    if not changed_texts:
        return template_impression.strip()
    with open(SYSTEM_PROMPT_FILE) as f:
        system = f.read()
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user = "Changed findings:\n" + "\n".join(f"- {t}"
                                             for t in changed_texts)
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                temperature=0.0, system=system,
                messages=[{"role": "user", "content": user}])
            return str(msg.content[0].text).strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
