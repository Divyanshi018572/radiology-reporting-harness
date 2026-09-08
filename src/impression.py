"""Stage 3 impression: changed findings -> concise summary via LLM."""
from src.llm import chat

SYSTEM_PROMPT_FILE = "prompts/impression_system_prompt.txt"


def write_impression(changed_texts: list[str],
                     template_impression: str = "") -> str:
    """Summarize changed findings only, return impression string."""
    if not changed_texts:
        return template_impression.strip()
    with open(SYSTEM_PROMPT_FILE) as f:
        system = f.read()
    user = "Changed findings:\n" + "\n".join(f"- {t}"
                                             for t in changed_texts)
    return chat(system, user, max_tokens=500).strip()
