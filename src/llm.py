"""LLM provider switch: free local Ollama by default, Anthropic optional."""
import json
import os
import time
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")


def get_provider() -> str:
    """Return configured provider name, default ollama (free, no key)."""
    return os.environ.get("LLM_PROVIDER", "ollama").lower()


def _ollama_chat(system: str, user: str, json_mode: bool,
                 num_ctx: int = 8192) -> str:
    """POST to local Ollama /api/chat, return message content."""
    payload = {"model": OLLAMA_MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "stream": False, "options": {"temperature": 0.0,
                                            "num_ctx": num_ctx}}
    if json_mode:
        payload["format"] = "json"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return str(body["message"]["content"])


def _anthropic_chat(system: str, user: str, max_tokens: int) -> str:
    """Call Anthropic (paid, key required), return text."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(model="claude-sonnet-4-20250514",
                                 max_tokens=max_tokens, temperature=0.0,
                                 system=system,
                                 messages=[{"role": "user",
                                            "content": user}])
    return str(msg.content[0].text)


def chat(system: str, user: str, max_tokens: int = 2000,
         json_mode: bool = False, retries: int = 2) -> str:
    """Route to configured provider with backoff, return raw text."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if get_provider() == "anthropic":
                return _anthropic_chat(system, user, max_tokens)
            return _ollama_chat(system, user, json_mode)
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed ({get_provider()}): {last}")
