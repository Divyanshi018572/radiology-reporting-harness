"""LLM provider switch: Groq primary, NVIDIA fallback."""
import json
import os
import time
import urllib.request

GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines into environ (no override), return None."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    except FileNotFoundError:
        pass


_load_dotenv()


def get_provider() -> str:
    """Return primary provider name, default groq."""
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def get_fallback() -> str:
    """Return fallback provider name, default nvidia."""
    return os.environ.get("LLM_FALLBACK", "nvidia").lower()


def _openai_compat(base: str, key: str, model: str, system: str, user: str,
                   max_tokens: int, json_mode: bool) -> str:
    """POST OpenAI-style chat completions, return message content."""
    payload: dict = {"model": model,
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                     "temperature": 0.0, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        base, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())
    return str(body["choices"][0]["message"]["content"])


def _groq_chat(system: str, user: str, max_tokens: int,
               json_mode: bool) -> str:
    """Call Groq (free tier, key required), return text."""
    return _openai_compat(
        GROQ_BASE, os.environ["GROQ_API_KEY"],
        os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        system, user, max_tokens, json_mode)


def _nvidia_chat(system: str, user: str, max_tokens: int,
                 json_mode: bool) -> str:
    """Call NVIDIA Build (free tier, key required), return text."""
    return _openai_compat(
        NVIDIA_BASE, os.environ["NVIDIA_API_KEY"],
        os.environ.get("NVIDIA_MODEL",
                       "meta/llama-3.2-11b-vision-instruct"),
        system, user, max_tokens, json_mode)


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


def _model_for(provider: str) -> str:
    """Return model id for provider from environ, return string."""
    if provider == "groq":
        return os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    if provider == "nvidia":
        return os.environ.get("NVIDIA_MODEL",
                              "meta/llama-3.2-11b-vision-instruct")
    if provider == "anthropic":
        return "claude-sonnet-4-20250514"
    return provider


def _cache_path() -> str:
    """Return cache file path, return string."""
    return os.environ.get("LLM_CACHE", "outputs/cache.jsonl")


def _cache_key(provider: str, system: str, user: str, max_tokens: int,
               json_mode: bool) -> str:
    """Hash request into stable key, return hex digest."""
    import hashlib
    raw = "\n".join([provider, _model_for(provider), system, user,
                     str(max_tokens), str(json_mode)])
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_lookup(key: str) -> str | None:
    """Return cached response or None, return text or None."""
    import json as _json
    try:
        with open(_cache_path()) as f:
            for line in f:
                try:
                    rec = _json.loads(line)
                except ValueError:
                    continue
                if rec.get("key") == key:
                    return str(rec.get("response"))
    except FileNotFoundError:
        pass
    return None


def _cache_store(key: str, response: str) -> None:
    """Append response to cache file, return None."""
    import json as _json
    os.makedirs(os.path.dirname(_cache_path()) or ".", exist_ok=True)
    with open(_cache_path(), "a") as f:
        f.write(_json.dumps({"key": key, "response": response}) + "\n")


def _once(provider: str, system: str, user: str, max_tokens: int,
          json_mode: bool) -> str:
    """Single attempt on one provider, return text."""
    if provider == "groq":
        return _groq_chat(system, user, max_tokens, json_mode)
    if provider == "nvidia":
        return _nvidia_chat(system, user, max_tokens, json_mode)
    if provider == "anthropic":
        return _anthropic_chat(system, user, max_tokens)
    raise ValueError(f"unknown LLM provider: {provider}")


def chat(system: str, user: str, max_tokens: int = 2000,
         json_mode: bool = False, retries: int = 2) -> str:
    """Try primary, then fallback, with backoff and cache; return raw text."""
    providers = [get_provider(), get_fallback()]
    last: Exception | None = None
    for provider in providers:
        hit = _cache_lookup(_cache_key(provider, system, user, max_tokens,
                                       json_mode))
        if hit is not None:
            return hit
        for attempt in range(retries + 1):
            try:
                out = _once(provider, system, user, max_tokens, json_mode)
                _cache_store(_cache_key(provider, system, user, max_tokens,
                                        json_mode), out)
                return out
            except Exception as e:
                last = e
                if attempt < retries:
                    time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed ({providers}): {last}")
