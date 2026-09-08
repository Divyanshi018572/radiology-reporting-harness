"""Tests for llm provider switch (no keys required)."""
import pandas as pd
from src import llm
from src.extract import extract_findings


def test_default_provider_is_groq_with_nvidia_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK", raising=False)
    assert llm.get_provider() == "groq"
    assert llm.get_fallback() == "nvidia"


def test_fallback_fires_when_primary_down(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_FALLBACK", "nvidia")
    monkeypatch.setattr(llm, "_groq_chat", lambda *a, **k: (_ for _ in ()).throw(ValueError("down")))
    monkeypatch.setattr(llm, "_nvidia_chat", lambda *a, **k: "OK")
    assert llm.chat("s", "u", retries=0) == "OK"


def test_normal_skips_llm_without_keys(monkeypatch):
    for k in ("GROQ_API_KEY", "NVIDIA_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    row = pd.Series({"dictation": "normal"})
    assert extract_findings(row) == []
