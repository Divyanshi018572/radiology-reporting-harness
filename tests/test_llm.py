"""Tests for llm provider switch (no daemon/key required)."""
import pandas as pd
from src import llm
from src.extract import extract_findings


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm.get_provider() == "ollama"


def test_normal_skips_llm_without_key_or_daemon(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    row = pd.Series({"dictation": "normal"})
    assert extract_findings(row) == []
