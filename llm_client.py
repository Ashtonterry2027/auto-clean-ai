"""Lightweight LLM client scaffolding.

This module provides a small base class and an optional OpenAI-backed client.
The OpenAI client is lazy-loaded and only required if you intend to use it.

Usage:
  from llm_client import OpenAIClient
  client = OpenAIClient(api_key=os.environ.get('OPENAI_API_KEY'))
  pipeline = AutoCleanPipeline(file, out, llm_client=client)

The client must implement suggest_label(text, current_label) -> Optional[str]
which returns either a new label string or the existing/current label.
"""
from typing import Optional
import os


class BaseLLMClient:
    """Interface for LLM clients used by AutoCleanPipeline."""
    def suggest_label(self, text: str, current_label: Optional[str]) -> Optional[str]:
        raise NotImplementedError()


class OpenAIClient(BaseLLMClient):
    """Simple OpenAI client wrapper. Uses Chat Completions if openai is installed.

    This class is optional and will raise an informative error if the `openai`
    package isn't installed when used.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")

        try:
            import openai
        except Exception as e:
            raise RuntimeError("openai package is required for OpenAIClient. Install with `pip install openai`.") from e

        openai.api_key = self.api_key
        self._openai = openai

    def suggest_label(self, text: str, current_label: Optional[str]) -> Optional[str]:
        prompt = (
            f"Given the text below, suggest a concise label/category for it."
            f"Return only a single lower-case token label or the existing label if unsure.\n\nText: {text}\nCurrent label: {current_label}\nLabel:")

        resp = self._openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
        )

        label = resp.choices[0].message.content.strip()
        if not label:
            return current_label
        return label


class OllamaClient(BaseLLMClient):
    """Simple Ollama client wrapper. Expects an Ollama HTTP API endpoint.

    By default it uses `http://localhost:11434` but you can set `OLLAMA_URL`
    environment variable or pass `url` in the constructor.
    """
    def __init__(self, url: Optional[str] = None, model: str = "llama2"):
        self.url = url or os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        self.model = model

    def suggest_label(self, text: str, current_label: Optional[str]) -> Optional[str]:
        # Use requests to call Ollama; keep minimal and raise informative errors
        try:
            import requests
        except Exception as e:
            raise RuntimeError("requests is required for OllamaClient. Install requests.") from e

        payload = {
            'model': self.model,
            'input': f"Suggest a concise label for the following text. Return a single token.\n\nText: {text}\nCurrent label: {current_label}\nLabel:"
        }

        resp = requests.post(f"{self.url}/api/generate", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Ollama response shapes vary; try a few common keys
        label = None
        if isinstance(data, dict):
            label = data.get('output') or data.get('text') or None

        if not label:
            return current_label
        return str(label).strip()
