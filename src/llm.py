"""
llm.py — LLM factory. Supports OpenAI and Groq seamlessly.
Switch providers by changing LLM_PROVIDER in .env — no code change needed.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from src.config import settings
from src.logger import logger


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached LLM instance based on the configured provider."""
    provider = settings.llm_provider.lower()
    logger.info(f"Initialising LLM — provider={provider}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            streaming=True,
        )
        logger.info(f"OpenAI LLM ready — model={settings.openai_model}")
        return llm

    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.temperature,
            streaming=True,
        )
        logger.info(f"Groq LLM ready — model={settings.groq_model}")
        return llm

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Choose 'openai' or 'groq' in .env"
        )


def get_provider_info() -> dict:
    """Return display info about the active LLM."""
    p = settings.llm_provider.lower()
    if p == "openai":
        return {"provider": "OpenAI", "model": settings.openai_model, "icon": "🟢"}
    elif p == "groq":
        return {"provider": "Groq (FREE)", "model": settings.groq_model, "icon": "⚡"}
    return {"provider": p, "model": "unknown", "icon": "🤖"}
