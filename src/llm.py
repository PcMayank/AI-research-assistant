"""
llm.py — LLM factory. Supports OpenAI, Groq, and Gemini.
Switch providers by changing LLM_PROVIDER in .env — no code change needed.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from src.config import settings
from src.logger import logger

# Deprecated Gemini model names → current replacement
_GEMINI_UPGRADES = {
    "gemini-1.5-flash": "gemini-2.0-flash",
    "gemini-1.5-flash-latest": "gemini-2.0-flash",
    "gemini-1.5-pro": "gemini-2.0-flash",
    "gemini-1.5-pro-latest": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.0-flash",
}


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached LLM instance based on the configured provider."""
    provider = settings.llm_provider.lower()
    logger.info(f"Initialising LLM — provider={provider}")

    try:
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

        elif provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is not installed.\n"
                    "Run: pip install langchain-google-genai==1.0.10 google-generativeai==0.7.2"
                )

            model = settings.gemini_model.strip()
            # Strip accidental "models/" prefix
            if model.startswith("models/"):
                model = model[len("models/"):]
            # Auto-upgrade deprecated model names
            if model in _GEMINI_UPGRADES:
                upgraded = _GEMINI_UPGRADES[model]
                logger.warning(
                    f"Gemini model '{model}' is deprecated. "
                    f"Auto-upgrading to '{upgraded}'. "
                    f"Set GEMINI_MODEL={upgraded} in .env to silence this."
                )
                model = upgraded

            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.temperature,
                streaming=True,
            )
            logger.info(f"Gemini LLM ready — model={model}")
            return llm

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER='{provider}'. "
                f"Choose 'openai', 'groq', or 'gemini' in .env"
            )

    except Exception:
        # Clear cache so next call retries instead of returning a broken instance
        get_llm.cache_clear()
        raise


def get_provider_info() -> dict:
    """Return display info about the active LLM."""
    p = settings.llm_provider.lower()
    if p == "openai":
        return {"provider": "OpenAI", "model": settings.openai_model, "icon": "🟢"}
    elif p == "groq":
        return {"provider": "Groq (FREE)", "model": settings.groq_model, "icon": "⚡"}
    elif p == "gemini":
        model = settings.gemini_model.strip()
        if model.startswith("models/"):
            model = model[len("models/"):]
        model = _GEMINI_UPGRADES.get(model, model)
        return {"provider": "Gemini (FREE)", "model": model, "icon": "♊"}
    return {"provider": p, "model": "unknown", "icon": "🤖"}