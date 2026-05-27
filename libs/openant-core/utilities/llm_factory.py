"""
LLM Provider Factory

Factory pattern for creating and managing LLM clients (Anthropic, Google Vertex AI).
Provides abstraction layer for swapping between providers.
"""

from typing import Optional, Union, Literal
from enum import Enum

ProviderType = Literal["anthropic", "google"]


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


def create_llm_client(
    provider: ProviderType = "anthropic",
    model: Optional[str] = None,
    **kwargs
) -> Union["AnthropicClient", "GoogleVertexClient"]:
    """
    Factory function to create LLM clients.

    Args:
        provider: "anthropic" or "google"
        model: Model identifier. If None, uses provider default.
        **kwargs: Additional arguments passed to the client constructor.
                  - For Anthropic: tracker
                  - For Google: project_id, location, use_genai, api_key, tracker

    Returns:
        Instance of AnthropicClient or GoogleVertexClient

    Raises:
        ValueError: If provider is unknown
        ImportError: If required dependencies are not installed

    Examples:
        # Anthropic Claude
        client = create_llm_client("anthropic", model="claude-opus-4-20250514")

        # Google Gemini 2.5 Flash via Vertex AI
        client = create_llm_client(
            "google",
            model="gemini-2.5-flash",
            project_id="my-project"
        )

        # Google Gemini via generativeai SDK
        client = create_llm_client(
            "google",
            model="gemini-2.5-flash",
            use_genai=True,
            api_key="your-api-key"
        )
    """
    if provider == "anthropic" or provider == LLMProvider.ANTHROPIC.value:
        from utilities.llm_client import AnthropicClient
        
        if model is None:
            model = "claude-opus-4-20250514"
        
        return AnthropicClient(model=model, **kwargs)

    elif provider == "google" or provider == LLMProvider.GOOGLE.value:
        import os
        from utilities.gemini_client import GoogleVertexClient
        
        if model is None:
            model = "gemini-2.5-flash"
        
        # Check environment to determine SDK (genai vs Vertex AI)
        # GOOGLE_GENAI_USE_VERTEXAI=True means use Vertex AI SDK
        # GOOGLE_GENAI_USE_VERTEXAI=False/unset means use genai SDK
        use_vertex_ai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
        kwargs.setdefault("use_genai", not use_vertex_ai)
        
        return GoogleVertexClient(model=model, **kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'anthropic' or 'google'")


def get_default_client(provider: ProviderType = "anthropic", **kwargs):
    """
    Get a default LLM client for the specified provider.

    Args:
        provider: "anthropic" or "google"
        **kwargs: Additional client-specific arguments

    Returns:
        Initialized LLM client with sensible defaults
    """
    return create_llm_client(provider, **kwargs)
