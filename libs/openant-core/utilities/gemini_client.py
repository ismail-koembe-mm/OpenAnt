"""
Google Vertex AI / Gemini LLM Client

Wrapper for Google Vertex AI and Generative AI API calls with token tracking and cost calculation.
Provides a compatible interface with AnthropicClient for easy provider swapping.

Classes:
    GoogleVertexClient: Client for Vertex AI Generative API or Google Generative AI API

Usage:
    from utilities.gemini_client import GoogleVertexClient, get_global_tracker

    client = GoogleVertexClient(
        model="gemini-2.5-flash",
        project_id="your-project-id"
    )
    response = client.analyze_sync("Analyze this code...")

    tracker = get_global_tracker()
    print(f"Total cost: ${tracker.total_cost_usd:.4f}")
"""

import os
import threading
from typing import Optional

# Force pure Python protobuf implementation for Python 3.14 compatibility
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv

from .rate_limiter import get_rate_limiter


# Pricing per million tokens (as of May 2026 - Gemini 2.5 Flash rates)
MODEL_PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    # Fallback for unknown models (use Flash pricing as conservative estimate)
    "default": {"input": 0.075, "output": 0.30}
}


class TokenTracker:
    """
    Tracks token usage and costs across LLM calls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self.reset()

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self.calls = []
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def record_call(self, model: str, input_tokens: int, output_tokens: int) -> dict:
        """
        Record a single LLM call.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dict with call details including cost
        """
        # Get pricing for model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

        # Calculate cost (pricing is per million tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        call_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(total_cost, 6)
        }

        # Update totals (thread-safe)
        with self._lock:
            self.calls.append(call_record)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += total_cost

        # Accumulate to thread-local unit tracking if active
        tl = self._thread_local
        if hasattr(tl, "unit_input"):
            tl.unit_input += input_tokens
            tl.unit_output += output_tokens
            tl.unit_cost += total_cost

        return call_record

    def add_prior_usage(self, input_tokens: int, output_tokens: int, cost_usd: float):
        """Inject usage from a prior run (e.g. restored checkpoints).

        This ensures step reports capture the total cost across all runs,
        not just the current run's API calls.
        """
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost_usd

    def start_unit_tracking(self):
        """Start tracking usage for the current unit on this thread.

        Call before processing a unit, then call ``get_unit_usage()``
        after to get the accumulated usage for just that unit. Thread-safe
        because each thread has its own ``threading.local()`` storage.
        """
        tl = self._thread_local
        tl.unit_input = 0
        tl.unit_output = 0
        tl.unit_cost = 0.0

    def get_unit_usage(self) -> dict:
        """Return usage accumulated since ``start_unit_tracking()`` on this thread."""
        tl = self._thread_local
        return {
            "input_tokens": getattr(tl, "unit_input", 0),
            "output_tokens": getattr(tl, "unit_output", 0),
            "cost_usd": round(getattr(tl, "unit_cost", 0.0), 6),
        }

    def get_summary(self) -> dict:
        """
        Get summary of all tracked calls.

        Returns:
            Dict with totals and per-call breakdown
        """
        with self._lock:
            return {
                "total_calls": len(self.calls),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "calls": list(self.calls),
            }

    def get_totals(self) -> dict:
        """
        Get just the totals (without per-call breakdown).

        Returns:
            Dict with totals only
        """
        with self._lock:
            return {
                "total_calls": len(self.calls),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
            }


# Global tracker instance for session-wide tracking
_global_tracker = TokenTracker()


def get_global_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return _global_tracker


def reset_global_tracker():
    """Reset the global token tracker."""
    _global_tracker.reset()


class GoogleVertexClient:
    """
    Client for Google Vertex AI Generative API.

    Provides compatible interface with AnthropicClient for provider swapping.
    Uses either Vertex AI (via google-cloud-aiplatform) or Google Generative AI SDK.
    """

    def __init__(self, model: str = "gemini-2.5-flash", project_id: str = None, 
                 location: str = "us-central1", use_genai: bool = False, 
                 api_key: str = None, tracker: TokenTracker = None):
        """
        Initialize the Google Vertex AI client.

        Args:
            model: Model identifier. Default is Gemini 2.5 Flash.
                  Examples: "gemini-2.5-flash", "gemini-1.5-pro"
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            location: GCP region. Default: us-central1
            use_genai: If True, use google-generativeai SDK instead of Vertex AI.
                      Set to True if you have a direct API key instead of service account.
            api_key: API key for google-generativeai. Used when use_genai=True.
            tracker: Optional TokenTracker instance. Uses global tracker if not provided.
        """
        load_dotenv()

        self.model = model
        self.tracker = tracker or _global_tracker
        self.last_call = None
        self.use_genai = use_genai

        if use_genai:
            # Use google-generativeai SDK
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai is required. Install with: pip install google-generativeai"
                )

            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment or provided as argument")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name=model)
            self.sdk_type = "genai"
        else:
            # Use Vertex AI SDK
            try:
                import google.cloud.aiplatform as aiplatform
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform is required. Install with: pip install google-cloud-aiplatform"
                )

            project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment or provided as argument")

            aiplatform.init(project=project_id, location=location)


            self.project_id = project_id
            self.location = location
            self.endpoint_id = None
            self.sdk_type = "vertex"

    async def analyze(self, prompt: str, max_tokens: int = 8192) -> str:
        """
        Send a prompt to Gemini and get a response (async).

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response

        Returns:
            Response text from Gemini
        """
        # For now, just call sync version
        # (Proper async implementation would use asyncio)
        return self.analyze_sync(prompt, max_tokens)

    def analyze_sync(self, prompt: str, max_tokens: int = 8192, model: str = None, system: str = None) -> str:
        """
        Synchronous version of analyze.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            model: Optional model override (uses instance model if not specified)
            system: Optional system prompt for context/instructions

        Returns:
            Response text from Gemini
        """
        used_model = model or self.model

        # Wait if we're in a global backoff period
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed()

        try:
            if self.use_genai:
                response = self._call_genai(prompt, max_tokens, system)
            else:
                response = self._call_vertex_ai(prompt, max_tokens, system)

            # Extract text and token counts
            response_text = response.get("text", "")
            input_tokens = response.get("input_tokens", 0)
            output_tokens = response.get("output_tokens", 0)

            # Track token usage
            self.last_call = self.tracker.record_call(
                model=used_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            return response_text

        except Exception as exc:
            # Check for rate limit errors
            error_str = str(exc).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                retry_after = 60  # Default backoff
                get_rate_limiter().report_rate_limit(retry_after)
            raise

    def _call_genai(self, prompt: str, max_tokens: int, system: str = None) -> dict:
        """Call google-generativeai API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed")

        # Build content with system prompt if provided
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Call API
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )

        # Extract token counts (genai API returns usage_metadata)
        usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return {
            "text": response.text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _call_vertex_ai(self, prompt: str, max_tokens: int, system: str = None) -> dict:
        """Call Vertex AI Generative API."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError:
            raise ImportError("vertexai is required. Install with: pip install google-cloud-aiplatform")
        vertexai.init(project=self.project_id, location=self.location)
        model = GenerativeModel(self.model)
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt
        response = model.generate_content(
            [full_prompt],
            generation_config=GenerationConfig(max_output_tokens=max_tokens),
        )
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        input_tokens = usage_metadata.prompt_token_count if usage_metadata else 0
        output_tokens = usage_metadata.candidates_token_count if usage_metadata else 0
        return {
            "text": response.text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    def get_last_call(self) -> Optional[dict]:
        """
        Get details of the last API call.

        Returns:
            Dict with model, input_tokens, output_tokens, cost_usd
        """
        return self.last_call

    def get_session_totals(self) -> dict:
        """
        Get cumulative totals for this session.

        Returns:
            Dict with total_calls, total_input_tokens, total_output_tokens, total_cost_usd
        """
        return self.tracker.get_totals()

    def get_session_summary(self) -> dict:
        """
        Get full summary including per-call breakdown.

        Returns:
            Dict with totals and calls list
        """
        return self.tracker.get_summary()

    def get_usage(self, message) -> dict:
        """
        Extract token usage from a message response.

        Args:
            message: Response from generate_content()

        Returns:
            Dict with input_tokens, output_tokens
        """
        usage = message.usage_metadata if hasattr(message, 'usage_metadata') else None
        return {
            "input_tokens": usage.prompt_token_count if usage else 0,
            "output_tokens": usage.candidates_token_count if usage else 0,
        }
