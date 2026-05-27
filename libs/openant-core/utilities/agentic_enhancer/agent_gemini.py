"""
Gemini-compatible Context Enhancer (no tool use)

Simplified version of agent.py that works with Gemini by using a single
prompt instead of iterative tool use. Less accurate than the Claude version
but functional without Anthropic API.
"""

import json
import os
from typing import Optional, Set, List

from ..llm_client import TokenTracker, get_global_tracker
from ..llm_factory import create_llm_client
from ..rate_limiter import get_rate_limiter
from .repository_index import RepositoryIndex
from .prompts import SYSTEM_PROMPT, get_user_prompt
from .entry_point_detector import EntryPointDetector
from .reachability_analyzer import ReachabilityAnalyzer
from .agent import AgentResult  # Reuse AgentResult from original

AGENT_MODEL = os.environ.get("OPENANT_LLM_MODEL", "gemini-2.5-flash")
LLM_PROVIDER = os.environ.get("OPENANT_LLM_PROVIDER", "google")
MAX_TOKENS_PER_RESPONSE = 4096

GEMINI_SYSTEM_PROMPT = """You are a security analysis agent. Analyze the provided code and return a JSON object with your findings.

You must respond with ONLY a valid JSON object, no markdown, no explanation. The JSON must have these exact fields:
{
  "include_functions": [],
  "usage_context": "description of how this code is used",
  "security_classification": "one of: exploitable, vulnerable_internal, security_control, neutral",
  "classification_reasoning": "explanation of your classification",
  "confidence": 0.8
}

security_classification values:
- exploitable: vulnerable and reachable from user input
- vulnerable_internal: vulnerable but not user-reachable  
- security_control: defensive/validation code
- neutral: no security relevance
"""


class GeminiContextAgent:
    """
    Simplified agent for Gemini — uses single prompt instead of tool use.
    """

    def __init__(
        self,
        index: RepositoryIndex,
        tracker: TokenTracker = None,
        verbose: bool = False,
        entry_points: Optional[Set[str]] = None,
        reachability: Optional[ReachabilityAnalyzer] = None,
        client=None,
    ):
        self.index = index
        self.tracker = tracker or get_global_tracker()
        self.verbose = verbose
        self.entry_points = entry_points or set()
        self.reachability = reachability
        self.client = client or create_llm_client(
            provider=LLM_PROVIDER,
            model=AGENT_MODEL,
            tracker=self.tracker
        )

    def analyze_unit(
        self,
        unit_id: str,
        unit_type: str,
        primary_code: str,
        static_deps: list,
        static_callers: list
    ) -> AgentResult:
        is_entry_point = unit_id in self.entry_points
        reachable_from_entry = None
        entry_point_path = None

        if self.reachability:
            reachable_from_entry = self.reachability.is_reachable_from_entry_point(unit_id)
            if reachable_from_entry:
                entry_point_path = self.reachability.get_entry_point_path(unit_id)

        # Build prompt
        user_prompt = get_user_prompt(
            unit_id=unit_id,
            unit_type=unit_type,
            primary_code=primary_code,
            static_deps=static_deps,
            static_callers=static_callers,
            is_entry_point=is_entry_point,
            reachable_from_entry=reachable_from_entry,
            entry_point_path=entry_point_path,
            reaching_entry_point=None
        )

        # Add callers context from index
        callers_code = []
        for caller_id in (static_callers or [])[:3]:  # max 3 callers
            func = self.index.get_function(caller_id)
            if func and func.get("code"):
                callers_code.append(f"// Caller: {caller_id}\n{func['code']}")

        if callers_code:
            user_prompt += "\n\n## Caller Code\n" + "\n\n".join(callers_code)

        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed()

        try:
            result_text = self.client.analyze_sync(
                user_prompt,
                max_tokens=MAX_TOKENS_PER_RESPONSE,
                system=GEMINI_SYSTEM_PROMPT
            )
        except Exception as exc:
            if self.verbose:
                print(f"  Gemini error: {exc}")
            return AgentResult(
                include_functions=[],
                usage_context="Analysis failed",
                security_classification="neutral",
                classification_reasoning=str(exc),
                confidence=0.0,
                iterations=1,
                total_tokens=0,
                is_entry_point=is_entry_point,
                reachable_from_entry=reachable_from_entry,
                entry_point_path=entry_point_path
            )

        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        return AgentResult(
            include_functions=data.get("include_functions", []),
            usage_context=data.get("usage_context", ""),
            security_classification=data.get("security_classification", "neutral"),
            classification_reasoning=data.get("classification_reasoning", ""),
            confidence=float(data.get("confidence", 0.5)),
            iterations=1,
            total_tokens=0,
            is_entry_point=is_entry_point,
            reachable_from_entry=reachable_from_entry,
            entry_point_path=entry_point_path
        )


def enhance_unit_with_agent(
    unit: dict,
    index: RepositoryIndex,
    tracker: TokenTracker = None,
    verbose: bool = False,
    entry_points: Optional[Set[str]] = None,
    reachability: Optional[ReachabilityAnalyzer] = None,
    client=None,
) -> dict:
    """Drop-in replacement for agent.enhance_unit_with_agent using Gemini."""

    agent = GeminiContextAgent(
        index=index,
        tracker=tracker,
        verbose=verbose,
        entry_points=entry_points,
        reachability=reachability,
        client=client,
    )

    unit_id = unit.get("id", "unknown")
    unit_type = unit.get("unit_type", "function")
    code_section = unit.get("code", {})
    primary_code = code_section.get("primary_code", "")
    static_deps = unit.get("metadata", {}).get("direct_calls", [])
    static_callers = unit.get("metadata", {}).get("direct_callers", [])

    result = agent.analyze_unit(
        unit_id=unit_id,
        unit_type=unit_type,
        primary_code=primary_code,
        static_deps=static_deps,
        static_callers=static_callers
    )

    unit["agent_context"] = result.to_dict()
    return unit
