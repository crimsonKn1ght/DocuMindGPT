"""
DocuMindGPT – Evaluation Agent
Acts as a QA auditor: checks the generated answer for hallucination
and relevance relative to the retrieved context and user query.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

EVAL_SYSTEM_INSTRUCTION: str = (
    "You are a strict QA auditor. You will be given a User Query, the Context that was retrieved "
    "from a knowledge base, and the Generated Answer produced by another AI.\n\n"
    "Your task:\n"
    "1. Hallucination Check – Does the answer contain any facts or claims NOT supported by the context?\n"
    "2. Relevance Check – Does the answer actually address the user query?\n\n"
    "Respond ONLY with a JSON object (no markdown fences) in this exact schema:\n"
    '{"score": <int 1-10>, "verdict": "<Pass or Fail>", "reasoning": "<brief explanation>"}\n\n'
    "Scoring guide:\n"
    "  9-10  = Fully grounded in context, directly answers the query.\n"
    "  7-8   = Mostly grounded, minor gaps or slight tangents.\n"
    "  4-6   = Partially grounded; some unsupported claims or only partially relevant.\n"
    "  1-3   = Mostly hallucinated or off-topic.\n"
    "Verdict: Pass if score >= 7, else Fail."
)


def _build_eval_prompt(query: str, context: str, answer: str) -> str:
    """Compose the evaluation prompt."""
    return (
        f"User Query:\n{query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"Generated Answer:\n{answer}"
    )


def _parse_eval_response(raw: str) -> Dict[str, Any]:
    """Parse the model output into a structured evaluation dict."""
    # Strip any accidental markdown fences
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse evaluation JSON: %s", raw[:200])
        return {
            "score": 0,
            "verdict": "Error",
            "reasoning": f"Could not parse evaluator response: {raw[:300]}",
        }

    # Normalise fields
    score = int(data.get("score", 0))
    verdict = str(data.get("verdict", "Fail"))
    reasoning = str(data.get("reasoning", ""))
    return {"score": score, "verdict": verdict, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    user_query: str,
    context: str,
    answer: str,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a generated answer for hallucination and relevance.

    Returns
    -------
    dict with keys: score (int 1-10), verdict (str), reasoning (str).
    """
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=gemini_api_key)

    prompt: str = _build_eval_prompt(user_query, context, answer)

    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=EVAL_SYSTEM_INSTRUCTION,
        )
        response = model.generate_content(prompt)
        raw_text: str = response.text
    except Exception:
        logger.exception("Evaluation model call failed.")
        raise

    result: Dict[str, Any] = _parse_eval_response(raw_text)
    logger.info(
        "Evaluation complete – score: %d, verdict: %s", result["score"], result["verdict"]
    )
    return result
