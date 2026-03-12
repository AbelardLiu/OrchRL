from __future__ import annotations

import re
import unicodedata
from typing import Any


_ANSWER_KEYS = (
    "expected_answer",
    "answer",
    "expected_answers",
    "gold_answer",
    "golden_answers",
    "target",
    "label",
)


def _extract_tag(text: str, tag: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _normalize_text(text: Any) -> str:
    raw = "" if text is None else str(text)
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _expected_candidates(metadata: dict[str, Any]) -> list[str]:
    expected = metadata.get("expected")
    if expected is None:
        prompt_row = metadata.get("prompt_row")
        if isinstance(prompt_row, dict):
            for key in _ANSWER_KEYS:
                if key in prompt_row and prompt_row[key] is not None:
                    expected = prompt_row[key]
                    break

    if expected is None:
        return []
    if isinstance(expected, (list, tuple)):
        return [candidate for candidate in (_normalize_text(item) for item in expected) if candidate]
    candidate = _normalize_text(expected)
    return [candidate] if candidate else []


def _compute_answer_score(predicted: str, expected_candidates: list[str]) -> float:
    """
    Compute reward score with partial credit:
    - 1.0: Exact match
    - 0.6: Candidate substring in prediction
    - 0.4: Prediction substring in candidate (partial match)
    - 0.0: No match
    """
    normalized_predicted = _normalize_text(predicted)
    if not normalized_predicted or not expected_candidates:
        return 0.0

    for candidate in expected_candidates:
        # Exact match - full reward
        if normalized_predicted == candidate:
            return 1.0

        # Candidate fully contained in prediction - high reward
        if candidate in normalized_predicted:
            # Calculate overlap ratio for better scoring
            overlap_ratio = len(candidate) / len(normalized_predicted)
            if overlap_ratio > 0.5:  # More than 50% of prediction is the answer
                return 0.8
            else:
                return 0.6

        # Prediction contained in candidate - partial reward
        if normalized_predicted in candidate:
            overlap_ratio = len(normalized_predicted) / len(candidate)
            if overlap_ratio > 0.5:  # More than 50% of answer is covered
                return 0.5
            else:
                return 0.3

    # Check for partial word overlap as last resort
    predicted_words = set(normalized_predicted.split())
    for candidate in expected_candidates:
        candidate_words = set(candidate.split())
        if predicted_words and candidate_words:
            overlap = predicted_words & candidate_words
            if overlap:
                overlap_ratio = len(overlap) / max(len(predicted_words), len(candidate_words))
                if overlap_ratio > 0.5:
                    return 0.4
                elif overlap_ratio > 0.3:
                    return 0.2

    return 0.0


def compute_reward(trajectory: Any) -> dict[str, Any]:
    answer_turns = trajectory.agent_trajectories.get("answerer", [])
    predicted = ""
    if answer_turns:
        last_response = getattr(answer_turns[-1], "response_text", "")
        predicted = _extract_tag(last_response, "answer") or last_response

    metadata = getattr(trajectory, "metadata", {}) or {}
    expected_candidates = _expected_candidates(metadata if isinstance(metadata, dict) else {})
    final_reward = _compute_answer_score(predicted, expected_candidates)

    return {
        "agent_rewards": {
            role: final_reward for role in trajectory.agent_trajectories
        },
        "final_reward": final_reward,
    }
