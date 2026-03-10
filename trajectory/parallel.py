from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from .archiver import LogArchiver
from .backend import InferenceBackend
from .datatypes import EpisodeResult
from .pipe import AgentPipe, AgentPipeConfig
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def parallel_rollout(
    prompts: list[str],
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    n_samples_per_prompt: int = 1,
    max_concurrent: int | None = None,
    archiver: LogArchiver | None = None,
) -> list[EpisodeResult]:
    """
    对每个 prompt 并行采样 n_samples_per_prompt 条 episode。
    max_concurrent 限制同时运行的 AgentPipe 数量（None = 不限制）。

    Args:
        prompts: List of prompts to process
        reward_provider: Reward provider for computing rewards
        config: AgentPipe configuration
        backend: Inference backend for model calls
        n_samples_per_prompt: Number of episodes to sample per prompt
        max_concurrent: Maximum concurrent episodes (None = unlimited)
        archiver: Optional log archiver for failure tracking

    Returns:
        List of successful episode results
    """
    if n_samples_per_prompt < 1:
        raise ValueError("n_samples_per_prompt must be >= 1")
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1 when provided")
    if not prompts:
        return []

    _LOGGER.info(f"[parallel_rollout] Starting with {len(prompts)} prompts, {n_samples_per_prompt} samples each, max_concurrent={max_concurrent}")

    # Create shared archiver if not provided and archiving is enabled
    if archiver is None and config.archive_logs:
        archiver = LogArchiver(archive_root=config.archive_root)

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None

    # Track failures for summary
    failures: list[dict] = []

    async def run_one(prompt: str, idx: int) -> EpisodeResult:
        pipe = AgentPipe(config=config, backend=backend, archiver=archiver)
        _LOGGER.debug(f"[parallel_rollout] Starting task {idx}")
        try:
            if semaphore is None:
                result = await pipe.run(prompt=prompt, reward_provider=reward_provider)
            else:
                async with semaphore:
                    result = await pipe.run(prompt=prompt, reward_provider=reward_provider)
            _LOGGER.debug(f"[parallel_rollout] Task {idx} completed")
            return result
        except Exception as e:
            _LOGGER.error(f"[parallel_rollout] Task {idx} failed with error: {type(e).__name__}: {e}")
            failures.append({
                "task_index": idx,
                "prompt": prompt[:200] if prompt else None,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            raise

    tasks = [
        run_one(prompt, idx)
        for idx, prompt in enumerate(prompts)
        for _ in range(n_samples_per_prompt)
    ]
    _LOGGER.info(f"[parallel_rollout] Created {len(tasks)} tasks")
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[EpisodeResult] = []
    failed_count = 0
    for idx, item in enumerate(gathered):
        if isinstance(item, Exception):
            _LOGGER.error(f"[parallel_rollout] Episode {idx} failed: {type(item).__name__}: {item}")
            failed_count += 1
            continue
        if isinstance(item, BaseException):
            _LOGGER.error(f"[parallel_rollout] Episode {idx} raised BaseException: {type(item).__name__}: {item}")
            raise item
        results.append(item)

    _LOGGER.info(f"[parallel_rollout] Completed: {len(results)} successes, {failed_count} failures")

    # Archive failure summary if there were failures
    if failures and archiver is not None:
        summary = {
            "session_id": archiver.session_dir.name,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "successful": len(results),
            "failed": failed_count,
            "failures": failures,
        }
        try:
            archiver.archive_failure_summary(summary)
            _LOGGER.info(f"[parallel_rollout] Failure summary archived to {archiver.session_dir}")
        except Exception as e:
            _LOGGER.error(f"[parallel_rollout] Failed to archive failure summary: {e}")

    return results
