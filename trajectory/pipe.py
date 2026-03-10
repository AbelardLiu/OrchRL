from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archiver import LogArchiver
from .backend import InferenceBackend
from .collector import TrajectoryCollector
from .datatypes import EpisodeResult, ModelMappingEntry
from .launcher import MASLauncher
from .monitor import ModelMonitor
from .reward import RewardProvider, RewardWorker

_LOGGER = logging.getLogger(__name__)


@dataclass
class AgentPipeConfig:
    mas_command_template: str
    config_template: dict[str, Any]
    model_mapping: dict[str, ModelMappingEntry]
    timeout: float = 300.0
    monitor_host: str = "127.0.0.1"
    monitor_port: int = 0
    mas_work_dir: str | Path | None = None
    archive_logs: bool = True
    archive_root: str | Path | None = None


class AgentPipe:
    def __init__(
        self,
        config: AgentPipeConfig,
        backend: InferenceBackend,
        archiver: LogArchiver | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._collector = TrajectoryCollector()
        self._reward_worker = RewardWorker()
        self._archiver = archiver
        if config.archive_logs and archiver is None:
            self._archiver = LogArchiver(archive_root=config.archive_root)

    async def run(
        self,
        prompt: str,
        reward_provider: RewardProvider,
    ) -> EpisodeResult:
        episode_id = uuid.uuid4().hex
        _LOGGER.info(f"[AgentPipe] Starting episode {episode_id}")
        _LOGGER.debug(f"[AgentPipe] Prompt: {prompt[:100]}...")

        monitor = ModelMonitor(
            backend=self._backend,
            model_mapping=self._config.model_mapping,
            episode_id=episode_id,
        )
        launcher = MASLauncher(work_dir=self._config.mas_work_dir, capture_output=True)
        primary_error: BaseException | None = None
        config_path: Path | None = None
        command: str | None = None
        exit_code: int | None = None

        try:
            _LOGGER.info(f"[AgentPipe] Starting monitor on {self._config.monitor_host}:{self._config.monitor_port}")
            port = await monitor.start(
                host=self._config.monitor_host,
                port=self._config.monitor_port,
            )
            monitor_url = f"http://{self._config.monitor_host}:{port}/v1"
            _LOGGER.info(f"[AgentPipe] Monitor started at {monitor_url}")

            _LOGGER.debug(f"[AgentPipe] Preparing config with {len(self._config.model_mapping)} agents")
            config_path = await asyncio.to_thread(
                launcher.prepare_config,
                config_template=self._config.config_template,
                monitor_url=monitor_url,
                agent_roles=list(self._config.model_mapping.keys()),
            )
            _LOGGER.info(f"[AgentPipe] Config prepared: {config_path}")

            command = self._config.mas_command_template.format(
                config_path=shlex.quote(str(config_path)),
                prompt=shlex.quote(prompt),
            )
            _LOGGER.info(f"[AgentPipe] MAS command: {command}")

            # Pass SEARCH_MAS environment variables to subprocess
            # These may not be available in Ray worker environments
            mas_env_vars = {}
            for key in os.environ:
                if key.startswith('SEARCH_MAS_') or key.startswith('OPENAI_'):
                    mas_env_vars[key] = os.environ[key]

            if mas_env_vars:
                _LOGGER.debug(f"[AgentPipe] Passing {len(mas_env_vars)} MAS env vars to subprocess")

            process = await asyncio.to_thread(launcher.launch, command=command, env_vars=mas_env_vars if mas_env_vars else None)
            _LOGGER.info(f"[AgentPipe] MAS process started (PID: {process.pid}), waiting (timeout: {self._config.timeout}s)")

            exit_code = await asyncio.to_thread(
                launcher.wait,
                process,
                self._config.timeout,
            )
            _LOGGER.info(f"[AgentPipe] MAS process exited with code: {exit_code}")

            if exit_code != 0:
                error_msg = f"MAS process exited with non-zero exit code {exit_code}. " \
                           f"Check logs above for stderr output. Command: {command}"
                _LOGGER.error(f"[AgentPipe] {error_msg}")
                raise RuntimeError(error_msg)

            trajectory = self._collector.build(buffer=monitor.get_buffer(), episode_id=episode_id)
            _LOGGER.info(f"[AgentPipe] Trajectory built with {len(trajectory.agent_trajectories)} agents")

            result = await asyncio.to_thread(
                self._reward_worker.compute,
                trajectory,
                reward_provider,
            )
            result.metadata["exit_code"] = exit_code
            _LOGGER.info(f"[AgentPipe] Episode {episode_id} completed successfully")
            return result
        except BaseException as exc:
            primary_error = exc
            _LOGGER.error(f"[AgentPipe] Episode {episode_id} failed: {type(exc).__name__}: {exc}")
            self._archive_failure(
                episode_id=episode_id,
                launcher=launcher,
                monitor=monitor,
                prompt=prompt,
                command=command,
                exit_code=exit_code,
                exception=exc,
                config_path=config_path,
            )
            raise
        finally:
            stop_error: Exception | None = None
            cleanup_error: Exception | None = None

            try:
                await monitor.stop()
                _LOGGER.debug(f"[AgentPipe] Monitor stopped for episode {episode_id}")
            except Exception as exc:
                _LOGGER.warning(f"[AgentPipe] Error stopping monitor: {exc}")
                stop_error = exc

            # Archive config file before cleanup (for both success and failure cases)
            # Note: failures already archived in the except block, so check if we need to archive
            if self._archiver is not None and config_path is not None and config_path.exists():
                if primary_error is None:  # Success case - archive wasn't called yet
                    try:
                        self._archiver.copy_file_to_archive(
                            episode_id=episode_id,
                            source_path=config_path,
                            dest_name="mas_config.yaml",
                        )
                        _LOGGER.debug(f"[AgentPipe] Archived config for successful episode {episode_id}")
                    except Exception as e:
                        _LOGGER.warning(f"[AgentPipe] Failed to archive config file: {e}")

            try:
                launcher.cleanup()
                _LOGGER.debug(f"[AgentPipe] Launcher cleanup completed for episode {episode_id}")
            except Exception as exc:
                _LOGGER.warning(f"[AgentPipe] Error in launcher cleanup: {exc}")
                cleanup_error = exc

            if primary_error is None:
                if stop_error is not None:
                    if cleanup_error is not None:
                        stop_error.add_note(f"launcher.cleanup() also failed: {cleanup_error}")
                    raise stop_error
                if cleanup_error is not None:
                    raise cleanup_error

    def _archive_failure(
        self,
        episode_id: str,
        launcher: MASLauncher,
        monitor: ModelMonitor,
        prompt: str,
        command: str | None,
        exit_code: int | None,
        exception: BaseException,
        config_path: Path | None,
    ) -> None:
        """Archive logs and context when an episode fails."""
        if self._archiver is None:
            return

        try:
            # Get captured process output
            stdout, stderr = launcher.get_captured_output()

            # Prepare metadata
            metadata = {
                "prompt": prompt,
                "command": command,
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "config_path": str(config_path) if config_path else None,
            }

            # Archive process output
            self._archiver.archive_process_output(
                episode_id=episode_id,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                metadata=metadata,
            )

            # Archive monitor buffer (HTTP interactions)
            monitor_buffer = monitor.get_buffer()
            if monitor_buffer:
                self._archiver.archive_monitor_buffer(
                    episode_id=episode_id,
                    buffer=monitor_buffer,
                )

            # Archive config file if it exists
            if config_path and config_path.exists():
                try:
                    self._archiver.copy_file_to_archive(
                        episode_id=episode_id,
                        source_path=config_path,
                        dest_name="mas_config.yaml",
                    )
                except Exception as e:
                    _LOGGER.warning(f"[AgentPipe] Failed to archive config file: {e}")

            _LOGGER.info(f"[AgentPipe] Archived failure logs for episode {episode_id} to {self._archiver.session_dir}")
        except Exception as archive_exc:
            _LOGGER.error(f"[AgentPipe] Failed to archive failure logs: {type(archive_exc).__name__}: {archive_exc}")
