from __future__ import annotations

import copy
import logging
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)


class MASLauncher:
    def __init__(
        self,
        work_dir: str | Path | None = None,
        capture_output: bool = True,
    ) -> None:
        self._work_dir = Path(work_dir).resolve() if work_dir is not None else None
        self._temp_files: list[Path] = []
        self._capture_output = capture_output
        self._captured_stdout: bytes | None = None
        self._captured_stderr: bytes | None = None

    def prepare_config(
        self,
        config_template: dict[str, Any],
        monitor_url: str,
        agent_roles: list[str],
    ) -> Path:
        config = copy.deepcopy(config_template)

        # Clear top-level llm.model to avoid conflicts with agent-level model settings
        llm_cfg = config.setdefault("llm", {})
        if isinstance(llm_cfg, dict):
            llm_cfg["base_url"] = monitor_url
            # Remove model from top-level to prevent agent inheritance conflicts
            llm_cfg.pop("model", None)

        agents_cfg = config.setdefault("agents", {})
        if not isinstance(agents_cfg, dict):
            agents_cfg = {}
            config["agents"] = agents_cfg

        for role in agent_roles:
            role_cfg = agents_cfg.setdefault(role, {})
            if not isinstance(role_cfg, dict):
                role_cfg = {}
                agents_cfg[role] = role_cfg
            # Set the agent's model name to the role name
            role_cfg["model"] = role

            # Ensure each agent has its own llm config pointing to the monitor
            role_llm_cfg = role_cfg.get("llm")
            if role_llm_cfg is None:
                # Create llm config for this agent if it doesn't exist
                role_llm_cfg = {}
                role_cfg["llm"] = role_llm_cfg
            if not isinstance(role_llm_cfg, dict):
                role_llm_cfg = {}
                role_cfg["llm"] = role_llm_cfg

            # Set the base_url to the monitor for this agent
            role_llm_cfg["base_url"] = monitor_url
            # Ensure no conflicting model setting in agent's llm config
            role_llm_cfg.pop("model", None)

        _LOGGER.info(f"[MAS Launcher] Creating temp file in directory: {self._work_dir}")
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="trajectory_mas_",
            dir=str(self._work_dir) if self._work_dir else None,
            delete=False,
            encoding="utf-8",
        )
        config_path = Path(temp_file.name)
        _LOGGER.info(f"[MAS Launcher] Created temp file: {config_path}")
        try:
            yaml.safe_dump(config, temp_file, sort_keys=False)
        except Exception:
            temp_file.close()
            try:
                config_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            temp_file.close()

        self._temp_files.append(config_path)
        return config_path

    def launch(
        self,
        command: str,
        env_vars: dict[str, str] | None = None,
        capture_output: bool | None = None,
    ) -> subprocess.Popen[str]:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        _LOGGER.info(f"[MAS Launcher] Working directory: {self._work_dir}")
        _LOGGER.info(f"[MAS Launcher] Command: {command}")

        # Reset captured output
        self._captured_stdout = None
        self._captured_stderr = None

        # Determine if we should capture output
        should_capture = capture_output if capture_output is not None else self._capture_output
        if should_capture or _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("[MAS Launcher] Capturing output for debugging and archival")
            stdout_pipe = subprocess.PIPE
            stderr_pipe = subprocess.PIPE
        else:
            stdout_pipe = subprocess.DEVNULL
            stderr_pipe = subprocess.DEVNULL

        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self._work_dir) if self._work_dir else None,
            env=env,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
            start_new_session=True,
        )

        _LOGGER.info(f"[MAS Launcher] Process started with PID: {process.pid}")
        return process

    def wait(
        self,
        process: subprocess.Popen[Any],
        timeout: float | None = None,
    ) -> int:
        try:
            _LOGGER.debug(f"[MAS Launcher] Waiting for process {process.pid} (timeout: {timeout})")

            # Use communicate() to wait and capture output
            # communicate() waits for the process to terminate and reads stdout/stderr
            if process.stdout is not None or process.stderr is not None:
                try:
                    self._captured_stdout, self._captured_stderr = process.communicate(timeout=timeout)
                    exit_code = process.returncode

                    if self._captured_stdout:
                        stdout_str = self._captured_stdout.decode('utf-8', errors='replace')
                        _LOGGER.info(f"[MAS Launcher] stdout:\n{stdout_str}")

                    if self._captured_stderr:
                        stderr_str = self._captured_stderr.decode('utf-8', errors='replace')
                        _LOGGER.error(f"[MAS Launcher] stderr:\n{stderr_str}")
                except subprocess.TimeoutExpired:
                    _LOGGER.warning(f"[MAS Launcher] Process {process.pid} timed out after {timeout}s, killing...")
                    self._kill_process_tree(process)
                    # Try to get any remaining output after killing
                    try:
                        self._captured_stdout, self._captured_stderr = process.communicate(timeout=1)
                    except Exception:
                        pass
                    exit_code = process.returncode or -1
            else:
                # No output capture, just wait
                exit_code = process.wait(timeout=timeout)

            _LOGGER.info(f"[MAS Launcher] Process {process.pid} exited with code: {exit_code}")
            return exit_code
        except subprocess.TimeoutExpired:
            _LOGGER.warning(f"[MAS Launcher] Process {process.pid} timed out after {timeout}s, killing...")
            self._kill_process_tree(process)
            return process.wait()

    def get_captured_output(self) -> tuple[bytes | None, bytes | None]:
        """
        Get the captured stdout and stderr from the last process.

        Returns:
            A tuple of (stdout, stderr) as bytes. May be None if output was not captured.
        """
        return self._captured_stdout, self._captured_stderr

    def cleanup(self) -> None:
        remaining: list[Path] = []
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                remaining.append(path)
        self._temp_files = remaining

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return

        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        except OSError:
            pass

        try:
            process.kill()
        except OSError:
            pass
