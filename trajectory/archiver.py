from __future__ import annotations

import datetime
import gzip
import json
import logging
import shutil
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class LogArchiver:
    """Archive logs and context from failed MAS episodes."""

    def __init__(self, archive_root: str | Path | None = None) -> None:
        """
        Initialize the log archiver.

        Args:
            archive_root: Root directory for archives. Defaults to 'archives/' in current directory.
        """
        if archive_root is None:
            archive_root = Path.cwd() / "archives"
        self._archive_root = Path(archive_root).resolve()
        self._session_dir = self._create_session_dir()

    def _create_session_dir(self) -> Path:
        """Create a timestamped session directory for this run."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        session_dir = self._archive_root / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        _LOGGER.info(f"[LogArchiver] Created archive session directory: {session_dir}")
        return session_dir

    def archive_process_output(
        self,
        episode_id: str,
        stdout: bytes | str | None,
        stderr: bytes | str | None,
        exit_code: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Archive process stdout/stderr with metadata.

        Args:
            episode_id: Unique identifier for the episode
            stdout: Process stdout output
            stderr: Process stderr output
            exit_code: Process exit code
            metadata: Additional metadata to include

        Returns:
            Path to the archive directory
        """
        episode_dir = self._episode_dir(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "episode_id": episode_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "exit_code": exit_code,
            "archived_at": datetime.datetime.now().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        # Write metadata
        meta_path = episode_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Archive stdout
        if stdout:
            stdout_content = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
            if stdout_content.strip():
                stdout_path = episode_dir / "stdout.log"
                self._write_with_compression(stdout_path, stdout_content)

        # Archive stderr
        if stderr:
            stderr_content = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
            if stderr_content.strip():
                stderr_path = episode_dir / "stderr.log"
                self._write_with_compression(stderr_path, stderr_content)

        _LOGGER.info(f"[LogArchiver] Archived process output for episode {episode_id} to {episode_dir}")
        return episode_dir

    def archive_monitor_buffer(
        self,
        episode_id: str,
        buffer: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Archive monitor buffer (HTTP interactions).

        Args:
            episode_id: Unique identifier for the episode
            buffer: Monitor buffer containing request/response data
            metadata: Additional metadata to include

        Returns:
            Path to the archive directory
        """
        episode_dir = self._episode_dir(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Write monitor buffer as JSON
        buffer_path = episode_dir / "monitor_buffer.json"
        with open(buffer_path, "w", encoding="utf-8") as f:
            json.dump(buffer, f, indent=2, ensure_ascii=False)

        # Update or create metadata
        meta_path = episode_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {
                "episode_id": episode_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        meta["monitor_buffer_size"] = len(buffer)
        meta["has_monitor_buffer"] = True
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        _LOGGER.info(f"[LogArchiver] Archived monitor buffer for episode {episode_id} ({len(buffer)} entries)")
        return episode_dir

    def archive_failure_summary(
        self,
        summary: dict[str, Any],
    ) -> Path:
        """
        Archive a failure summary for the session.

        Args:
            summary: Summary of failures (counts, types, etc.)

        Returns:
            Path to the summary file
        """
        summary_path = self._session_dir / "failure_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        _LOGGER.info(f"[LogArchiver] Archived failure summary to {summary_path}")
        return summary_path

    def archive_config(
        self,
        episode_id: str,
        config: dict[str, Any],
        config_name: str = "config.yaml",
    ) -> Path:
        """
        Archive configuration file.

        Args:
            episode_id: Unique identifier for the episode
            config: Configuration dictionary
            config_name: Name for the config file

        Returns:
            Path to the archived config
        """
        import yaml

        episode_dir = self._episode_dir(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        config_path = episode_dir / config_name
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        _LOGGER.debug(f"[LogArchiver] Archived config for episode {episode_id}")
        return config_path

    def copy_file_to_archive(
        self,
        episode_id: str,
        source_path: str | Path,
        dest_name: str | None = None,
    ) -> Path:
        """
        Copy a file to the archive directory.

        Args:
            episode_id: Unique identifier for the episode
            source_path: Path to the source file
            dest_name: Optional destination filename (defaults to source filename)

        Returns:
            Path to the copied file
        """
        source = Path(source_path)
        if not source.exists():
            _LOGGER.warning(f"[LogArchiver] Source file does not exist: {source}")
            raise FileNotFoundError(f"Source file not found: {source}")

        episode_dir = self._episode_dir(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)

        dest_name = dest_name or source.name
        dest_path = episode_dir / dest_name

        shutil.copy2(source, dest_path)
        _LOGGER.debug(f"[LogArchiver] Copied {source} to {dest_path}")
        return dest_path

    def _episode_dir(self, episode_id: str) -> Path:
        """Get the archive directory for a specific episode."""
        return self._session_dir / "episodes" / episode_id

    def _write_with_compression(self, path: Path, content: str) -> None:
        """Write content to file, compressing if large."""
        content_size = len(content.encode("utf-8"))

        # Compress if larger than 100KB
        if content_size > 100 * 1024:
            gzip_path = path.with_suffix(path.suffix + ".gz")
            with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
                f.write(content)
            _LOGGER.debug(f"[LogArchiver] Compressed {path} -> {gzip_path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    @property
    def archive_root(self) -> Path:
        """Get the archive root directory."""
        return self._archive_root

    @property
    def session_dir(self) -> Path:
        """Get the current session directory."""
        return self._session_dir
