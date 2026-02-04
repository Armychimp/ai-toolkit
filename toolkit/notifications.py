"""
ntfy push notification support for AI Toolkit training events.

Provides non-blocking notifications for training start, samples, saves, completion, and errors.
Uses stdlib urllib only - no additional dependencies required.
"""

import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import traceback


@dataclass
class NotificationConfig:
    """Configuration for ntfy notifications."""
    enabled: bool = False
    topic: str = ""
    server_url: str = "https://ntfy.sh"

    # Event toggles
    on_training_start: bool = True
    on_sample_generated: bool = False  # Off by default (can be noisy)
    on_checkpoint_saved: bool = True
    on_training_complete: bool = True
    on_error: bool = True

    # Optional settings
    priority: str = "default"  # min/low/default/high/urgent
    default_tags: List[str] = field(default_factory=lambda: ["ai-toolkit"])

    # Auth for private servers (optional)
    access_token: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> "NotificationConfig":
        """Create config from dictionary, handling None and missing keys."""
        if config_dict is None:
            return cls()

        return cls(
            enabled=config_dict.get("enabled", False),
            topic=config_dict.get("topic", ""),
            server_url=config_dict.get("server_url", "https://ntfy.sh"),
            on_training_start=config_dict.get("on_training_start", True),
            on_sample_generated=config_dict.get("on_sample_generated", False),
            on_checkpoint_saved=config_dict.get("on_checkpoint_saved", True),
            on_training_complete=config_dict.get("on_training_complete", True),
            on_error=config_dict.get("on_error", True),
            priority=config_dict.get("priority", "default"),
            default_tags=config_dict.get("default_tags", ["ai-toolkit"]),
            access_token=config_dict.get("access_token"),
        )

    def is_valid(self) -> bool:
        """Check if config is valid for sending notifications."""
        return self.enabled and bool(self.topic)


class NtfyNotifier:
    """Sends push notifications via ntfy.sh or compatible server."""

    def __init__(self, config: NotificationConfig, job_name: str):
        self.config = config
        self.job_name = job_name
        self._executor: Optional[ThreadPoolExecutor] = None

        if config.is_valid():
            # Single worker thread for non-blocking sends
            self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_url(self) -> str:
        """Build the ntfy endpoint URL."""
        base = self.config.server_url.rstrip("/")
        return f"{base}/{self.config.topic}"

    def _send(
        self,
        title: str,
        message: str,
        tags: Optional[List[str]] = None,
        priority: Optional[str] = None,
        blocking: bool = False,
    ) -> None:
        """
        Send a notification to ntfy.

        Args:
            title: Notification title
            message: Notification body
            tags: List of emoji tags (e.g., ["rocket", "tada"])
            priority: Override default priority
            blocking: If True, wait for send to complete (used for errors)
        """
        if not self.config.is_valid():
            return

        if self._executor is None:
            return

        all_tags = list(self.config.default_tags)
        if tags:
            all_tags = tags + all_tags

        def do_send():
            try:
                headers = {
                    "Title": title,
                    "Priority": priority or self.config.priority,
                    "Tags": ",".join(all_tags),
                }

                if self.config.access_token:
                    headers["Authorization"] = f"Bearer {self.config.access_token}"

                data = message.encode("utf-8")
                req = urllib.request.Request(
                    self._get_url(),
                    data=data,
                    headers=headers,
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=10) as response:
                    response.read()

            except urllib.error.URLError as e:
                print(f"[Notifications] Failed to send: {e}")
            except Exception as e:
                print(f"[Notifications] Error: {e}")

        if blocking:
            do_send()
        else:
            self._executor.submit(do_send)

    def notify_training_start(self, total_steps: int, model_name: str) -> None:
        """Send notification when training starts."""
        if not self.config.on_training_start:
            return

        self._send(
            title=f"Training Started: {self.job_name}",
            message=f"Model: {model_name}\nTotal steps: {total_steps}",
            tags=["rocket"],
            priority="default",
        )

    def notify_sample_generated(self, step: int, total_steps: int) -> None:
        """Send notification when a sample is generated."""
        if not self.config.on_sample_generated:
            return

        progress = (step / total_steps * 100) if total_steps > 0 else 0

        self._send(
            title=f"Sample Generated: {self.job_name}",
            message=f"Step {step}/{total_steps} ({progress:.1f}%)",
            tags=["art"],
            priority="low",
        )

    def notify_checkpoint_saved(self, step: int, save_path: str) -> None:
        """Send notification when a checkpoint is saved."""
        if not self.config.on_checkpoint_saved:
            return

        # Extract just the filename for cleaner notification
        import os
        filename = os.path.basename(save_path)

        self._send(
            title=f"Checkpoint Saved: {self.job_name}",
            message=f"Step {step}\nFile: {filename}",
            tags=["floppy_disk"],
            priority="default",
        )

    def notify_training_complete(self, total_steps: int, save_root: str) -> None:
        """Send notification when training completes."""
        if not self.config.on_training_complete:
            return

        self._send(
            title=f"Training Complete: {self.job_name}",
            message=f"Finished {total_steps} steps\nOutput: {save_root}",
            tags=["tada", "white_check_mark"],
            priority="high",
        )

    def notify_error(self, error: Exception, step: Optional[int] = None) -> None:
        """
        Send notification when an error occurs.

        This method blocks to ensure the user is notified immediately.
        """
        if not self.config.on_error:
            return

        step_info = f" at step {step}" if step is not None else ""
        error_msg = str(error)

        # Truncate very long error messages
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        self._send(
            title=f"Training Error: {self.job_name}",
            message=f"Error{step_info}:\n{error_msg}",
            tags=["warning", "x"],
            priority="urgent",
            blocking=True,  # Block to ensure error notification is sent
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the notification executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None


class NoOpNotifier:
    """A no-op notifier that does nothing. Used when notifications are disabled."""

    def __init__(self):
        pass

    def notify_training_start(self, total_steps: int, model_name: str) -> None:
        pass

    def notify_sample_generated(self, step: int, total_steps: int) -> None:
        pass

    def notify_checkpoint_saved(self, step: int, save_path: str) -> None:
        pass

    def notify_training_complete(self, total_steps: int, save_root: str) -> None:
        pass

    def notify_error(self, error: Exception, step: Optional[int] = None) -> None:
        pass

    def shutdown(self, wait: bool = True) -> None:
        pass


def create_notifier(config_dict: Optional[Dict[str, Any]], job_name: str) -> NtfyNotifier:
    """
    Factory function to create a notifier from config dictionary.

    Returns a NoOpNotifier if notifications are disabled or config is invalid.

    Args:
        config_dict: Raw notification config from YAML/JSON
        job_name: Name of the training job

    Returns:
        NtfyNotifier or NoOpNotifier instance
    """
    config = NotificationConfig.from_dict(config_dict)

    if not config.is_valid():
        return NoOpNotifier()

    return NtfyNotifier(config, job_name)
