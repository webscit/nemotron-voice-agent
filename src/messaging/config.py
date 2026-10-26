# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven configuration for Matrix messaging.

Environment variables:
  - MATRIX_HOMESERVER      Homeserver base URL, e.g. https://matrix-client.matrix.org (required)
  - MATRIX_USER_ID         The agent's own Matrix account, e.g. @atlas-bot:matrix.org (required)
  - MATRIX_ACCESS_TOKEN    Access token of that account (required)
  - MATRIX_DEVICE_ID       Device id the token was issued for (optional)
  - MATRIX_DATA_DIR        Inbox database and sync state (default: <project_root>/data/matrix)
  - MATRIX_CONTACTS_FILE   Address book (default: <MATRIX_DATA_DIR>/contacts.yaml)

Messaging is enabled when the three required variables are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from utils import PROJECT_ROOT


@dataclass(frozen=True)
class MatrixConfig:
    """Resolved Matrix settings."""

    homeserver: str
    user_id: str
    access_token: str
    device_id: str
    data_dir: Path
    contacts_file: Path

    @property
    def enabled(self) -> bool:
        """Whether the agent has a Matrix account to use."""
        return bool(self.homeserver and self.user_id and self.access_token)

    @property
    def db_path(self) -> Path:
        """SQLite file of the inbox and the sync state."""
        return self.data_dir / "inbox.db"


def _path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_matrix_config() -> MatrixConfig:
    """Read the Matrix configuration from the environment."""
    data_dir = _path(os.getenv("MATRIX_DATA_DIR", "").strip(), Path("data") / "matrix")
    return MatrixConfig(
        homeserver=os.getenv("MATRIX_HOMESERVER", "").strip().rstrip("/"),
        user_id=os.getenv("MATRIX_USER_ID", "").strip(),
        access_token=os.getenv("MATRIX_ACCESS_TOKEN", "").strip(),
        device_id=os.getenv("MATRIX_DEVICE_ID", "").strip(),
        data_dir=data_dir,
        contacts_file=_path(os.getenv("MATRIX_CONTACTS_FILE", "").strip(), data_dir / "contacts.yaml"),
    )
