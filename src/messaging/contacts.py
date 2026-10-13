# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's Matrix address book (``MATRIX_CONTACTS_FILE``).

```yaml
contacts:
  - name: Bob
    matrix_id: "@bob:matrix.org"
    aliases: [Robert]       # other names people use for Bob (optional)
    owner: Alice            # only Alice may message Bob and hear his replies (optional)
```

A contact without ``owner`` is shared by everybody who talks with the agent.
``owner`` is a person's name or id from the People page. The file is reloaded
when it changes, so contacts can be edited while the server runs.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from utils import load_yaml_file

_MATRIX_ID_RE = re.compile(r"^@[^:\s]+:\S+$")
# Similarity a spoken name needs to match a contact it does not spell exactly (ASR spelling: "Jon" / "John").
FUZZY_MATCH_RATIO = 0.8


@dataclass(frozen=True)
class Contact:
    """One address-book entry."""

    name: str
    matrix_id: str
    aliases: tuple[str, ...] = ()
    owner: str | None = None

    def names(self) -> tuple[str, ...]:
        """The contact's name and aliases."""
        return (self.name, *self.aliases)

    def allowed_for(self, person: dict[str, Any] | None) -> bool:
        """Whether ``person`` (a people-store row, or None when unknown) may use this contact."""
        if not self.owner:
            return True
        if person is None:
            return False
        owner = _fold(self.owner)
        return owner in (_fold(str(person.get("id") or "")), _fold(str(person.get("name") or "")))


def _fold(text: str) -> str:
    """Case- and accent-insensitive form of a name ("Élodie" == "elodie")."""
    decomposed = unicodedata.normalize("NFKD", " ".join(text.split()).casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def parse_contacts(data: dict[str, Any]) -> list[Contact]:
    """Contacts of a parsed contacts file; invalid entries are skipped with a warning."""
    raw = data.get("contacts")
    if not isinstance(raw, list):
        return []
    contacts: list[Contact] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = " ".join(str(entry.get("name") or "").split())
        matrix_id = str(entry.get("matrix_id") or "").strip()
        if not name or not _MATRIX_ID_RE.match(matrix_id):
            logger.warning(f"Matrix contacts: skipping an entry without a name or a valid matrix_id: {entry!r}")
            continue
        if matrix_id in seen:
            logger.warning(f"Matrix contacts: skipping a duplicate entry for {matrix_id}")
            continue
        seen.add(matrix_id)
        aliases = entry.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [aliases]
        owner = str(entry.get("owner") or "").strip() or None
        contacts.append(
            Contact(
                name=name,
                matrix_id=matrix_id,
                aliases=tuple(a for a in (" ".join(str(x).split()) for x in aliases) if a),
                owner=owner,
            )
        )
    return contacts


def match_contacts(query: str, contacts: list[Contact]) -> list[Contact]:
    """Contacts named ``query``: exact (name, alias or Matrix id) matches, else close spellings."""
    wanted = _fold(query)
    if not wanted:
        return []
    exact = [c for c in contacts if wanted == _fold(c.matrix_id) or any(wanted == _fold(n) for n in c.names())]
    if exact:
        return exact
    # "Bob Smith" for "Bob", or "Bob" for "Bob Smith".
    words = set(wanted.split())
    partial = [c for c in contacts if any(words & set(_fold(n).split()) for n in c.names())]
    if partial:
        return partial
    scored = []
    for contact in contacts:
        ratio = max(difflib.SequenceMatcher(None, wanted, _fold(n)).ratio() for n in contact.names())
        if ratio >= FUZZY_MATCH_RATIO:
            scored.append((ratio, contact))
    return [c for _, c in sorted(scored, key=lambda item: -item[0])]


class ContactBook:
    """The contacts file, reloaded when it changes on disk."""

    def __init__(self, path: Path):
        """Read contacts from ``path`` (which may not exist yet)."""
        self._path = path
        self._mtime: float | None = None
        self._contacts: list[Contact] = []

    def all(self) -> list[Contact]:
        """Every contact, in file order."""
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self._mtime:
            self._mtime = mtime
            self._contacts = parse_contacts(load_yaml_file(self._path)) if mtime is not None else []
            logger.info(f"Matrix contacts: {len(self._contacts)} loaded from {self._path}")
        return list(self._contacts)

    def by_matrix_id(self, matrix_id: str) -> Contact | None:
        """The contact with this Matrix id, or None."""
        return next((c for c in self.all() if c.matrix_id == matrix_id), None)
