# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D103

"""Matrix address book: parsing, name matching and owners."""

import os

from messaging.contacts import Contact, ContactBook, match_contacts, parse_contacts

BOB = Contact(name="Bob Smith", matrix_id="@bob:matrix.org", aliases=("Robert",))
ELODIE = Contact(name="Élodie", matrix_id="@elodie:matrix.org", owner="Alice")
JOHN = Contact(name="John", matrix_id="@john:example.org")
CONTACTS = [BOB, ELODIE, JOHN]


def test_parse_skips_invalid_and_duplicate_entries():
    contacts = parse_contacts(
        {
            "contacts": [
                {"name": "Bob", "matrix_id": "@bob:matrix.org", "aliases": "Robert", "owner": " Alice "},
                {"name": "No id"},
                {"name": "Bad id", "matrix_id": "bob"},
                {"name": "Bob again", "matrix_id": "@bob:matrix.org"},
                "junk",
            ]
        }
    )
    assert contacts == [Contact(name="Bob", matrix_id="@bob:matrix.org", aliases=("Robert",), owner="Alice")]
    assert parse_contacts({}) == []


def test_match_exact_alias_accent_and_matrix_id():
    assert match_contacts("bob smith", CONTACTS) == [BOB]
    assert match_contacts("ROBERT", CONTACTS) == [BOB]
    assert match_contacts("elodie", CONTACTS) == [ELODIE]
    assert match_contacts("@john:example.org", CONTACTS) == [JOHN]


def test_match_by_word_then_close_spelling():
    assert match_contacts("Bob", CONTACTS) == [BOB]
    assert match_contacts("Jon", CONTACTS) == [JOHN]
    assert match_contacts("Zed", CONTACTS) == []
    assert match_contacts("  ", CONTACTS) == []


def test_owner_restricts_contact_to_that_person():
    assert BOB.allowed_for(None)
    assert not ELODIE.allowed_for(None)
    assert ELODIE.allowed_for({"id": "p1", "name": "alice"})
    assert not ELODIE.allowed_for({"id": "p2", "name": "Carol"})
    by_id = Contact(name="X", matrix_id="@x:y", owner="p1")
    assert by_id.allowed_for({"id": "p1", "name": "Alice"})


def test_contact_book_reloads_on_change(tmp_path):
    path = tmp_path / "contacts.yaml"
    book = ContactBook(path)
    assert book.all() == []
    path.write_text('contacts:\n  - {name: Bob, matrix_id: "@bob:matrix.org"}\n', encoding="utf-8")
    assert [c.name for c in book.all()] == ["Bob"]
    path.write_text('contacts:\n  - {name: Carol, matrix_id: "@carol:matrix.org"}\n', encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))
    assert book.by_matrix_id("@carol:matrix.org").name == "Carol"
    assert book.by_matrix_id("@bob:matrix.org") is None
