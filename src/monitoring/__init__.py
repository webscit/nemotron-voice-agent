# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation recording, metrics persistence and offline post-processing.

The package is split in a *hot path* and a *cold path*:

- Hot path (``recorder``, ``writer``): runs inside a live pipecat session. It only
  enqueues plain dicts; a single background writer batches them to the session
  store and the artifact store off the event loop.
- Cold path (``jobs``): a separate process ("dreamer") that runs post-processing
  jobs over recorded sessions only while no conversation is ongoing.

Storage is behind two narrow interfaces (``store.SessionStore`` and
``store.ArtifactStore``) so a remote database or object storage can replace the
local SQLite + filesystem defaults.
"""
