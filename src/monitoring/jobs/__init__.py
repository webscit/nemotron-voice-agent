# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline post-processing jobs over recorded sessions.

Importing this package registers the built-in jobs. Add a job by subclassing
``base.Job``, decorating it with ``@base.register`` and importing it here.
"""

from monitoring.jobs import reasr  # noqa: F401  (registers ``reasr``)
from monitoring.jobs.base import JOB_REGISTRY, Job, JobContext, Preempted, register

__all__ = ["JOB_REGISTRY", "Job", "JobContext", "Preempted", "register"]
