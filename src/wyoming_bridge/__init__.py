# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wyoming protocol bridge to a Riva gRPC speech server (stock NeMo-Speech.cpp).

Home Assistant talks Wyoming for speech-to-text and text-to-speech. NeMo-Speech.cpp
only serves Riva gRPC, so this standalone server translates between the two:

    python -m wyoming_bridge --riva-server nemo-speech:50051 --uri tcp://0.0.0.0:10700
"""
