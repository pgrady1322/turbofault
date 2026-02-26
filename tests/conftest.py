"""
TurboFault v0.1.0

conftest.py — Shared pytest configuration and fixtures.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import os

# Prevent PyTorch multi-threading segfaults under pytest on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# TurboFault v0.1.0
# Any usage is subject to this software's license.
