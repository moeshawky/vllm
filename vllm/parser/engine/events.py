# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Semantic event types emitted by the streaming parser engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    """Enum of semantic event types emitted by the streaming parser engine
    during token processing.
    """

    TEXT_CHUNK = auto()
    REASONING_START = auto()
    REASONING_CHUNK = auto()
    REASONING_END = auto()
    TOOL_CALL_START = auto()
    TOOL_NAME = auto()
    ARG_VALUE_CHUNK = auto()
    TOOL_CALL_END = auto()


@dataclass(slots=True)
class SemanticEvent:
    """Represents a single semantic event emitted by the parser engine.

    Contains event type, string value, tool index, and token count.
    """

    type: EventType
    # value is the text content of the event (e.g. a chunk of text or reasoning).
    # tool_index identifies which tool call slot this event belongs to. token_count
    # tracks how many tokens this event spans.
    value: str = ""
    tool_index: int = -1
    token_count: int = 0
