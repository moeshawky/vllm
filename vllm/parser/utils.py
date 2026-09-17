# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions for counting tool calls across chat history,
response items, and requests.
"""

from collections.abc import Iterable, Sequence

from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
)


def count_tool_calls(tool_calls: object) -> int:
    """Count tool calls from an object that may be None, a string,
    bytes, dict, or iterable. Returns 1 for non-iterable
    non-None values.
    """
    if tool_calls is None:
        return 0
    if isinstance(tool_calls, (str, bytes, dict)):
        # These represent a single tool call object serialized
        # as a string or dict, not an iterable of calls.
        return 1
    if isinstance(tool_calls, Iterable):
        return sum(1 for _ in tool_calls)
    return 1


def count_chat_history_tool_calls(
    messages: Sequence[ChatCompletionMessageParam],
) -> int:
    """Count tool calls in chat history messages by filtering
    for assistant role messages and summing their tool_calls.
    """
    # Use isinstance(msg, dict) rather than TypedDict
    # because the function operates on runtime dict objects.
    return sum(
        count_tool_calls(msg.get("tool_calls"))
        for msg in messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    )


def count_response_history_tool_calls(
    response_items: Sequence[ResponseInputOutputItem],
) -> int:
    """Count tool calls in response items, handling
    ResponseFunctionToolCall, dict items with type 'function_call',
    and assistant-role dict items.
    """
    count = 0
    for item in response_items:
        if isinstance(item, ResponseFunctionToolCall):
            count += 1
            continue

        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "function_call":
                count += 1
            elif item.get("role") == "assistant":
                count += count_tool_calls(item.get("tool_calls"))

    return count


def count_history_tool_calls(
    request: ChatCompletionRequest | ResponsesRequest,
) -> int:
    """Count total tool calls in a request by dispatching to chat
    or response history counting based on request type.
    """
    if isinstance(request, ChatCompletionRequest):
        return count_chat_history_tool_calls(request.messages)

    request_input = request.input
    if isinstance(request_input, str):
        return 0

    return count_response_history_tool_calls(request_input)
