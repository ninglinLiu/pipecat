#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Anthropic tool_choice mapping from LLMContext."""

import pytest
from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN

from pipecat.adapters.llm.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    anthropic_from_llm_context_tool_choice,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.types import NOT_GIVEN


def test_anthropic_from_llm_context_tool_choice_mapping():
    assert anthropic_from_llm_context_tool_choice(NOT_GIVEN) is ANTHROPIC_NOT_GIVEN
    assert anthropic_from_llm_context_tool_choice("none") == {"type": "none"}
    assert anthropic_from_llm_context_tool_choice("auto") == {"type": "auto"}
    assert anthropic_from_llm_context_tool_choice("required") == {"type": "any"}
    assert anthropic_from_llm_context_tool_choice(
        {"type": "function", "function": {"name": "lookup"}}
    ) == {"type": "tool", "name": "lookup"}
    assert anthropic_from_llm_context_tool_choice({"type": "tool", "name": "lookup"}) == {
        "type": "tool",
        "name": "lookup",
    }


@pytest.mark.asyncio
async def test_get_llm_invocation_params_includes_tool_choice():
    adapter = AnthropicLLMAdapter()
    context = LLMContext(messages=[{"role": "user", "content": "hi"}])
    context.set_tool_choice("none")
    params = await adapter.get_llm_invocation_params(
        context, enable_prompt_caching=False, system_instruction=None
    )
    assert params["tool_choice"] == {"type": "none"}

    context.set_tool_choice(NOT_GIVEN)
    params = await adapter.get_llm_invocation_params(
        context, enable_prompt_caching=False, system_instruction=None
    )
    assert params["tool_choice"] is ANTHROPIC_NOT_GIVEN
