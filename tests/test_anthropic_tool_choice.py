#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Anthropic tool_choice mapping from LLMContext."""

import unittest

from anthropic import NOT_GIVEN as ANTHROPIC_NOT_GIVEN

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.adapters.llm.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    anthropic_from_llm_context_tool_choice,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.types import NOT_GIVEN


class TestAnthropicToolChoiceMapping(unittest.TestCase):
    def test_from_llm_context_tool_choice_literals(self):
        self.assertEqual(anthropic_from_llm_context_tool_choice("none"), {"type": "none"})
        self.assertEqual(anthropic_from_llm_context_tool_choice("auto"), {"type": "auto"})
        self.assertEqual(anthropic_from_llm_context_tool_choice("required"), {"type": "any"})
        self.assertIs(anthropic_from_llm_context_tool_choice(NOT_GIVEN), ANTHROPIC_NOT_GIVEN)

    def test_from_llm_context_tool_choice_named_function(self):
        self.assertEqual(
            anthropic_from_llm_context_tool_choice(
                {"type": "function", "function": {"name": "save_note"}}
            ),
            {"type": "tool", "name": "save_note"},
        )

    async def _params_with_choice(self, tool_choice):
        tools = ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="save_note",
                    description="Save a note",
                    properties={"text": {"type": "string"}},
                    required=["text"],
                )
            ]
        )
        context = LLMContext(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice=tool_choice,
        )
        adapter = AnthropicLLMAdapter()
        return await adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

    def test_get_llm_invocation_params_includes_mapped_tool_choice(self):
        import asyncio

        params = asyncio.run(self._params_with_choice("none"))
        self.assertEqual(params["tool_choice"], {"type": "none"})

        params = asyncio.run(self._params_with_choice("required"))
        self.assertEqual(params["tool_choice"], {"type": "any"})

        params = asyncio.run(
            self._params_with_choice({"type": "function", "function": {"name": "save_note"}})
        )
        self.assertEqual(params["tool_choice"], {"type": "tool", "name": "save_note"})

        params = asyncio.run(self._params_with_choice(NOT_GIVEN))
        self.assertIs(params["tool_choice"], ANTHROPIC_NOT_GIVEN)


if __name__ == "__main__":
    unittest.main()
