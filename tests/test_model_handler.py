# -*- coding: utf-8 -*-

import asyncio
import types as py_types
import unittest

from translation_web_app.model_handler import ModelHandler


class FakeUsageMetadata:
    def __init__(self, prompt_token_count=0, candidates_token_count=0,
                 thoughts_token_count=0, cached_content_token_count=0):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.thoughts_token_count = thoughts_token_count
        self.cached_content_token_count = cached_content_token_count


class FakeGeminiResponse:
    def __init__(self, text="ok", usage_metadata=None):
        self.text = text
        self.usage_metadata = usage_metadata


class FakeCompletionTokensDetails:
    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, completion_tokens_details=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.completion_tokens_details = completion_tokens_details


class FakeChoiceMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeChoiceMessage(content)


class FakeGptResponse:
    def __init__(self, content="ok", usage=None):
        self.choices = [FakeChoice(content)]
        self.usage = usage


def _handler_with_fake_gemini(generate_content_fn):
    handler = ModelHandler(gemini_api_key=None, openai_api_key=None)
    fake_models = py_types.SimpleNamespace(generate_content=generate_content_fn)
    fake_aio = py_types.SimpleNamespace(models=fake_models)
    handler.gemini_client = py_types.SimpleNamespace(aio=fake_aio)
    handler.is_vertex_ai = False
    return handler


def _handler_with_fake_gpt(create_fn):
    handler = ModelHandler(gemini_api_key=None, openai_api_key=None)
    fake_completions = py_types.SimpleNamespace(create=create_fn)
    fake_chat = py_types.SimpleNamespace(completions=fake_completions)
    handler.openai_client = py_types.SimpleNamespace(chat=fake_chat)
    return handler


class ModelHandlerUsageTests(unittest.TestCase):
    def test_gemini_usage_recorded_with_thinking_tokens(self):
        async def fake_generate_content(model, contents, config):
            return FakeGeminiResponse(usage_metadata=FakeUsageMetadata(
                prompt_token_count=100, candidates_token_count=50,
                thoughts_token_count=512, cached_content_token_count=0,
            ))

        handler = _handler_with_fake_gemini(fake_generate_content)
        asyncio.run(handler.call_gemini("prompt", model_name="gemini-2.5-pro"))

        stats = handler.usage_stats["gemini-2.5-pro"]
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["input_tokens"], 100)
        self.assertEqual(stats["output_tokens"], 50)
        self.assertEqual(stats["thinking_tokens"], 512)

        report = handler.get_usage_report()
        self.assertIn("Thinking", report)
        self.assertIn("512", report)

    def test_gpt_usage_recorded_with_reasoning_tokens(self):
        async def fake_create(**kwargs):
            return FakeGptResponse(usage=FakeUsage(
                prompt_tokens=80, completion_tokens=40,
                completion_tokens_details=FakeCompletionTokensDetails(reasoning_tokens=200),
            ))

        handler = _handler_with_fake_gpt(fake_create)
        asyncio.run(handler.call_gpt("prompt", model_name="o3-mini"))

        stats = handler.usage_stats["o3-mini"]
        self.assertEqual(stats["thinking_tokens"], 200)

        report = handler.get_usage_report()
        self.assertIn("Reasoning", report)
        self.assertIn("200", report)

    def test_gpt_usage_without_reasoning_tokens_details(self):
        async def fake_create(**kwargs):
            return FakeGptResponse(usage=FakeUsage(prompt_tokens=10, completion_tokens=5))

        handler = _handler_with_fake_gpt(fake_create)
        asyncio.run(handler.call_gpt("prompt", model_name="gpt-5.4-mini"))

        stats = handler.usage_stats["gpt-5.4-mini"]
        self.assertEqual(stats["thinking_tokens"], 0)

    def test_multiple_calls_accumulate(self):
        calls = {"n": 0}

        async def fake_generate_content(model, contents, config):
            calls["n"] += 1
            return FakeGeminiResponse(usage_metadata=FakeUsageMetadata(
                prompt_token_count=10, candidates_token_count=5,
            ))

        handler = _handler_with_fake_gemini(fake_generate_content)
        asyncio.run(handler.call_gemini("prompt", model_name="gemini-2.5-flash"))
        asyncio.run(handler.call_gemini("prompt", model_name="gemini-2.5-flash"))

        stats = handler.usage_stats["gemini-2.5-flash"]
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["input_tokens"], 20)
        self.assertEqual(stats["output_tokens"], 10)

    def test_report_omits_provider_with_zero_calls(self):
        async def fake_create(**kwargs):
            return FakeGptResponse(usage=FakeUsage(prompt_tokens=1, completion_tokens=1))

        handler = _handler_with_fake_gpt(fake_create)
        asyncio.run(handler.call_gpt("prompt", model_name="gpt-5.4-mini"))

        report = handler.get_usage_report()
        self.assertNotIn("[Gemini]", report)
        self.assertIn("[OpenAI]", report)

    def test_report_empty_when_no_calls(self):
        handler = ModelHandler(gemini_api_key=None, openai_api_key=None)
        self.assertEqual(handler.get_usage_report(), "")

    def test_thinking_budget_passed_to_gemini_config(self):
        captured = {}

        async def fake_generate_content(model, contents, config):
            captured["config"] = config
            return FakeGeminiResponse(usage_metadata=FakeUsageMetadata())

        handler = _handler_with_fake_gemini(fake_generate_content)
        asyncio.run(handler.call_gemini("prompt", model_name="gemini-2.5-pro", thinking_budget=4096))
        self.assertEqual(captured["config"].thinking_config.thinking_budget, 4096)

        captured.clear()
        asyncio.run(handler.call_gemini("prompt", model_name="gemini-2.5-pro"))
        # thinking_budget=None (기본) -> config 자체가 None일 수 있음(시스템 지시/응답형식도 없으므로)
        self.assertIsNone(captured["config"])

    def test_gpt_reasoning_effort_passed_when_set(self):
        captured = {}

        async def fake_create(**kwargs):
            captured.update(kwargs)
            return FakeGptResponse(usage=FakeUsage(prompt_tokens=1, completion_tokens=1))

        handler = _handler_with_fake_gpt(fake_create)
        asyncio.run(handler.call_gpt("prompt", model_name="o3-mini", reasoning_effort="high"))
        self.assertEqual(captured["reasoning_effort"], "high")

    def test_generate_content_forwards_params_to_correct_provider_only(self):
        gemini_calls = {}
        gpt_calls = {}

        async def fake_call_gemini(prompt, model_name, system_instruction=None, response_json=False, thinking_budget=None):
            gemini_calls["thinking_budget"] = thinking_budget
            return "gemini ok"

        async def fake_call_gpt(prompt, model_name, system_instruction=None, response_json=False, reasoning_effort=None):
            gpt_calls["reasoning_effort"] = reasoning_effort
            return "gpt ok"

        handler = ModelHandler(gemini_api_key=None, openai_api_key=None)
        handler.call_gemini = fake_call_gemini
        handler.call_gpt = fake_call_gpt

        asyncio.run(handler.generate_content("p", model_name="o3-mini", reasoning_effort="medium", thinking_budget=999))
        self.assertEqual(gpt_calls["reasoning_effort"], "medium")
        self.assertEqual(gemini_calls, {})

        asyncio.run(handler.generate_content("p", model_name="gemini-2.5-pro", thinking_budget=777, reasoning_effort="low"))
        self.assertEqual(gemini_calls["thinking_budget"], 777)
        self.assertEqual(set(gpt_calls.keys()) - {"reasoning_effort"}, set())


if __name__ == "__main__":
    unittest.main()
