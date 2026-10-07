"""End-to-end checks against a running Qwen3 server using the OpenAI SDK.

Start the server first, then run:
    .venv/bin/python -m unittest discover -s tests -p 'test_api_e2e.py' -v
"""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import BadRequestError, OpenAI


FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_chat_golden.json"
BASE_URL = os.environ.get("SGLANG_E2E_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("SGLANG_E2E_MODEL", "default")


class ApiE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = OpenAI(base_url=BASE_URL, api_key="unused", timeout=120.0, max_retries=0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def chat(self, messages, *, max_tokens=1, stream=False, tools=None, extra_body=None):
        return self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools or [],
            temperature=0,
            max_tokens=max_tokens,
            stream=stream,
            extra_body={"ignore_eos": True, **(extra_body or {})},
        )

    def completion(self, prompt, *, max_tokens=1, stream=False):
        return self.client.completions.create(
            model=MODEL,
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            stream=stream,
            extra_body={"ignore_eos": True},
        )

    def test_non_greedy_mixed_batch(self):
        settings = [(0.8, -1, 1.0), (0.8, 50, 0.9), (0.8, -1, 0.9),
                    (0.0, -1, 1.0), (1.0, 1, 1.0), (0.8, 50, 0.0)]

        def generate(item):
            index, (temperature, top_k, top_p) = item
            response = self.client.completions.create(
                model=MODEL, prompt=f"Task {index}: Explain why the sky is blue.",
                temperature=temperature, top_p=top_p, max_tokens=17,
                extra_body={"top_k": top_k, "ignore_eos": True},
            )
            self.assertEqual(response.usage.completion_tokens, 17)
            self.assertEqual(response.choices[0].finish_reason, "length")
            self.assertTrue(response.choices[0].text)

        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(generate, enumerate(settings * 2)))

    def test_list_models(self):
        models = self.client.models.list()
        self.assertEqual(models.object, "list")
        self.assertEqual(len(models.data), 1)
        model = models.data[0]
        self.assertTrue(model.id)
        self.assertEqual(model.object, "model")
        self.assertIsInstance(model.created, int)
        self.assertGreater(model.created, 0)
        self.assertEqual(model.owned_by, "sglang-rust")
        response = self.client.completions.create(
            model=model.id, prompt="Say hi", max_tokens=1
        )
        self.assertEqual(response.model, model.id)

    def test_max_tokens_and_usage(self):
        for limit in (1, 3):
            with self.subTest(max_tokens=limit):
                result = self.completion("Say hi", max_tokens=limit)
                self.assertEqual(result.object, "text_completion")
                self.assertEqual(result.choices[0].finish_reason, "length")
                self.assertEqual(result.usage.completion_tokens, limit)
                self.assertGreater(result.usage.prompt_tokens, 0)
                self.assertEqual(
                    result.usage.total_tokens,
                    result.usage.prompt_tokens + limit,
                )

    def test_chat_stream_matches_non_stream(self):
        messages = [{"role": "user", "content": "Say hi"}]
        full = self.chat(messages, max_tokens=3)
        chunks = list(self.chat(messages, max_tokens=3, stream=True))

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.object == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual({chunk.id for chunk in chunks}, {chunks[0].id})
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")
        self.assertEqual(chunks[-1].usage.completion_tokens, 3)
        self.assertEqual(chunks[-1].usage.prompt_tokens, full.usage.prompt_tokens)
        self.assertEqual(
            "".join(chunk.choices[0].delta.content or "" for chunk in chunks),
            full.choices[0].message.content or "",
        )
        self.assertEqual(
            "".join(
                getattr(chunk.choices[0].delta, "reasoning_content", None) or ""
                for chunk in chunks
            ),
            getattr(full.choices[0].message, "reasoning_content", None) or "",
        )

    def test_chat_max_completion_tokens_and_precedence(self):
        cases = (
            {"max_completion_tokens": 3},
            {"max_tokens": 1, "max_completion_tokens": 3},
            {"max_tokens": 3, "max_completion_tokens": None},
        )
        for stream in (False, True):
            for limits in cases:
                with self.subTest(stream=stream, limits=limits):
                    response = self.client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": "Say hi"}],
                        temperature=0,
                        stream=stream,
                        extra_body={"ignore_eos": True},
                        **limits,
                    )
                    result = list(response)[-1] if stream else response
                    self.assertEqual(result.choices[0].finish_reason, "length")
                    self.assertEqual(result.usage.completion_tokens, 3)
                    self.assertEqual(
                        result.usage.total_tokens, result.usage.prompt_tokens + 3
                    )

    def test_completion_stream_matches_non_stream(self):
        full = self.completion("Say hi", max_tokens=3)
        chunks = list(self.completion("Say hi", max_tokens=3, stream=True))

        self.assertTrue(chunks)
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")
        self.assertEqual(chunks[-1].usage.completion_tokens, 3)
        self.assertEqual(chunks[-1].usage.prompt_tokens, full.usage.prompt_tokens)
        self.assertEqual(
            "".join(chunk.choices[0].text for chunk in chunks),
            full.choices[0].text,
        )

    def test_concurrent_chat_limits_and_usage(self):
        # Run with max-running-req >= 2 to exercise packed QKV decode and
        # row compaction; a single-request server also validates queue reuse.
        def generate(item):
            index, limit = item
            result = self.client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Explain topic {index}: "
                    + "the blue sky and atmosphere " * (1 + index * 12),
                }],
                temperature=0,
                max_completion_tokens=limit,
                extra_body={"ignore_eos": True},
            )
            return limit, result

        limits = [1, 3, 16, 48, 32, 8, 64, 17, 2, 24, 4, 80]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for limit, result in pool.map(generate, enumerate(limits)):
                self.assertEqual(result.choices[0].finish_reason, "length")
                self.assertEqual(result.usage.completion_tokens, limit)
                self.assertEqual(
                    result.usage.total_tokens, result.usage.prompt_tokens + limit
                )

    def test_chat_matches_hugging_face_golden_prompts(self):
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cases_by_name = {case["name"]: case for case in cases}
        self.assertTrue(
            {"simple", "tools_and_system", "tool_calls_and_responses", "reasoning_content"}
            <= cases_by_name.keys()
        )
        tool_history = cases_by_name["tool_calls_and_responses"]["messages"]
        self.assertTrue(
            any(message.get("content", "missing") is None for message in tool_history)
        )
        self.assertTrue(any(message.get("tool_calls") for message in tool_history))
        self.assertTrue(
            any(
                message.get("reasoning_content")
                for message in cases_by_name["reasoning_content"]["messages"]
            )
        )
        for case in cases:
            # The HTTP chat route always adds the assistant generation prompt.
            if not case["add_generation_prompt"]:
                continue
            with self.subTest(case=case["name"]):
                extra_body = {}
                if "enable_thinking" in case:
                    extra_body["enable_thinking"] = case["enable_thinking"]
                chat = self.chat(
                    case["messages"],
                    max_tokens=3,
                    tools=case["tools"],
                    extra_body=extra_body,
                )
                raw = self.completion(case["expected"], max_tokens=3)
                self.assertEqual(chat.object, "chat.completion")
                self.assertEqual(chat.choices[0].message.role, "assistant")
                self.assertEqual(chat.choices[0].finish_reason, "length")
                self.assertEqual(chat.usage.prompt_tokens, raw.usage.prompt_tokens)
                if "<think>" not in raw.choices[0].text and "<tool_call>" not in raw.choices[0].text:
                    self.assertEqual(chat.choices[0].message.content, raw.choices[0].text)

    def test_enable_thinking_in_template_kwargs(self):
        messages = [{"role": "user", "content": "Say hi"}]
        top_level = self.chat(messages, extra_body={"enable_thinking": False})
        in_kwargs = self.chat(
            messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        thinking_enabled = self.chat(messages, extra_body={"enable_thinking": True})
        top_level_wins = self.chat(
            messages,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )

        self.assertEqual(top_level.usage.prompt_tokens, in_kwargs.usage.prompt_tokens)
        self.assertEqual(
            top_level.choices[0].message.content,
            in_kwargs.choices[0].message.content,
        )
        self.assertEqual(top_level.usage.prompt_tokens, top_level_wins.usage.prompt_tokens)
        self.assertNotEqual(top_level.usage.prompt_tokens, thinking_enabled.usage.prompt_tokens)

    def test_reasoning_content_in_full_and_streamed_chat(self):
        messages = [{"role": "user", "content": "What is 1 + 1?"}]
        kwargs = {"max_tokens": 48, "extra_body": {"enable_thinking": True}}
        full = self.chat(messages, **kwargs)
        chunks = list(self.chat(messages, stream=True, **kwargs))

        reasoning = getattr(full.choices[0].message, "reasoning_content", None)
        self.assertTrue(reasoning)
        self.assertNotIn("<think>", reasoning)
        self.assertNotIn("</think>", reasoning)
        self.assertEqual(
            "".join(
                getattr(chunk.choices[0].delta, "reasoning_content", None) or ""
                for chunk in chunks
            ),
            reasoning,
        )
        self.assertEqual(
            "".join(chunk.choices[0].delta.content or "" for chunk in chunks),
            full.choices[0].message.content or "",
        )
        self.assertEqual(chunks[-1].choices[0].finish_reason, full.choices[0].finish_reason)

    def test_tool_calls_in_full_and_streamed_chat(self):
        messages = [{
            "role": "user",
            "content": "What is the weather in Beijing? Call get_weather with city Beijing. "
            "Do not answer from memory.",
        }]
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        kwargs = {
            "tools": tools,
            "max_tokens": 160,
            "extra_body": {"enable_thinking": False, "ignore_eos": False},
        }
        full = self.chat(messages, **kwargs)
        chunks = list(self.chat(messages, stream=True, **kwargs))

        self.assertEqual(full.choices[0].finish_reason, "tool_calls")
        self.assertIsNone(full.choices[0].message.content)
        self.assertEqual(len(full.choices[0].message.tool_calls), 1)
        call = full.choices[0].message.tool_calls[0]
        self.assertEqual(call.type, "function")
        self.assertEqual(call.function.name, "get_weather")
        self.assertEqual(json.loads(call.function.arguments)["city"].lower(), "beijing")

        streamed_calls = [
            delta_call
            for chunk in chunks
            for delta_call in (chunk.choices[0].delta.tool_calls or [])
        ]
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(streamed_calls), 1)
        self.assertEqual(streamed_calls[0].index, 0)
        self.assertEqual(streamed_calls[0].function.name, call.function.name)
        self.assertEqual(streamed_calls[0].function.arguments, call.function.arguments)
        self.assertFalse(any(chunk.choices[0].delta.content for chunk in chunks))

    def test_malformed_tool_call_returns_bad_request(self):
        with self.assertRaises(BadRequestError) as caught:
            self.chat(
                [
                    {"role": "user", "content": "Call"},
                    {"role": "assistant", "content": None, "tool_calls": [{"arguments": {}}]},
                ]
            )
        self.assertIn("name", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
