"""Qwen3.8-27B 实际权重回归；使用独立开关，避免误跑在 Qwen3 服务上。"""
import json
from concurrent.futures import ThreadPoolExecutor
import os
import unittest
from pathlib import Path

from .support import ApiTestCase, MODEL

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_8_greedy_golden.json"


@unittest.skipUnless(os.environ.get("SGLANG_E2E_QWEN3_5") == "1", "set SGLANG_E2E_QWEN3_5=1 for Qwen3.8-27B")
class Qwen35ApiTest(ApiTestCase):
    def test_reference_generation_thinking_streaming_and_repeated_requests(self):
        cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for case in [*cases, cases[0]]:
            with self.subTest(prompt=case["text"], thinking=case["thinking"]):
                budget = len(case["output_ids"])
                raw = self.client.completions.create(model=MODEL, prompt=case["prompt"], max_tokens=budget, temperature=0)
                self.assertEqual(raw.choices[0].text, case["output"].removesuffix("<|im_end|>"))
                self.assertEqual(raw.usage.prompt_tokens, len(case["input_ids"]))
                extra = {"ignore_eos": False, "enable_thinking": case["thinking"], "chat_template_kwargs": {"reasoning_effort": "low"}}
                messages = [{"role": "user", "content": case["text"]}]
                chat = self.chat(messages, max_tokens=budget, extra_body=extra)
                content, reasoning = "", ""
                with self.chat(messages, max_tokens=budget, extra_body=extra, stream=True) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        content += delta.content or ""
                        reasoning += getattr(delta, "reasoning_content", None) or ""
                message = chat.choices[0].message
                self.assertEqual(content, message.content or "")
                self.assertEqual(reasoning, getattr(message, "reasoning_content", None) or "")
                self.assertEqual(chat.usage.prompt_tokens, len(case["input_ids"]))
                if case["thinking"]:
                    self.assertTrue(reasoning)
                    self.assertIn("391", content)

    def test_xml_tool_calls_and_openai_history(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather of a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        messages = [{"role": "user", "content": "请调用get_weather查询上海天气。"}]
        extra = {"ignore_eos": False, "enable_thinking": False}
        response = self.chat(messages, tools=tools, max_tokens=128, extra_body=extra)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        call = choice.message.tool_calls[0]
        self.assertEqual(call.function.name, "get_weather")
        self.assertIn("city", json.loads(call.function.arguments))
        messages += [choice.message.model_dump(exclude_none=True), {"role": "tool", "tool_call_id": call.id, "content": "上海晴天，25摄氏度"}]
        followup = self.chat(messages, tools=tools, max_tokens=64, extra_body=extra)
        self.assertTrue(followup.choices[0].message.content)

    def test_checkpoint_capacity_pressure_preserves_cold_and_warm_answers(self):
        # 压力规模可按服务启动日志中的自动 checkpoint 容量设置为 C+8。
        count = int(os.environ.get("SGLANG_E2E_CHECKPOINT_PRESSURE_PREFIXES", "8"))
        prompts = [
            f"Request label: {index:03d}. This label is only metadata.\n"
            + ("The blue sky is clear. " * 20)
            + "\nCompute 7 + 8. Output only the number.\nAnswer:"
            for index in range(count)
        ]
        answers = []
        for prompt in prompts:
            cold = self.client.completions.create(model=MODEL, prompt=prompt, max_tokens=4, temperature=0)
            warm = self.client.completions.create(model=MODEL, prompt=prompt, max_tokens=4, temperature=0)
            self.assertEqual(cold.choices[0].text, warm.choices[0].text)
            self.assertEqual(cold.choices[0].text.strip(), "15")
            answers.append(cold.choices[0].text)
            if len(answers) == 8:
                # 自动容量大于 8 时，首个历史前缀应仍可命中；日志检查实际命中长度。
                revisit = self.client.completions.create(model=MODEL, prompt=prompts[0], max_tokens=4, temperature=0)
                self.assertEqual(revisit.choices[0].text, answers[0])
        after_pressure = self.client.completions.create(model=MODEL, prompt=prompts[0], max_tokens=4, temperature=0)
        self.assertEqual(after_pressure.choices[0].text, answers[0])

    def test_longer_kv_branch_recomputation_and_repeated_answers(self):
        # 原前缀的状态在更深位置，分叉处只有 KV；重算时补出页对齐状态。
        common = "Branch checkpoint regression: " + ("one two three four five. " * 25)
        original = common + "Original suffix. Reply with OK only."
        branch = common + "Different suffix. Reply with OK only."
        first = self.completion(original, max_tokens=4)
        recomputed = self.completion(branch, max_tokens=4)
        restored = self.completion(branch, max_tokens=4)
        original_again = self.completion(original, max_tokens=4)
        self.assertEqual(recomputed.choices[0].text, restored.choices[0].text)
        self.assertEqual(first.choices[0].text, original_again.choices[0].text)

    def test_overlap_incoming_prefills_cancel_and_reuse(self):
        # On a server with >=3 running slots, the incoming requests are
        # prefills mixed with the first request's relayed decode. This also
        # runs against the default single-slot server as a queued-work test.
        prompt = "Concurrent request metadata. Compute 7 + 8. Output only the number.\nAnswer:"

        def answer():
            result = self.client.completions.create(
                model=MODEL, prompt=prompt, max_tokens=4, temperature=0,
            )
            self.assertEqual(result.choices[0].text.strip(), "15")
            return result.choices[0].text

        def cancel():
            with self.completion("Describe the lifecycle of a running process in detail.\nAnswer:",
                                 max_tokens=64, stream=True) as stream:
                first = next(iter(stream))
                self.assertTrue(first.choices)
                # Closing the response cancels while further decode may be
                # queued. A subsequent request must not see that state.

        with ThreadPoolExecutor(max_workers=2) as executor:
            with self.completion("Explain how a computer executes a program in detail.\nAnswer:",
                                 max_tokens=64, stream=True) as stream:
                chunks = iter(stream)
                first = next(chunks)
                self.assertTrue(first.choices)
                incoming = executor.submit(answer)
                cancelled = executor.submit(cancel)
                text = first.choices[0].text
                reason = first.choices[0].finish_reason
                for chunk in chunks:
                    text += chunk.choices[0].text
                    reason = chunk.choices[0].finish_reason or reason
                self.assertTrue(text)
                self.assertEqual(reason, "length")
            expected = incoming.result(timeout=120)
            cancelled.result(timeout=120)
        for _ in range(2):
            self.assertEqual(answer(), expected)
