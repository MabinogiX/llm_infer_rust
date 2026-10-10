from concurrent.futures import ThreadPoolExecutor

from .support import ApiTestCase, MODEL


class ConcurrentApiTest(ApiTestCase):
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

