from .support import ApiTestCase, MODEL


class ProtocolApiTest(ApiTestCase):
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
        self.assertEqual({chunk.model for chunk in chunks}, {full.model})
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
        self.assertTrue(all(chunk.object == "text_completion" for chunk in chunks))
        self.assertEqual({chunk.id for chunk in chunks}, {chunks[0].id})
        self.assertEqual({chunk.model for chunk in chunks}, {full.model})
        self.assertEqual(chunks[-1].choices[0].finish_reason, "length")
        self.assertEqual(chunks[-1].usage.completion_tokens, 3)
        self.assertEqual(chunks[-1].usage.prompt_tokens, full.usage.prompt_tokens)
        self.assertEqual(
            "".join(chunk.choices[0].text for chunk in chunks),
            full.choices[0].text,
        )
