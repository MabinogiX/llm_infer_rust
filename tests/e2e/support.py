"""Shared client lifecycle and request helpers; no test cases here."""

import os
import unittest

from openai import OpenAI

MODEL = os.environ.get("SGLANG_E2E_MODEL", "default")


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_url = os.environ.get("SGLANG_E2E_BASE_URL")
        if not base_url:
            raise unittest.SkipTest("set SGLANG_E2E_BASE_URL to opt into live server tests")
        cls.client = OpenAI(base_url=base_url, api_key="unused", timeout=120.0, max_retries=0)

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
