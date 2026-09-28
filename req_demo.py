import requests

url = "http://127.0.0.1:8001/v1/completions"

payload = {
    "prompt": "Once upon a time",
    "max_tokens": 30,
    "stream": False,
}

response = requests.post(url, json=payload, timeout=60)
response.raise_for_status()

data = response.json()
print(data)

# 兼容 OpenAI Text Completions 响应格式
print("\n生成结果：")
print(data["choices"][0]["text"])