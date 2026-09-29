import requests

url = "http://127.0.0.1:8001/v1/completions"

content = """
火红的太阳，用自己的能力，将清晨的第一抹光亮洒向大地。这似火的朝阳告诉我，人不能只活在过去，活在不停地抱怨“为什么会这样”的遗憾里，这样，你的人生将是一片灰暗；这似火的朝阳告诉我，要对未知的世界进行探寻，一遍遍追问“为什么不能这样？”朝阳告诉我，事件之所以会流逝，就是为了鼓舞人们向前奔跑。
"""

payload = {
    "prompt": content,
    "max_tokens": 100,
    "stream": False,
}

response = requests.post(url, json=payload, timeout=60)
response.raise_for_status()

data = response.json()
print(data)

# 兼容 OpenAI Text Completions 响应格式
print("\n生成结果：")
print(data["choices"][0]["text"])