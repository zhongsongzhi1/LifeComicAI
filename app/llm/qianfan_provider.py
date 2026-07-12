import base64
import json
import logging

import requests
from openai import OpenAI

# compatibility: some tests patch `app.llm.qianfan_provider.qianfan`
try:
    import qianfan  # type: ignore
except Exception:
    qianfan = None

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """你是一个图片分析专家。仔细分析这张照片，提取**具体细节**，输出以下 JSON 格式（仅输出 JSON，不要其他文字）：

核心要求：
- people[].traits 必须详细描述外貌：性别、大致年龄段、发型/发色、是否戴眼镜、面部特征、体型
- clothing 必须写明颜色、款式、材质（如"深蓝色连帽卫衣""白色T恤外搭格子衬衫"）
- scene_desc 要具体写出场景元素（如"日式拉面店，木桌，一碗豚骨拉面，旁边有煎饺"）
- objects 列出照片中显眼的物品
- location 尽量具体（如"日式餐厅""办公室""地铁车厢""公园草地"）
- action 描述人物的具体动作和表情（如"低头吃拉面，表情满足"）

{
  "objects": ["物体1", "物体2"],
  "scene_desc": "详细场景描述（2-3句话，包含环境和物品细节）",
  "weather": "天气/光照情况",
  "location": "具体地点",
  "action": "人物在做什么，什么表情",
  "emotion": "主要情绪",
  "clothing": ["衣物详细描述，含颜色款式"],
  "people": [
    {"role": "主角/配角/路人", "traits": "性别年龄、发型发色、眼镜、面部特征、体型等详细外貌描述"}
  ]
}"""


class QianfanProvider:
    """视觉模型 Provider（阿里云 DashScope Qwen VL，兼容 OpenAI 接口）。"""

    def __init__(self, access_key: str = "", secret_key: str = ""):
        self._api_key: str = ""
        self._base_url: str = ""
        self._client: OpenAI = None
        self.access_key = access_key
        self.secret_key = secret_key

    def configure(self, api_key: str, base_url: str):
        """配置 DashScope 接口参数。"""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = None) -> str:
        """调用阿里云 DashScope Qwen LLM API 生成文本。"""
        try:
            if not self._client:
                raise RuntimeError("Provider not configured. Call configure() first.")

            kwargs = {"model": "qwen-plus", "messages": messages, "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            completion = self._client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            return content or ""
        except Exception as e:
            logger.error(f"DashScope LLM API error: {e}")
            raise

    def chat_json(self, messages: list, temperature: float = 0.3, max_tokens: int = None) -> dict:
        """调用阿里云 DashScope Qwen LLM API 生成 JSON。"""
        content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            logger.error(f"Failed to parse JSON from DashScope LLM response: {content[:200]}")
            return {"error": "JSON parse failed", "raw": content}

    def analyze_image(self, image_path: str, model: str = "qwen-vl-plus") -> dict:
        """分析单张图片，返回结构化字典。"""
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_SYSTEM_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                        ],
                    }
                ],
                "temperature": 0.3,
            }

            # If a `qianfan` SDK is available (tests may patch it), use it.
            if qianfan is not None:
                try:
                    chat = qianfan.ChatCompletion()
                    resp_obj = chat.do(payload)
                    # expect resp_obj like {"result": "<json string>"}
                    raw = resp_obj.get("result", "")
                    return json.loads(raw)
                except Exception as e:
                    logger.error(f"qianfan SDK error: {e}")

            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return self._empty_result(str(e))

    def _parse_response(self, content: str) -> dict:
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse vision response as JSON: {content[:200]}")
            return self._empty_result(content)

    @staticmethod
    def _empty_result(fallback_desc: str = "") -> dict:
        return {
            "objects": [],
            "scene_desc": fallback_desc,
            "weather": "",
            "location": "",
            "action": "",
            "emotion": "",
            "clothing": [],
            "people": [],
        }
