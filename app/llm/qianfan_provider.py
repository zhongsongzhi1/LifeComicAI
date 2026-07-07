import base64
import json
import logging

import requests

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """你是一个图片分析专家。分析这张照片，输出以下 JSON 格式（仅输出 JSON，不要其他文字）：

{
  "objects": ["物体1", "物体2"],
  "scene_desc": "场景描述（一句话）",
  "weather": "天气/光照情况",
  "location": "地点",
  "action": "人物在做什么",
  "emotion": "主要情绪",
  "clothing": ["衣物描述"],
  "people": [
    {"role": "主角/配角/路人", "traits": "外貌特征描述"}
  ]
}"""


class QianfanProvider:
    """视觉模型 Provider（阿里云 DashScope Qwen VL，兼容 OpenAI 接口）。"""

    def __init__(self, access_key: str = "", secret_key: str = ""):
        self._api_key: str = ""
        self._base_url: str = ""

    def configure(self, api_key: str, base_url: str):
        """配置 DashScope 接口参数。"""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

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

            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
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
