import json
import logging

import qianfan

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
    """百度千帆视觉模型 Provider。"""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self._client = qianfan.ChatCompletion(
            ak=access_key,
            sk=secret_key,
        )

    def analyze_image(self, image_path: str) -> dict:
        """分析单张图片，返回结构化字典。"""
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": VISION_SYSTEM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]}
        ]

        try:
            resp = self._client.do(messages=messages)
            content = resp.get("result", "")
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"Qianfan vision API error: {e}")
            return self._empty_result(str(e))

    def _parse_response(self, content: str) -> dict:
        try:
            return json.loads(content)
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
