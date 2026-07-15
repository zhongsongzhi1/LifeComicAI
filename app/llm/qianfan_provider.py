import base64
import json
import logging

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """你是一个图片分析专家。仔细分析这张照片，提取**具体细节**，输出以下 JSON 格式（仅输出 JSON，不要其他文字）：

核心要求：
- 【重要】必须识别照片中的所有人物，即使是部分露出的也要列出，不要遗漏任何人！
- people[].traits 必须详细描述外貌：性别、大致年龄段、发型/发色、是否戴眼镜、面部特征、体型、肤色
- people[].clothing：该人物的服装描述（颜色、款式、材质），如"深蓝色连帽卫衣，黑色牛仔裤"
- people[].face_position：脸部在画面中的大致位置，格式为 "水平_垂直"，水平可选 left/center/right，垂直可选 top/middle/bottom，例如 "center_middle" 表示画面中央
- people[].facing：人物面向方向，可选 front（朝向镜头）/left（面朝左）/right（面朝右）/back（背对）
- people[].position_rel：人物在画面中的相对位置关系，如 "左侧，靠近窗户" / "中间偏右，坐在桌子旁"
- people[].action：该人物的具体动作（如"微笑着看向右侧，右手举起比耶"）
- people[].emotion：该人物的表情情绪
- clothing（顶层）：汇总所有人物的服装关键词
- scene_desc 要具体写出场景元素（如"日式拉面店，木桌，一碗豚骨拉面，旁边有煎饺，暖黄色灯光，木质装修"），包含光线、色彩、环境细节
- lighting：光线描述，如 "温暖的室内黄光，柔和侧光" / "明亮自然日光，从左侧窗户照入" / "柔和漫射光，阴影不明显"
- color_tone：画面整体色调，如 "暖色调，橙黄为主" / "冷色调，蓝绿为主" / "清新明亮，色彩饱和度高" / "柔和淡雅，低饱和度"
- objects 列出照片中显眼的物品
- location 尽量具体（如"日式餐厅""办公室""地铁车厢""公园草地"）
- action（顶层）：整体场景的动作描述
- people_count：照片中的人物数量

{
  "people_count": 3,
  "objects": ["物体1", "物体2"],
  "scene_desc": "详细场景描述（2-3句话，包含环境、物品、光线、色彩细节）",
  "lighting": "光线描述（方向、强度、色温）",
  "color_tone": "画面整体色调",
  "weather": "天气/光照情况",
  "location": "具体地点",
  "action": "整体场景动作描述",
  "emotion": "整体主要情绪",
  "clothing": ["衣物关键词汇总"],
  "people": [
    {"role": "主角/配角/路人", "traits": "性别年龄、发型发色、眼镜、面部特征、体型肤色等详细外貌描述", "clothing": "服装描述，含颜色款式", "face_position": "left_top/center_middle/right_bottom等", "facing": "front/left/right/back", "position_rel": "在画面中的位置描述", "action": "人物动作", "emotion": "表情情绪"}
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
        """鲁棒的 JSON 解析：优先解析失败时逐步降级。"""
        import re
        import json

        text = content.strip()

        # 步骤1：直接尝试解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 步骤2：去掉 markdown 代码块标记
        if text.startswith("```"):
            # 去掉开头的 ```json 或 ```
            lines = text.split("\n")
            # 去掉第一行的 ``` 标记行
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # 去掉最后一行的 ``` 标记行
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # 步骤3：提取第一个 { 到最后一个 } 之间的内容（最外层 JSON 对象提取
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 步骤4：用正则尝试找 JSON 对象
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 全部失败，返回空结果（但 scene_desc 用原始文本的前200字
        logger.warning(f"Failed to parse vision response as JSON: {content[:200]}")
        return self._empty_result(content[:200])

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
