import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI 兼容接口 Provider。"""

    def __init__(self, api_key: str, base_url: str, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> Dict[str, Any]:
        content = self.chat(messages, temperature=temperature)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            logger.error(f"Failed to parse JSON from response: {content[:200]}")
            return {"error": "JSON parse failed", "raw": content}
