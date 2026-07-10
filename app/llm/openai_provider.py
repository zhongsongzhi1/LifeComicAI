import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI 兼容接口 Provider（基于 LangChain ChatOpenAI）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self._client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> str:
        try:
            kwargs = {"model": self.model_name, "api_key": self.api_key, "base_url": self.base_url}
            kwargs["temperature"] = temperature if temperature is not None else self.temperature
            kwargs["top_p"] = self.top_p
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if kwargs["temperature"] == self.temperature and max_tokens is None:
                response = self._client.invoke(messages)
            else:
                client = ChatOpenAI(**kwargs)
                response = client.invoke(messages)
            return response.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = None) -> Dict[str, Any]:
        content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            logger.error(f"Failed to parse JSON from response: {content[:200]}")
            return {"error": "JSON parse failed", "raw": content}
