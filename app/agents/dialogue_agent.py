import json
import logging
from typing import Any, Dict

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

DIALOGUE_SYSTEM_PROMPT = """你是一位专业漫画对白编剧。根据给定的分镜脚本，为每一页生成自然、生动的角色对白。

对白要求：
- 语言自然口语化，符合角色性格
- 对白内容与画面描述(desc)呼应
- 封面页(page=1)通常无对白或只有标题性对白
- 无对白的页面 dialogue 为空字符串 ""
- 每页对白控制在 1-3 句话

输出格式（严格 JSON）：
{"pages": [{"page": 1, "dialogue": ""}, {"page": 2, "dialogue": "..."}, ...]}"""


class DialogueAgent(BaseAgent):
    """Dialogue Agent：根据分镜脚本为漫画页生成角色对白。"""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, **kwargs) -> Dict[str, Any]:
        """为漫画分镜页生成角色对白。

        Args:
            storyboard: 分镜脚本 { total_pages, pages: [{ page, shot, desc, source_scene_seq }] }
            story_title: 故事标题

        Returns:
            dict: { pages: [{ page, dialogue }] }
        """
        storyboard = kwargs.get("storyboard", {})
        story_title = kwargs.get("story_title", "")
        pages = storyboard.get("pages", [])

        if not pages:
            return {"pages": []}

        try:
            user_prompt = self._build_prompt(storyboard, story_title)
            messages = [
                {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.7)
            if result and result.get("pages"):
                return result
            return self._default_dialogue(storyboard)
        except Exception as e:
            logger.error(f"Dialogue generation failed: {e}")
            return self._default_dialogue(storyboard)

    def _build_prompt(self, storyboard: Dict[str, Any], story_title: str) -> str:
        """构建发送给 LLM 的提示词。"""
        prompt_data = {
            "title": story_title,
            "total_pages": storyboard.get("total_pages", 0),
            "pages": storyboard.get("pages", []),
        }
        return json.dumps(prompt_data, ensure_ascii=False, indent=2)

    def _default_dialogue(self, storyboard: Dict[str, Any]) -> Dict[str, Any]:
        """为每个页面返回空对白。"""
        pages = storyboard.get("pages", [])
        return {
            "pages": [
                {"page": p.get("page", i + 1), "dialogue": ""}
                for i, p in enumerate(pages)
            ]
        }
