import json
import logging
from typing import Any, Dict

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

DIALOGUE_SYSTEM_PROMPT = """你是一位专业漫画对白编剧。根据给定的分镜脚本，为每一页生成自然、生动的角色对白。

额外上下文字段（可选）:
- `characters_info`: 角色列表和性格说明
- `visual_details`: 视觉细节（场景、动作、表情、构图、光影）
- `tone`: 希望对白的语气（例如：幽默、温情、沉重、神秘）

对白要求：
- 语言自然口语化，符合角色性格
- 对白内容与画面描述(desc)呼应，并可适当反映 `visual_details` 与 `tone`
- 封面页(page=1)通常无对白或只有标题性对白
- 无对白的页面 dialogue 为空字符串 ""
- 每页对白控制在 1-3 句话

输出格式（严格 JSON）:
{"pages": [{"page": 1, "dialogue": ""}, {"page": 2, "dialogue": "..."}, ...]}

如果无法解析或返回格式不正确，请返回最保守的默认结构（每页空对白）。"""


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
        # optional enhanced context
        characters_info = kwargs.get("characters_info", None)
        visual_details = kwargs.get("visual_details", None)
        tone = kwargs.get("tone", None)

        if not pages:
            return {"pages": []}

        try:
            user_prompt = self._build_prompt(
                storyboard, story_title, characters_info, visual_details, tone
            )
            messages = [
                {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.7)
            # validate structure
            if isinstance(result, dict) and isinstance(result.get("pages"), list):
                # ensure each page has expected keys
                valid = True
                for p in result["pages"]:
                    if not (isinstance(p, dict) and "page" in p and "dialogue" in p):
                        valid = False
                        break
                if valid:
                    return result
            return self._default_dialogue(storyboard)
        except Exception as e:
            logger.error(f"Dialogue generation failed: {e}")
            return self._default_dialogue(storyboard)

    def _build_prompt(
        self,
        storyboard: Dict[str, Any],
        story_title: str,
        characters_info: Any = None,
        visual_details: Any = None,
        tone: str | None = None,
    ) -> str:
        """构建发送给 LLM 的提示词，包含可选的角色与视觉上下文。"""
        prompt_data = {
            "title": story_title,
            "total_pages": storyboard.get("total_pages", 0),
            "pages": storyboard.get("pages", []),
        }
        if characters_info:
            prompt_data["characters_info"] = characters_info
        if visual_details:
            prompt_data["visual_details"] = visual_details
        if tone:
            prompt_data["tone"] = tone

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
