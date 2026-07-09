import json
import logging
from typing import Any, Dict

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class DirectorAgent(BaseAgent):
    """Director Agent：根据风格模板调整分镜描述。"""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, **kwargs) -> Dict[str, Any]:
        """根据风格模板修订分镜脚本。

        Args:
            storyboard_with_dialogue: 带对白的分镜 { total_pages, pages: [{ page, shot, desc, dialogue, source_scene_seq }] }
            style_config: 风格配置 { style_key, style_name, prompt_template, mood, dialogue_level, cinematic }

        Returns:
            dict: { style_notes, pages, director_applied }
        """
        storyboard = kwargs.get("storyboard_with_dialogue", {})
        style_config = kwargs.get("style_config", {})
        pages = storyboard.get("pages", [])

        if not pages:
            return {"style_notes": "", "pages": [], "director_applied": False}

        try:
            prompt_template = style_config.get("prompt_template", "")
            storyboard_json = json.dumps(storyboard, ensure_ascii=False)
            system_prompt = prompt_template.replace("{storyboard_json}", storyboard_json)

            messages = [
                {"role": "system", "content": system_prompt},
            ]

            result = await self.llm_provider.chat_json(messages, temperature=0.5)
            if result and result.get("pages"):
                result["director_applied"] = True
                return result

            return self._default_result(pages)
        except Exception as e:
            logger.error(f"Director revision failed: {e}")
            return self._default_result(pages)

    def _default_result(self, pages: list) -> Dict[str, Any]:
        """返回原始 pages，标记 director_applied=False。"""
        return {
            "style_notes": "",
            "pages": pages,
            "director_applied": False,
        }
