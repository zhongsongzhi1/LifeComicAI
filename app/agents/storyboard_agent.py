import json
import logging
from typing import Any, Dict

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

STORYBOARD_SYSTEM_PROMPT = """你是一位专业漫画分镜师。根据给定的故事场景列表，为每个场景拆分为 1-2 个漫画分镜页。

分镜要求：
- 镜头类型：Wide（全景）、Medium（中景）、Close-up（特写）、Action（动作）、Ending（结尾）
- 描述要求：具体的视觉描述，包括构图、光线、氛围、角色动作和表情
- source_scene_seq：标记该分镜页来源于哪个场景（从 0 开始，封面页用 0）
- 为故事封面单独生成一页（Wide 镜头，page=1）

输出格式（严格 JSON）：
{"total_pages": N, "pages": [{"page": 1, "shot": "Wide", "desc": "...", "source_scene_seq": 0}, ...]}"""


class StoryboardAgent(BaseAgent):
    """Storyboard Agent：根据 Story Graph 拆分为漫画分镜页。"""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, **kwargs) -> Dict[str, Any]:
        """将故事场景拆分为漫画分镜页。

        Args:
            story_graph: Story Graph 字典 { scenes, characters, title }

        Returns:
            dict: { total_pages, pages: [{ page, shot, desc, source_scene_seq }] }
        """
        story_graph = kwargs.get("story_graph", {})
        scenes = story_graph.get("scenes", [])

        if not scenes:
            return {"total_pages": 0, "pages": []}

        try:
            user_prompt = self._build_prompt(story_graph)
            messages = [
                {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.6)
            if result and result.get("pages"):
                return result
            return self._default_storyboard(story_graph)
        except Exception as e:
            logger.error(f"Storyboard generation failed: {e}")
            return self._default_storyboard(story_graph)

    def _build_prompt(self, story_graph: Dict[str, Any]) -> str:
        """构建发送给 LLM 的提示词。"""
        prompt_data = {
            "title": story_graph.get("title", ""),
            "characters": story_graph.get("characters", {}),
            "scenes": story_graph.get("scenes", []),
        }
        return json.dumps(prompt_data, ensure_ascii=False, indent=2)

    def _default_storyboard(self, story_graph: Dict[str, Any]) -> Dict[str, Any]:
        """为每个场景生成默认的 Medium 镜头分镜页。"""
        scenes = story_graph.get("scenes", [])
        pages = []
        for i, scene in enumerate(scenes):
            pages.append({
                "page": i + 1,
                "shot": "Medium",
                "desc": scene.get("summary", scene.get("narration", "")),
                "source_scene_seq": scene.get("seq", i),
            })
        return {"total_pages": len(pages), "pages": pages}
