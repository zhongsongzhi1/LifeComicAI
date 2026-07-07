import json
import logging
from typing import Any, Dict, List

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

STORY_SYSTEM_PROMPT = """你是一个生活故事讲述者。根据提供的时间线和每张照片的视觉分析结果，创作一个连贯的故事。

要求：
1. 为这个故事起一个标题（story_title）和一句话摘要（story_summary）
2. 将照片分组为场景（scenes），每个场景是一个时间+地点的故事片段
3. 识别故事中的角色（global_characters），包括他们的特征

输出格式（仅输出 JSON，不要其他文字）：
{
  "story_title": "故事标题",
  "story_summary": "一句话故事摘要",
  "scenes": [
    {
      "seq": 1,
      "time": "08:30",
      "location": "地点名称",
      "summary": "这个场景发生了什么",
      "characters": [{"name": "角色名", "role": "主角/配角"}],
      "actions": [{"verb": "动作", "object": "对象"}],
      "emotions": [{"type": "情绪", "intensity": 0.7}],
      "source_photos": [1, 2]
    }
  ],
  "global_characters": [
    {"name": "角色名", "traits": ["特征1", "特征2"], "first_appearance_scene": 1}
  ]
}"""


class StoryAgent(BaseAgent):
    """Story Agent：理解时间线 + 视觉数据 → 生成连贯的 Story Graph。"""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, timeline: Dict[str, Any], analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成故事图谱。

        Args:
            timeline: Album Agent 输出的时间线
            analyses: Vision Agent 输出的分析结果列表

        Returns:
            dict: StoryGraph 字典 { story_title, story_summary, scenes, global_characters }
        """
        user_prompt = self._build_prompt(timeline, analyses)

        try:
            messages = [
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.7)
            return result
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            return self._empty_result()

    def _build_prompt(self, timeline: Dict[str, Any], analyses: List[Dict[str, Any]]) -> str:
        """构建发送给 LLM 的提示词。"""
        analysis_map = {a["photo_id"]: a for a in analyses}

        entries_info = []
        for entry in timeline.get("entries", []):
            photo_analyses = []
            for pid in entry.get("photos", []):
                if pid in analysis_map:
                    a = analysis_map[pid]
                    photo_analyses.append({
                        "photo_id": pid,
                        "scene_desc": a.get("scene_desc", ""),
                        "location": a.get("location", ""),
                        "action": a.get("action", ""),
                        "emotion": a.get("emotion", ""),
                        "people": a.get("people", []),
                    })

            entries_info.append({
                "timeline_entry": {
                    "seq": entry.get("seq"),
                    "time": entry.get("time"),
                    "label": entry.get("label"),
                },
                "photos": photo_analyses,
            })

        prompt_data = {
            "date": timeline.get("date"),
            "entries": entries_info,
        }
        return json.dumps(prompt_data, ensure_ascii=False, indent=2)

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "story_title": "",
            "story_summary": "",
            "scenes": [],
            "global_characters": [],
        }
