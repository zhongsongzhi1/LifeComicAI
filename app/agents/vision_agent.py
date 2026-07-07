import logging
from typing import Any, Dict

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    def __init__(self, qianfan_provider):
        self.qianfan_provider = qianfan_provider

    async def run(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        try:
            analysis = self.qianfan_provider.analyze_image(image_path)
            return {
                "photo_id": photo_id,
                "objects": analysis.get("objects", []),
                "scene_desc": analysis.get("scene_desc", ""),
                "weather": analysis.get("weather", ""),
                "location": analysis.get("location", ""),
                "action": analysis.get("action", ""),
                "emotion": analysis.get("emotion", ""),
                "clothing": analysis.get("clothing", []),
                "people": analysis.get("people", []),
            }
        except Exception as e:
            logger.error(f"Vision analysis failed for photo {photo_id}: {e}")
            return {
                "photo_id": photo_id,
                "objects": [],
                "scene_desc": "",
                "weather": "",
                "location": "",
                "action": "",
                "emotion": "",
                "clothing": [],
                "people": [],
            }
