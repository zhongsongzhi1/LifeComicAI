import hashlib
import logging
import os
from typing import Any, Dict, Optional

from PIL import Image

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# 视觉分析缓存 TTL（秒）- 24小时
VISION_CACHE_TTL = 24 * 60 * 60


class VisionAgent(BaseAgent):
    def __init__(self, qianfan_provider, cache_service=None):
        self.qianfan_provider = qianfan_provider
        self.cache_service = cache_service

    @staticmethod
    def _extract_dominant_colors(image_path: str, num_colors: int = 5) -> list:
        """从图片提取主色调 HEX 值。"""
        try:
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img = img.convert("RGB")
            pixels = list(img.getdata())
            color_counts = {}
            for r, g, b in pixels:
                key = (r, g, b)
                color_counts[key] = color_counts.get(key, 0) + 1
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            hex_colors = []
            for color, _ in sorted_colors[:num_colors]:
                hex_colors.append(f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}")
            return hex_colors
        except Exception as e:
            logger.warning(f"Failed to extract colors from {image_path}: {e}")
            return []

    @staticmethod
    def _file_hash(image_path: str) -> Optional[str]:
        """计算文件内容的 MD5 hash，用于缓存 key。"""
        try:
            if not os.path.exists(image_path):
                return None
            h = hashlib.md5()
            with open(image_path, "rb") as f:
                # 只读取前 1MB 用于快速 hash，足够区分不同照片
                h.update(f.read(1024 * 1024))
            # 加上文件大小，避免碰撞
            size = os.path.getsize(image_path)
            return f"{h.hexdigest()}_{size}"
        except Exception as e:
            logger.warning(f"Failed to hash file {image_path}: {e}")
            return None

    def _get_cache_key(self, image_path: str) -> Optional[str]:
        file_hash = self._file_hash(image_path)
        if not file_hash:
            return None
        return f"vision_analysis:{file_hash}"

    async def run(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        # 尝试缓存
        if self.cache_service:
            cache_key = self._get_cache_key(image_path)
            if cache_key:
                cached = self.cache_service.get(cache_key)
                if cached is not None:
                    cached["photo_id"] = photo_id
                    logger.info(f"Vision analysis cache hit for photo {photo_id}")
                    return cached

        try:
            analysis = self.qianfan_provider.analyze_image(image_path)
            colors = self._extract_dominant_colors(image_path)
            result = {
                "photo_id": photo_id,
                "objects": analysis.get("objects", []),
                "scene_desc": analysis.get("scene_desc", ""),
                "weather": analysis.get("weather", ""),
                "location": analysis.get("location", ""),
                "action": analysis.get("action", ""),
                "emotion": analysis.get("emotion", ""),
                "clothing": analysis.get("clothing", []),
                "people": analysis.get("people", []),
                "lighting": analysis.get("lighting", ""),
                "color_tone": analysis.get("color_tone", ""),
                "colors": colors,
                "people_count": analysis.get("people_count", 0),
            }
            # 写入缓存
            if self.cache_service:
                cache_key = self._get_cache_key(image_path)
                if cache_key:
                    cache_result = {k: v for k, v in result.items() if k != "photo_id"}
                    self.cache_service.set(cache_key, cache_result, ttl=VISION_CACHE_TTL)
            return result
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
                "lighting": "",
                "color_tone": "",
                "people_count": 0,
            }

    async def run_async(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_run, photo_id, image_path)

    def _sync_run(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        # 尝试缓存
        if self.cache_service:
            cache_key = self._get_cache_key(image_path)
            if cache_key:
                cached = self.cache_service.get(cache_key)
                if cached is not None:
                    cached["photo_id"] = photo_id
                    logger.info(f"Vision analysis cache hit for photo {photo_id}")
                    return cached

        try:
            analysis = self.qianfan_provider.analyze_image(image_path)
            colors = self._extract_dominant_colors(image_path)
            result = {
                "photo_id": photo_id,
                "scene_desc": analysis.get("scene_desc", ""),
                "weather": analysis.get("weather", ""),
                "location": analysis.get("location", ""),
                "action": analysis.get("action", ""),
                "emotion": analysis.get("emotion", ""),
                "clothing": analysis.get("clothing", []),
                "people": analysis.get("people", []),
                "objects": analysis.get("objects", []),
                "lighting": analysis.get("lighting", ""),
                "color_tone": analysis.get("color_tone", ""),
                "colors": colors,
                "people_count": analysis.get("people_count", 0),
            }
            # 写入缓存
            if self.cache_service:
                cache_key = self._get_cache_key(image_path)
                if cache_key:
                    cache_result = {k: v for k, v in result.items() if k != "photo_id"}
                    self.cache_service.set(cache_key, cache_result, ttl=VISION_CACHE_TTL)
            return result
        except Exception as e:
            logger.error(f"Vision analysis failed for photo {photo_id}: {e}")
            return {"photo_id": photo_id, "scene_desc": "", "weather": "", "location": "", "action": "", "emotion": "", "clothing": [], "people": [], "objects": [], "lighting": "", "color_tone": "", "people_count": 0}
