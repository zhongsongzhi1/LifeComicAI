import logging
import os
from typing import Any, Dict

import requests

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ComicAgent(BaseAgent):
    """Comic Agent：根据分镜稿生图并下载到本地。"""

    def __init__(self, image_provider, comics_dir: str):
        self.image_provider = image_provider
        self.comics_dir = comics_dir

    async def run(self, **kwargs) -> Dict[str, Any]:
        """根据分镜稿生成漫画图片并下载。

        Args:
            revised_storyboard: 修订后的分镜稿 { total_pages, pages }
            comic_id: 漫画 ID
            style_key: 风格 key
            style_name: 风格名称

        Returns:
            dict: { image_partial, all_failed, pages }
        """
        revised_storyboard = kwargs.get("revised_storyboard", {})
        comic_id = kwargs.get("comic_id", 0)
        style_key = kwargs.get("style_key", "")
        style_name = kwargs.get("style_name", "")

        pages = revised_storyboard.get("pages", [])

        output_dir = os.path.join(self.comics_dir, str(comic_id))
        os.makedirs(output_dir, exist_ok=True)

        fail_count = 0
        success_count = 0

        for page_info in pages:
            page_num = page_info["page"]
            desc = page_info.get("desc", "")
            dialogue = page_info.get("dialogue", "")

            prompt = self._build_image_prompt(style_name, desc, dialogue)
            result = self.image_provider.generate_with_retry(
                prompt, size="1024x1024", max_retries=2
            )

            if result and result.get("urls"):
                image_url = result["urls"][0]
                local_path = os.path.join(output_dir, f"page_{page_num}.png")
                ok = self._download_image(image_url, local_path)
                if ok:
                    page_info["image_url"] = local_path
                    success_count += 1
                else:
                    page_info["image_url"] = ""
                    fail_count += 1
            else:
                page_info["image_url"] = ""
                fail_count += 1

        total = len(pages)
        all_failed = fail_count == total and total > 0
        image_partial = success_count > 0 and fail_count > 0

        return {
            "image_partial": image_partial,
            "all_failed": all_failed,
            "pages": pages,
        }

    @staticmethod
    def _build_image_prompt(style_name: str, desc: str, dialogue: str) -> str:
        """构建生图提示词。"""
        return (
            f"{style_name}风格漫画，{desc}。"
            f"角色对白：「{dialogue}」。"
            f"日系漫画分镜，精细线稿，高质量上色。"
        )

    @staticmethod
    def _download_image(url: str, local_path: str) -> bool:
        """下载图片到本地，返回是否成功。"""
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Image downloaded: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return False
