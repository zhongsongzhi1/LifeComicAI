import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx

from app.agents.base import BaseAgent
from app.utils.prompt_enhancer import PromptEnhancer

logger = logging.getLogger(__name__)

# 并发生成的最大数量（ModelScope 限流保护，如需提速可调高）
MAX_CONCURRENT_GENERATION = 3


class ComicAgent(BaseAgent):
    """Comic Agent：根据分镜稿生图并下载到本地（并行化版本）。"""

    def __init__(self, image_provider, comics_dir: str):
        self.image_provider = image_provider
        self.comics_dir = comics_dir
        self.prompt_enhancer = PromptEnhancer()

    async def run(self, **kwargs) -> Dict[str, Any]:
        """根据分镜稿生成漫画图片并下载（并行生成）。

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
        if not pages:
            return {"image_partial": False, "all_failed": True, "pages": []}

        output_dir = os.path.join(self.comics_dir, str(comic_id))
        os.makedirs(output_dir, exist_ok=True)

        sem = asyncio.Semaphore(MAX_CONCURRENT_GENERATION)

        async def _gen_one(page_info: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                page_num = page_info["page"]
                desc = page_info.get("desc", "")
                dialogue = page_info.get("dialogue", "")

                enhanced_prompt = self._build_image_prompt(
                    style_name,
                    desc,
                    dialogue,
                    page_info=page_info
                )

                result = None
                if hasattr(self.image_provider, "generate_async"):
                    try:
                        result = await self.image_provider.generate_async(
                            enhanced_prompt["positive_prompt"],
                            size="1024x1024",
                            negative_prompt=enhanced_prompt.get("negative_prompt", ""),
                            max_retries=3,
                        )
                    except Exception as e:
                        logger.warning(f"Page {page_num} generate_async failed: {e}, fallback to sync")

                if result is None and hasattr(self.image_provider, "generate_with_retry"):
                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(
                            None,
                            lambda: self.image_provider.generate_with_retry(
                                enhanced_prompt["positive_prompt"],
                                negative_prompt=enhanced_prompt.get("negative_prompt", ""),
                                size="1024x1024",
                                max_retries=3,
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Page {page_num} generation failed: {e}")

                if result and result.get("urls"):
                    image_url = result["urls"][0]
                    local_path = os.path.join(output_dir, f"page_{page_num}.png")
                    ok = await self._download_image_async(image_url, local_path)
                    if ok:
                        page_info["image_url"] = local_path
                    else:
                        page_info["image_url"] = ""
                else:
                    page_info["image_url"] = ""

                return page_info

        # 并行生成所有页面
        logger.info(f"ComicAgent: generating {len(pages)} pages with concurrency={MAX_CONCURRENT_GENERATION}")
        results = await asyncio.gather(*[_gen_one(dict(p)) for p in pages])

        # 按 page 号排序，保持原始顺序
        results.sort(key=lambda x: x.get("page", 0))

        success_count = sum(1 for p in results if p.get("image_url"))
        fail_count = len(results) - success_count
        total = len(results)
        all_failed = fail_count == total and total > 0
        image_partial = success_count > 0 and fail_count > 0

        logger.info(f"ComicAgent done: {success_count}/{total} success, {fail_count} failed")

        return {
            "image_partial": image_partial,
            "all_failed": all_failed,
            "pages": results,
        }

    def _build_image_prompt(self, style_name: str, desc: str, dialogue: str, page_info: Dict[str, str] = None) -> Dict[str, str]:
        """使用PromptEnhancer构建生图提示词。"""
        if page_info is None:
            page_info = {}

        shot_type = page_info.get("shot", "Medium")
        characters_detail = page_info.get("characters_detail", "")

        # 根据对白内容选择合适的光影风格
        lighting = "natural"
        if dialogue:
            negative_emotions = ["惊", "怒", "哭", "急", "恐", "绝望"]
            positive_emotions = ["开心", "笑", "幸福", "甜蜜"]

            if any(emotion in dialogue for emotion in negative_emotions):
                lighting = "dramatic"
            elif any(emotion in dialogue for emotion in positive_emotions):
                lighting = "warm"

        return self.prompt_enhancer.enhance_prompt(
            style_name=style_name,
            desc=desc,
            dialogue=dialogue,
            shot_type=shot_type,
            characters_detail=characters_detail,
            lighting=lighting
        )

    @staticmethod
    async def _download_image_async(url: str, local_path: str) -> bool:
        """异步下载图片到本地，返回是否成功。"""
        try:
            async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
            logger.info(f"Image downloaded: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return False

    @staticmethod
    def _download_image(url: str, local_path: str) -> bool:
        """同步下载图片（兼容旧接口）。"""
        try:
            import requests
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Image downloaded: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return False
