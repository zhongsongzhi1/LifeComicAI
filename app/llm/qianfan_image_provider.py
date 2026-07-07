import logging
import random
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# 千帆文生图 API
IMAGE_GEN_URL = "https://qianfan.baidubce.com/v2/musesteamer/images/generations"


class QianfanImageProvider:
    """百度千帆文生图 Provider。"""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        model: str = "musesteamer-air-image",
        seed: Optional[int] = None,
        prompt_extend: bool = True,
    ) -> Optional[dict]:
        """调用千帆文生图 API 生成图片。

        Args:
            prompt: 图片描述提示词
            size: 图片尺寸，默认 1024x1024
            model: 模型名称
            seed: 随机种子，不传则随机生成
            prompt_extend: 是否开启提示词扩展

        Returns:
            dict: {"urls": [...], "raw": ...} 或 None
        """
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        # AK 可能已含 ALTAK- 前缀，兼容两种格式
        ak = self.access_key if self.access_key.startswith("ALTAK-") else f"ALTAK-{self.access_key}"
        headers = {
            "Authorization": f"Bearer bce-v3/{ak}/{self.secret_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "response_format": "url",
        }

        try:
            logger.info(f"Calling Qianfan image gen: prompt={prompt[:50]}..., size={size}")
            resp = requests.post(IMAGE_GEN_URL, headers=headers, json=payload, timeout=60)
            logger.info(f"Response status: {resp.status_code}, body: {resp.text[:300]}")
            resp.raise_for_status()
            result = resp.json()

            urls = []
            if result.get("data"):
                for item in result["data"]:
                    if item.get("url"):
                        urls.append(item["url"])

            return {"urls": urls, "raw": result}
        except requests.RequestException as e:
            logger.error(f"Qianfan image generation failed: {e}")
            return None

    def generate_with_retry(
        self,
        prompt: str,
        size: str = "1024x1024",
        model: str = "musesteamer-air-image",
        max_retries: int = 3,
    ) -> Optional[dict]:
        """带重试的图片生成。"""
        for attempt in range(max_retries):
            result = self.generate(prompt, size=size, model=model)
            if result and result.get("urls"):
                return result
            wait = 2 ** attempt
            logger.warning(f"Image generation attempt {attempt + 1} failed, retrying in {wait}s...")
            time.sleep(wait)
        return None
