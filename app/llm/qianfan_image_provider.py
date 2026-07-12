import asyncio
import logging
import random
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 千帆文生图 API
IMAGE_GEN_URL = "https://qianfan.baidubce.com/v2/musesteamer/images/generations"


class QianfanImageProvider:
    """百度千帆文生图 Provider。"""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def _build_headers(self) -> dict:
        ak = self.access_key if self.access_key.startswith("ALTAK-") else f"ALTAK-{self.access_key}"
        return {
            "Authorization": f"Bearer bce-v3/{ak}/{self.secret_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, prompt: str, size: str, model: str, seed: int, prompt_extend: bool, negative_prompt: str) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seed": seed,
            "prompt_extend": prompt_extend,
            "response_format": "url",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return payload

    def _parse_response(self, resp_body: str) -> Optional[dict]:
        import json
        try:
            result = json.loads(resp_body)
            urls = []
            if result.get("data"):
                for item in result["data"]:
                    if item.get("url"):
                        urls.append(item["url"])
            return {"urls": urls, "raw": result}
        except (json.JSONDecodeError, KeyError):
            return None

    async def generate_async(
        self,
        prompt: str,
        size: str = "864x1152",
        model: str = "musesteamer-air-image",
        seed: Optional[int] = None,
        prompt_extend: bool = False,
        negative_prompt: str = "",
        max_retries: int = 4,
    ) -> Optional[dict]:
        """异步生成图片，带指数退避重试。"""
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        headers = self._build_headers()
        payload = self._build_payload(prompt, size, model, seed, prompt_extend, negative_prompt)

        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Calling Qianfan image gen: prompt={prompt[:50]}..., size={size}")
                    resp = await client.post(IMAGE_GEN_URL, headers=headers, json=payload)
                    logger.info(f"Response status: {resp.status_code}, body: {resp.text[:300]}")
                    resp.raise_for_status()
                    result = self._parse_response(resp.text)
                    if result and result.get("urls"):
                        return result
                except httpx.HTTPStatusError as e:
                    logger.error(f"Qianfan image generation failed: {e}")
                    if e.response.status_code != 429:
                        return None
                except (httpx.RequestError, Exception) as e:
                    logger.error(f"Qianfan image generation failed: {e}")

                if attempt == 0:
                    wait = 3 + random.uniform(0, 2)
                else:
                    wait = min(5 * (2 ** attempt) + random.uniform(0, 3), 30)
                logger.warning(f"Image generation attempt {attempt + 1} failed, retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)

        return None

    def generate(self, prompt: str, size: str = "864x1152", model: str = "musesteamer-air-image", seed: Optional[int] = None, prompt_extend: bool = False, negative_prompt: str = "", max_retries: int = 4) -> Optional[dict]:
        """同步兼容方法：调用异步生成并返回结果。"""
        try:
            return asyncio.run(self.generate_async(prompt, size=size, model=model, seed=seed, prompt_extend=prompt_extend, negative_prompt=negative_prompt, max_retries=max_retries))
        except RuntimeError:
            # 如果已有运行中的事件循环（例如在异步测试环境中），创建临时任务
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.generate_async(prompt, size=size, model=model, seed=seed, prompt_extend=prompt_extend, negative_prompt=negative_prompt, max_retries=max_retries))
            finally:
                loop.close()

    def generate_with_retry(self, prompt: str, size: str = "864x1152", max_retries: int = 3, **kwargs) -> Optional[dict]:
        """兼容旧接口，增加最大重试参数并调用 generate。"""
        return self.generate(prompt, size=size, max_retries=max_retries, **kwargs)
