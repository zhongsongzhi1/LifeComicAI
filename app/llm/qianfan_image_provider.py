import asyncio
import hashlib
import logging
import random
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# 千帆文生图 API
IMAGE_GEN_URL = "https://qianfan.baidubce.com/v2/musesteamer/images/generations"

# 图像生成缓存 TTL（秒）- 7天
IMAGE_CACHE_TTL = 7 * 24 * 60 * 60

# RPM 限流保护：每次请求之间的最小间隔（秒）
# 千帆 musesteamer 的 RPM 通常在 1-2，取保守值 5 秒确保不触发限流
MIN_REQUEST_INTERVAL = 5.0


class QianfanImageProvider:
    """百度千帆文生图 Provider（支持结果缓存 + 全局速率锁）。"""

    # 类级别：全局速率锁，确保所有实例共享同一个锁
    _rate_lock = asyncio.Lock()
    _last_request_time = 0.0

    def __init__(self, access_key: str, secret_key: str, cache_service=None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.cache_service = cache_service

    @staticmethod
    def _cache_key(prompt: str, size: str, model: str, seed: int, negative_prompt: str) -> str:
        """生成缓存 key。"""
        raw = f"{model}|{size}|{seed}|{prompt}|{negative_prompt}"
        return f"image_gen:{hashlib.md5(raw.encode('utf-8')).hexdigest()}"

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
        model: str = "qwen-image",
        seed: Optional[int] = None,
        prompt_extend: bool = False,
        negative_prompt: str = "",
        max_retries: int = 4,
    ) -> Optional[dict]:
        """异步生成图片，全局速率锁 + 强制请求间隔 + 指数退避重试。"""
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        # 尝试缓存
        if self.cache_service:
            cache_key = self._cache_key(prompt, size, model, seed, negative_prompt)
            cached = self.cache_service.get(cache_key)
            if cached is not None:
                logger.info(f"Image generation cache hit (seed={seed})")
                return cached

        headers = self._build_headers()
        payload = self._build_payload(prompt, size, model, seed, prompt_extend, negative_prompt)

        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            for attempt in range(max_retries):
                # 全局速率锁：同一时刻只允许一个请求
                async with QianfanImageProvider._rate_lock:
                    # 强制最小请求间隔
                    elapsed = time.time() - QianfanImageProvider._last_request_time
                    if elapsed < MIN_REQUEST_INTERVAL:
                        wait_before = MIN_REQUEST_INTERVAL - elapsed
                        logger.info(f"Rate limiter: waiting {wait_before:.1f}s before next request...")
                        await asyncio.sleep(wait_before)

                    try:
                        logger.info(f"Calling Qianfan image gen: prompt={prompt[:50]}..., size={size}")
                        resp = await client.post(IMAGE_GEN_URL, headers=headers, json=payload)
                        QianfanImageProvider._last_request_time = time.time()
                        logger.info(f"Response status: {resp.status_code}, body: {resp.text[:300]}")
                        resp.raise_for_status()
                        parsed = self._parse_response(resp.text)
                        if parsed and parsed.get("urls"):
                            # 写入缓存
                            if self.cache_service:
                                self.cache_service.set(cache_key, parsed, ttl=IMAGE_CACHE_TTL)
                            return parsed
                        # API 返回成功但无 URL，视为失败
                        logger.warning("Image gen succeeded but no URLs in response")
                        return None
                    except httpx.HTTPStatusError as e:
                        QianfanImageProvider._last_request_time = time.time()
                        logger.error(f"Qianfan image generation failed: {e}")
                        if e.response.status_code == 429:
                            # 429 时重置时间戳，强制下次请求也要等待
                            QianfanImageProvider._last_request_time = time.time()
                        else:
                            return None
                    except (httpx.RequestError, Exception) as e:
                        QianfanImageProvider._last_request_time = time.time()
                        logger.error(f"Qianfan image generation failed: {e}")

                # 在锁外等待，不阻塞其他请求排队
                if attempt == 0:
                    wait = MIN_REQUEST_INTERVAL + random.uniform(1, 3)
                else:
                    wait = min(MIN_REQUEST_INTERVAL * (2 ** attempt) + random.uniform(2, 5), 60)
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
