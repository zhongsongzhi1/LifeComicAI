import asyncio
import hashlib
import logging
import random
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ModelScope 文生图 API
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 图像生成缓存 TTL（秒）- 7天
IMAGE_CACHE_TTL = 7 * 24 * 60 * 60

# 轮询间隔（秒）
POLL_INTERVAL = 2

# 最大轮询次数
MAX_POLL_ATTEMPTS = 60  # 3 分钟


class ModelScopeImageProvider:
    """ModelScope 文生图 Provider（支持结果缓存 + 速率锁）。

    与 QianfanImageProvider 保持完全相同的接口契约：
    - generate_async(prompt, size, model, seed, prompt_extend, negative_prompt, max_retries) → {"urls": [...], "raw": ...}
    - generate(...)
    - generate_with_retry(...)
    """

    # 类级别：全局速率锁
    _rate_lock = asyncio.Lock()
    _last_request_time = 0.0
    MIN_REQUEST_INTERVAL = 1.0  # ModelScope 限流较宽松

    def __init__(self, access_key: str, secret_key: str = "", cache_service=None):
        """兼容旧接口：access_key = ModelScope API Key, secret_key 保留但忽略。"""
        self.api_key = access_key
        self.cache_service = cache_service

    @staticmethod
    def _cache_key(prompt: str, size: str, model: str, seed: int, negative_prompt: str) -> str:
        raw = f"{model}|{size}|{seed}|{prompt}|{negative_prompt}"
        return f"image_gen:{hashlib.md5(raw.encode('utf-8')).hexdigest()}"

    def _build_headers(self, async_mode: bool = True) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if async_mode:
            headers["X-ModelScope-Async-Mode"] = "true"
        return headers

    @staticmethod
    def _build_payload(prompt: str, size: str, model: str, seed: int, negative_prompt: str, reference_image: Optional[str] = None) -> dict:
        """构建请求体，支持参考图机制。"""
        payload = {
            "model": model,
            "prompt": prompt,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        # ModelScope Z-Image 支持 width/height 或 size
        if size and "x" in size:
            parts = size.split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                payload["width"] = int(parts[0])
                payload["height"] = int(parts[1])
        # 参考图：支持本地文件路径或 base64
        if reference_image:
            if os.path.exists(reference_image):
                import base64
                with open(reference_image, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                payload["reference_image"] = img_b64
            elif reference_image.startswith("data:image/"):
                payload["reference_image"] = reference_image.split(",", 1)[1]
            else:
                payload["reference_image"] = reference_image
        return payload

    def _parse_poll_response(self, data: dict) -> Optional[dict]:
        """解析轮询结果，返回与千帆相同格式的 dict。"""
        urls = []
        if data.get("output_images"):
            for url in data["output_images"]:
                if url:
                    urls.append(url)
        if urls:
            return {"urls": urls, "raw": data}
        return None

    async def generate_async(
        self,
        prompt: str,
        size: str = "864x1152",
        model: str = "Tongyi-MAI/Z-Image-Turbo",
        seed: Optional[int] = None,
        prompt_extend: bool = False,
        negative_prompt: str = "",
        max_retries: int = 3,
        reference_image: Optional[str] = None,
    ) -> Optional[dict]:
        """异步生成图片，使用 ModelScope 异步任务模式。"""
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        # 尝试缓存
        if self.cache_service:
            cache_key = self._cache_key(prompt, size, model, seed, negative_prompt)
            cached = self.cache_service.get(cache_key)
            if cached is not None:
                logger.info(f"ModelScope image generation cache hit (seed={seed})")
                return cached

        headers = self._build_headers(async_mode=True)
        payload = self._build_payload(prompt, size, model, seed, negative_prompt, reference_image)

        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            for attempt in range(max_retries):
                async with ModelScopeImageProvider._rate_lock:
                    elapsed = time.time() - ModelScopeImageProvider._last_request_time
                    if elapsed < ModelScopeImageProvider.MIN_REQUEST_INTERVAL:
                        wait_before = ModelScopeImageProvider.MIN_REQUEST_INTERVAL - elapsed
                        logger.info(f"Rate limiter: waiting {wait_before:.1f}s...")
                        await asyncio.sleep(wait_before)

                    try:
                        # 1. 创建异步任务
                        logger.info(f"ModelScope: creating task, prompt={prompt[:80]}...")
                        resp = await client.post(
                            f"{MODELSCOPE_BASE_URL}/images/generations",
                            headers=headers,
                            json=payload,
                        )
                        ModelScopeImageProvider._last_request_time = time.time()
                        logger.info(f"Task creation status: {resp.status_code}, body: {resp.text[:300]}")
                        resp.raise_for_status()

                        task_data = resp.json()
                        task_id = task_data.get("task_id")
                        if not task_id:
                            logger.error(f"No task_id in response: {task_data}")
                            return None

                        logger.info(f"Task created: {task_id}")

                        # 2. 轮询等待完成
                        poll_headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "X-ModelScope-Task-Type": "image_generation",
                        }
                        for poll_attempt in range(MAX_POLL_ATTEMPTS):
                            await asyncio.sleep(POLL_INTERVAL)
                            poll_resp = await client.get(
                                f"{MODELSCOPE_BASE_URL}/tasks/{task_id}",
                                headers=poll_headers,
                            )
                            poll_resp.raise_for_status()
                            poll_data = poll_resp.json()

                            status = poll_data.get("task_status", "")
                            if status == "SUCCEED":
                                result = self._parse_poll_response(poll_data)
                                if result and result.get("urls"):
                                    if self.cache_service:
                                        self.cache_service.set(cache_key, result, ttl=IMAGE_CACHE_TTL)
                                    logger.info(f"ModelScope generation succeeded: {len(result['urls'])} images")
                                    return result
                                logger.warning("Task SUCCEED but no output_images")
                                return None
                            elif status == "FAILED":
                                error_msg = poll_data.get("error", poll_data.get("error_msg", "unknown"))
                                logger.error(f"Task failed: {error_msg}")
                                break
                            elif status in ("RUNNING", "PENDING", "WAITING"):
                                if poll_attempt % 5 == 0:
                                    logger.info(f"Task {task_id}: still {status} ({poll_attempt + 1}/{MAX_POLL_ATTEMPTS})")
                                continue
                            else:
                                logger.warning(f"Unknown task status: {status}")

                        # 轮询超时或失败
                        logger.error(f"Task {task_id}: polling exhausted or failed")

                    except httpx.HTTPStatusError as e:
                        ModelScopeImageProvider._last_request_time = time.time()
                        logger.error(f"ModelScope generation failed: {e}")
                        if e.response.status_code == 429:
                            ModelScopeImageProvider._last_request_time = time.time()
                        else:
                            return None
                    except (httpx.RequestError, Exception) as e:
                        ModelScopeImageProvider._last_request_time = time.time()
                        logger.error(f"ModelScope generation failed: {e}")

                # 重试等待
                if attempt < max_retries - 1:
                    wait = ModelScopeImageProvider.MIN_REQUEST_INTERVAL * (2 ** attempt) + random.uniform(2, 5)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)

        return None

    def generate(
        self,
        prompt: str,
        size: str = "864x1152",
        model: str = "Tongyi-MAI/Z-Image-Turbo",
        seed: Optional[int] = None,
        prompt_extend: bool = False,
        negative_prompt: str = "",
        max_retries: int = 3,
        reference_image: Optional[str] = None,
    ) -> Optional[dict]:
        """同步兼容方法。"""
        try:
            return asyncio.run(
                self.generate_async(
                    prompt, size=size, model=model, seed=seed,
                    prompt_extend=prompt_extend, negative_prompt=negative_prompt,
                    max_retries=max_retries, reference_image=reference_image,
                )
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.generate_async(
                        prompt, size=size, model=model, seed=seed,
                        prompt_extend=prompt_extend, negative_prompt=negative_prompt,
                        max_retries=max_retries, reference_image=reference_image,
                    )
                )
            finally:
                loop.close()

    def generate_with_retry(
        self,
        prompt: str,
        size: str = "864x1152",
        max_retries: int = 3,
        **kwargs,
    ) -> Optional[dict]:
        """兼容旧接口。"""
        return self.generate(prompt, size=size, max_retries=max_retries, **kwargs)
