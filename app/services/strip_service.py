import asyncio
import logging
import os
import random
from typing import Any, Dict, List

import httpx
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.strip_layout_agent import PANEL_H, PANEL_W
from app.db.models import QuickStrip

logger = logging.getLogger(__name__)

# 场景复杂度阈值：prompt 字符数超过此值视为"丰富"，关闭 prompt_extend
PROMPT_RICH_THRESHOLD = 80

# 镜头语言映射（电影化描述，用于AI绘画提示词）
SHOT_DESC = {
    "establishing": "广角远景，鸟瞰或平视，展现完整环境和场景氛围，人物在画面中较小或仅作为环境的一部分",
    "full_shot": "全身镜头，人物从头到脚完整呈现，展示全身姿态和服装，环境作为背景衬托",
    "medium": "中景镜头，人物腰部以上，突出上半身动作和面部表情，环境适度虚化",
    "close_up": "面部特写镜头，聚焦人物眼睛和表情变化，背景完全虚化，强调情绪张力",
    "detail": "微距细节特写，聚焦手部动作、物品细节或局部特征，极浅景深",
}
# 构图方式映射
COMPOSITION_DESC = {
    "rule_of_thirds": "三分法构图，主体位于画面横竖三分线交叉点，视觉平衡自然",
    "center": "中心对称构图，主体居中，画面稳定有力，适合强调人物",
    "diagonal": "对角线构图，主体沿对角线分布，画面充满动感和张力",
    "framing": "框架式构图，利用门窗、树枝等前景元素框住主体，增加层次感",
    "leading_lines": "引导线构图，利用道路、栏杆等线条将视线导向主体",
}
# 光影氛围映射
LIGHTING_DESC = {
    "natural": "自然日光，明亮柔和，色彩还原真实，有清晰的方向性阴影",
    "warm": "暖色温光，金黄或橙色调，温馨舒适，适合室内和傍晚",
    "cool": "冷色温光，蓝白或青色调，清新安静，适合夜晚或室内荧光灯",
    "dramatic": "戏剧性侧光或顶光，强烈明暗对比，高反差，营造情绪张力",
    "soft": "柔光扩散，无明确阴影，整体柔和淡雅，梦幻朦胧",
}
# 负面提示词
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, disfigured, ugly, duplicate, mutilated, out of frame, extra limbs, bad proportions, gross proportions, poorly drawn face, poorly drawn hands, cloned face, malformed limbs, fused fingers, too many fingers, long neck, incoherent background"


def _panel_count_for(num_photos: int) -> int:
    if num_photos <= 2:
        return 4
    elif num_photos <= 3:
        return 5
    else:
        return 6


class StripService:
    def __init__(self, vision_agent: BaseAgent, script_agent: BaseAgent, image_provider, layout_agent: BaseAgent, comics_dir: str):
        self.vision_agent = vision_agent
        self.script_agent = script_agent
        self.image_provider = image_provider
        self.layout_agent = layout_agent
        self.comics_dir = comics_dir

    async def generate_strip(self, photo_paths: List[str], session: AsyncSession) -> Dict[str, Any]:
        if not (2 <= len(photo_paths) <= 6):
            raise ValueError("请上传2-6张照片")
        strip = QuickStrip(status="processing")
        session.add(strip)
        await session.flush()
        strip_id = strip.id
        out_dir = os.path.join(self.comics_dir, "strips", str(strip_id))
        os.makedirs(out_dir, exist_ok=True)
        target_panels = _panel_count_for(len(photo_paths))
        try:
            logger.info(f"Strip {strip_id}: analyzing {len(photo_paths)} photos, target panels={target_panels}")
            analyses = await self._analyze_photos(photo_paths)
            logger.info(f"Strip {strip_id}: analysis done")
            script = await self.script_agent.run(photo_analyses=analyses, num_panels=target_panels)
            logger.info(f"Strip {strip_id}: script done, title={script.get('title')}, panels={len(script.get('panels', []))}")
            strip.title = script.get("title", "生活条漫")
            char_desc = script.get("character_desc", "")
            panels_script = script.get("panels", [])[:target_panels]
            logger.info(f"Strip {strip_id}: generating {len(panels_script)} panels (full parallel)...")
            panel_results = await self._generate_panels(panels_script, char_desc, out_dir, photo_analyses=analyses)
            logger.info(f"Strip {strip_id}: panels generated, composing...")
            compose = await self.layout_agent.run(panels=panel_results, strip_dir=out_dir)
            logger.info(f"Strip {strip_id}: compose done -> {compose['image_path']}")
            strip.status = "completed"
            strip.image_path = compose["image_path"]
            await session.commit()
            await session.refresh(strip)
            return {
                "strip_id": strip_id, "title": strip.title, "status": "completed",
                "image_path": compose["image_path"],
                "panels": [{"panel_num": p["panel_num"], "dialogue": p.get("dialogue", ""), "image_path": p.get("image_path", "")} for p in panel_results],
            }
        except Exception as e:
            logger.error(f"Strip {strip_id} failed: {e}", exc_info=True)
            strip.status = "failed"
            strip.error_msg = str(e)[:500]
            await session.commit()
            raise

    async def _analyze_photos(self, photo_paths: List[str]) -> List[Dict[str, Any]]:
        tasks = [self.vision_agent.run_async(photo_id=i, image_path=p) for i, p in enumerate(photo_paths)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Photo {i} analysis failed: {r}")
                out.append({"scene_desc": "生活场景", "people": [], "action": "", "emotion": "", "location": "", "clothing": [], "objects": []})
            else:
                out.append(r)
        return out

    async def _generate_panels(self, panels: List[Dict[str, Any]], char_desc: str, out_dir: str, photo_analyses: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        photo_analyses = photo_analyses or []
        shared_seed = random.randint(1, 2**32 - 1)
        logger.info(f"Generating {len(panels)} panels with shared seed={shared_seed}")
        sem = asyncio.Semaphore(3)
        
        def _collect_other_scene_keywords(current_idx: int) -> set:
            current_pa = photo_analyses[current_idx] if current_idx < len(photo_analyses) else None
            current_keywords = set()
            if current_pa:
                if current_pa.get("location"):
                    for w in current_pa["location"].split("，"):
                        if w.strip():
                            current_keywords.add(w.strip())
                for obj in current_pa.get("objects", []):
                    current_keywords.add(obj)
            
            other_keywords = set()
            for i, pa in enumerate(photo_analyses):
                if i == current_idx:
                    continue
                if pa.get("location"):
                    for w in pa["location"].split("，"):
                        if w.strip():
                            other_keywords.add(w.strip())
                for obj in pa.get("objects", []):
                    other_keywords.add(obj)
            
            return other_keywords - current_keywords
        
        def _extract_expression_only(desc: str) -> str:
            if not desc:
                return ""
            expr_keywords = ["微笑", "笑", "开心", "专注", "期待", "惊讶", "疑惑", "温柔", "满足", "疲惫", "兴奋", "害羞", "坚定", "轻松", "惬意", "眼神", "嘴角", "脸颊", "表情", "神情", "眼睛"]
            sentences = desc.split("，")
            expr_parts = []
            for s in sentences:
                if any(kw in s for kw in expr_keywords) and not any(obj_kw in s for obj_kw in ["碗", "筷", "勺", "盘", "食物", "饭", "面", "菜", "汤", "杯", "瓶", "饮料", "书", "手机", "包", "桌子", "椅子", "沙发", "床"]):
                    expr_parts.append(s)
            return "，".join(expr_parts)

        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            async def gen_one(p):
                async with sem:
                    pn = p["panel_num"]
                    desc = p.get("description", "")
                    dialogue = p.get("dialogue", "")
                    src_idx = p.get("source_photo_idx", 0)
                    if src_idx >= len(photo_analyses):
                        src_idx = min(src_idx % len(photo_analyses), len(photo_analyses) - 1)
                    pa = photo_analyses[src_idx] if src_idx < len(photo_analyses) else None
                    
                    scene_desc = ""
                    location = ""
                    if pa:
                        location = pa.get("location", "")
                        scene_desc = pa.get("scene_desc", "")
                    
                    other_scene_keywords = _collect_other_scene_keywords(src_idx)
                    expr_desc = _extract_expression_only(desc)
                    
                    logger.info(f"Panel {pn} (photo#{src_idx}): scene='{scene_desc[:80]}...', expr='{expr_desc[:60]}...'")
                    if other_scene_keywords:
                        logger.info(f"Panel {pn}: negative_scene_keywords={other_scene_keywords}")

                    shot_type = p.get("shot_type", "medium")
                    composition = p.get("composition", "rule_of_thirds")
                    lighting = p.get("lighting", "warm")
                    shot_desc = SHOT_DESC.get(shot_type, SHOT_DESC["medium"])
                    comp_desc = COMPOSITION_DESC.get(composition, COMPOSITION_DESC["rule_of_thirds"])
                    light_desc = LIGHTING_DESC.get(lighting, LIGHTING_DESC["warm"])

                    full_negative = NEGATIVE_PROMPT
                    if other_scene_keywords:
                        full_negative = f"{NEGATIVE_PROMPT}, {'、'.join(other_scene_keywords)}"

                    prompt = (
                        f"场景：{scene_desc}，地点：{location}。"
                        f"{shot_desc}。"
                        f"{char_desc}。"
                    )
                    if expr_desc:
                        prompt += f"人物表情：{expr_desc}。"
                    prompt += (
                        f"{light_desc}。"
                        f"{comp_desc}。"
                        f"漫画风格，日系条漫，粗黑线稿，半色调网点，高质量竖版插画。"
                    )
                    # 按场景复杂度决定是否开启 prompt_extend
                    has_structured = all(k in p for k in ("shot_type", "composition", "lighting"))
                    has_rich_scene = bool(scene_desc) and len(scene_desc) > 30
                    use_extend = not (has_structured and has_rich_scene)
                    
                    try:
                        logger.info(f"Panel {pn} (photo#{src_idx}): extend={use_extend}, prompt[:150]={prompt[:150]}...")
                        r = await self.image_provider.generate_async(prompt, size="864x1152", seed=shared_seed, prompt_extend=use_extend, negative_prompt=full_negative)
                        if r and r.get("urls"):
                            lp = os.path.join(out_dir, f"panel_{pn}.png")
                            ok = await self._download_async(client, r["urls"][0], lp)
                            logger.info(f"Panel {pn}: ok={ok}")
                            if ok:
                                p["image_path"] = lp
                                return p
                    except Exception as e:
                        logger.warning(f"Panel {pn} failed: {e}", exc_info=True)
                    lp = os.path.join(out_dir, f"panel_{pn}.png")
                    self._placeholder(lp)
                    p["image_path"] = lp
                    return p
            return await asyncio.gather(*[gen_one(dict(p)) for p in panels])

    @staticmethod
    async def _download_async(client: httpx.AsyncClient, url: str, path: str) -> bool:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            return False

    @staticmethod
    def _placeholder(path: str):
        Image.new("RGB", (PANEL_W, PANEL_H), (245, 245, 245)).save(path, "PNG")
