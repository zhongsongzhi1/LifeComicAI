import asyncio
import logging
import os
import random
from typing import Any, Dict, List, Tuple

from typing import List
import httpx
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.strip_layout_agent import PANEL_H, PANEL_W
from app.db.models import QuickStrip

logger = logging.getLogger(__name__)

# 场景复杂度阈值：prompt 字符数超过此值视为"丰富"，关闭 prompt_extend
PROMPT_RICH_THRESHOLD = 80
MAX_IMAGE_ATTEMPTS = 2
SIMILARITY_THRESHOLD = 0.18

# ===== 宫崎骏水彩风格提示词 =====
GHIBLI_STYLE_FRONT = (
    "杰作级构图，电影级光影，层次丰富的背景细节，"
    "宫崎骏风格，吉卜力动画，手绘水彩插画，"
    "柔和的水彩质感，细腻的笔触，温暖的色彩，"
    "精致的细节，梦幻的氛围，"
)

GHIBLI_STYLE_BACK = (
    "吉卜力工作室画风，动画电影质感，"
    "柔和边缘，水彩晕染效果，色彩通透，"
    "光影柔和，层次丰富，"
    "高质量插画，杰作，最佳质量"
)

# 负面提示词（排除非水彩风格 + 常见缺陷）
GHIBLI_NEGATIVE_PROMPT = (
    "3D渲染，写实照片，真人，赛博朋克，粗黑轮廓线，漫画网点，像素风，"
    "低质量，模糊，变形，丑陋，文字水印，签名，"
    "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
    "deformed, disfigured, ugly, duplicate, mutilated, out of frame, extra limbs, "
    "bad proportions, gross proportions, poorly drawn face, poorly drawn hands, "
    "cloned face, malformed limbs, fused fingers, too many fingers, long neck, "
    "incoherent background, messy background, cluttered, crowded, "
    "noise, grain, scratch, smudge, blur, overexposed, underexposed, "
    "cartoon style, anime style, 2D flat, vector art, simple lines"
)

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
# 保留旧的 NEGATIVE_PROMPT 别名以兼容
NEGATIVE_PROMPT = GHIBLI_NEGATIVE_PROMPT


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
            char_desc = script.get("character_desc", script.get("character_desc", ""))
            panels_script = script.get("panels", [])
            characters = script.get("characters", []) if isinstance(script.get("characters", []), list) else []
            logger.info(f"Strip {strip_id}: generating {len(panels_script)} panels (full parallel)...")
            panel_results = await self._generate_panels(panels_script, char_desc, out_dir, photo_analyses=analyses, characters=characters, photo_paths=photo_paths)
            logger.info(f"Strip {strip_id}: panels generated, composing...")
            compose = await self.layout_agent.run(
                panels=panel_results,
                strip_dir=out_dir,
                photo_paths=photo_paths,
                title=script.get("title", ""),
            )
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

    async def _generate_panels(self, panels: List[Dict[str, Any]], char_desc: str, out_dir: str, photo_analyses: List[Dict[str, Any]] = None, characters: List[Dict[str, Any]] = None, photo_paths: List[str] = None) -> List[Dict[str, Any]]:
        photo_analyses = photo_analyses or []
        characters = characters or []
        photo_paths = photo_paths or []
        shared_seed = random.randint(1, 2**32 - 1)
        logger.info(f"Generating {len(panels)} panels with shared seed={shared_seed}, characters={len(characters)}")
        sem = asyncio.Semaphore(3)

        def _build_panel_prompt(panel: Dict[str, Any], pa: Dict[str, Any], present_chars: List[int], char_desc: str, forbidden: List[str], ref_idx: int) -> str:
            prompt_parts = [GHIBLI_STYLE_FRONT]
            if pa:
                scene_desc = pa.get("scene_desc", "").strip()
                if scene_desc:
                    prompt_parts.append(f"参考照片场景：{scene_desc}")
                if pa.get("location"):
                    prompt_parts.append(f"地点：{pa.get('location')}")
                if pa.get("action"):
                    prompt_parts.append(f"场景动作：{pa.get('action')}")
                if pa.get("emotion"):
                    prompt_parts.append(f"氛围情绪：{pa.get('emotion')}")
                if pa.get("lighting"):
                    prompt_parts.append(f"光线：{pa.get('lighting')}")
                if pa.get("color_tone"):
                    prompt_parts.append(f"整体色调：{pa.get('color_tone')}")
                if pa.get("objects"):
                    prompt_parts.append(f"可见物品：{'、'.join(pa.get('objects', [])[:5])}")
                if pa.get("clothing"):
                    prompt_parts.append(f"服装关键词：{'、'.join(pa.get('clothing', [])[:5])}")
            # 明确约束：只参考一张主照片，禁止使用其他照片的显著元素
            if forbidden:
                prompt_parts.append(f"仅参考照片索引 {ref_idx} 的内容。禁止借鉴或使用其他照片中的以下元素：{'、'.join(forbidden)}")
            if characters and present_chars:
                char_descs = []
                for ci in present_chars:
                    if 0 <= ci < len(characters):
                        c = characters[ci]
                        cdesc = c.get("char_desc", "")
                        if cdesc:
                            char_descs.append(cdesc)
                if char_descs:
                    prompt_parts.append(f"出场人物外观：{'，'.join(char_descs)}")
            elif char_desc:
                prompt_parts.append(f"出场人物外观：{char_desc}")
            if panel.get("description"):
                prompt_parts.append(f"画面描述：{panel.get('description')}")
            if panel.get("dialogue"):
                prompt_parts.append(f"对白内容：{panel.get('dialogue')}")
            prompt_parts.append(SHOT_DESC.get(panel.get("shot_type", "medium"), SHOT_DESC["medium"]))
            prompt_parts.append(LIGHTING_DESC.get(panel.get("lighting", "warm"), LIGHTING_DESC["warm"]))
            prompt_parts.append(COMPOSITION_DESC.get(panel.get("composition", "rule_of_thirds"), COMPOSITION_DESC["rule_of_thirds"]))
            prompt_parts.append(GHIBLI_STYLE_BACK)
            return "，".join([p for p in prompt_parts if p]) + "。"

        def _collect_other_scene_keywords(current_idx: int) -> set:
            current_pa = photo_analyses[current_idx] if current_idx < len(photo_analyses) else None
            current_keywords = set()
            if current_pa:
                for text in [current_pa.get("location", ""), current_pa.get("scene_desc", "")]:
                    for w in text.split("，"):
                        if w.strip():
                            current_keywords.add(w.strip())
                for obj in current_pa.get("objects", []):
                    current_keywords.add(obj)

            other_keywords = set()
            for i, pa in enumerate(photo_analyses):
                if i == current_idx:
                    continue
                for text in [pa.get("location", ""), pa.get("scene_desc", "")]:
                    for w in text.split("，"):
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

        # _build_panel_prompt is defined above with forbidden/ref_idx support; remove duplicate older version

        def _choose_face_info(pa: Dict[str, Any], speaker_id: int, characters: List[Dict[str, Any]]) -> Dict[str, str] | None:
            if not pa or not pa.get("people") or speaker_id < 0 or not characters:
                return None
            if speaker_id >= len(characters):
                return None
            photo_people_idx = characters[speaker_id].get("photo_people_idx", speaker_id)
            people = pa.get("people", [])
            if 0 <= photo_people_idx < len(people):
                return {
                    "face_position": people[photo_people_idx].get("face_position", "center_middle"),
                    "facing": people[photo_people_idx].get("facing", "front"),
                }
            return None

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
                        if scene_desc and (scene_desc.startswith("```") or scene_desc.startswith("{") or '"people_count"' in scene_desc):
                            logger.warning(f"Panel {pn}: scene_desc appears to be invalid JSON, cleaning up")
                            if location:
                                scene_desc = f"{location} 场景"
                            else:
                                scene_desc = "日常生活场景"
                            if location and ("{" in location or "```" in location):
                                location = ""

                    other_scene_keywords = _collect_other_scene_keywords(src_idx)
                    expr_desc = _extract_expression_only(desc)

                    logger.info(f"Panel {pn} (photo#{src_idx}): scene='{scene_desc[:80]}...', expr='{expr_desc[:60]}...'")
                    if other_scene_keywords:
                        logger.info(f"Panel {pn}: negative_scene_keywords={other_scene_keywords}")

                    full_negative = NEGATIVE_PROMPT
                    if other_scene_keywords:
                        full_negative = f"{NEGATIVE_PROMPT}, {'、'.join(other_scene_keywords)}"

                    forbidden_list = list(other_scene_keywords) if other_scene_keywords else []
                    prompt = _build_panel_prompt(
                        panel=p,
                        pa=pa,
                        present_chars=p.get("present_chars", []),
                        char_desc=char_desc,
                        forbidden=forbidden_list,
                        ref_idx=src_idx,
                    )
                    p["face_info"] = _choose_face_info(pa, p.get("speaker_id", 0), characters)

                    has_structured = all(k in p for k in ("shot_type", "composition", "lighting"))
                    has_rich_scene = bool(scene_desc) and len(scene_desc) > 30
                    use_extend = not (has_structured and has_rich_scene)

                    for attempt in range(1, MAX_IMAGE_ATTEMPTS + 1):
                        attempt_seed = shared_seed + attempt
                        try:
                            logger.info(
                                f"Panel {pn} attempt {attempt}: prompt[:120]={prompt[:120]}... extend={use_extend} seed={attempt_seed}"
                            )
                            r = await self.image_provider.generate_async(
                                prompt,
                                size="864x1152",
                                seed=attempt_seed,
                                prompt_extend=use_extend,
                                negative_prompt=full_negative,
                                max_retries=3,
                                reference_image=photo_paths[src_idx] if src_idx < len(photo_paths) else None,
                            )
                            if r and r.get("urls"):
                                lp = os.path.join(out_dir, f"panel_{pn}.png")
                                ok = await self._download_async(client, r["urls"][0], lp)
                                if ok and self._is_valid_image(lp):
                                    # 语义相似度检测（基于颜色直方图）
                                    similar = True
                                    try:
                                        if pa and src_idx < len(photo_paths):
                                            ref_path = photo_paths[src_idx]
                                            sim = self._histogram_similarity(ref_path, lp)
                                            logger.info(f"Panel {pn}: histogram similarity to reference={sim:.3f}")
                                            similar = sim >= SIMILARITY_THRESHOLD
                                    except Exception as e:
                                        logger.warning(f"Panel {pn}: similarity check failed: {e}")

                                    if similar:
                                        p["image_path"] = lp
                                        p["generation_attempts"] = attempt
                                        logger.info(f"Panel {pn}: generated successfully on attempt {attempt}")
                                        return p
                                    else:
                                        logger.warning(f"Panel {pn}: low similarity ({sim:.3f}), will retry if attempts remain")
                                else:
                                    logger.warning(f"Panel {pn}: generated file invalid or download failed on attempt {attempt}")
                        except Exception as e:
                            logger.warning(f"Panel {pn} attempt {attempt} failed: {e}", exc_info=True)

                    lp = os.path.join(out_dir, f"panel_{pn}.png")
                    self._placeholder(lp)
                    p["image_path"] = lp
                    p["generation_attempts"] = MAX_IMAGE_ATTEMPTS
                    return p
            return await asyncio.gather(*[gen_one(dict(p)) for p in panels])

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
                width, height = img.size
                return width >= 200 and height >= 200
        except Exception as e:
            logger.warning(f"Invalid image file {path}: {e}")
            return False

    @staticmethod
    def _histogram_similarity(path_a: str, path_b: str) -> float:
        try:
            a = Image.open(path_a).convert('RGB').resize((256, 256), Image.LANCZOS)
            b = Image.open(path_b).convert('RGB').resize((256, 256), Image.LANCZOS)
            ha = a.histogram()
            hb = b.histogram()
            # cosine similarity
            import math
            dot = 0.0
            lena = len(ha)
            for i in range(lena):
                dot += ha[i] * hb[i]
            suma = sum(x * x for x in ha)
            sumb = sum(x * x for x in hb)
            if suma == 0 or sumb == 0:
                return 0.0
            return dot / (math.sqrt(suma) * math.sqrt(sumb))
        except Exception as e:
            logger.warning(f"Histogram similarity failed: {e}")
            return 0.0

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
