import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PromptEnhancer:
    """多维提示词增强引擎（工具模块）

    功能: 根据多个维度（镜头语言、构图、光影、质量）生成高质量的AI绘画提示词
    """

    SHOT_DESC = {
        "Wide": "广角全景，鸟瞰或平视，展现完整环境和场景氛围，人物在画面中较小",
        "Medium": "中景镜头，人物腰部以上，突出上半身动作和面部表情",
        "Close-up": "面部特写镜头，聚焦人物眼睛和表情变化，背景完全虚化",
        "Action": "动作镜头，强调动态张力和流畅性，人物肢体动作清晰",
        "Ending": "结尾镜头，强调情感和视觉冲击，画面充满张力"
    }

    COMPOSITION_DESC = {
        "rule_of_thirds": "三分法构图，主体位于横竖三分线交叉点",
        "center": "中心对称构图，主体居中，画面稳定有力",
        "diagonal": "对角线构图，主体沿对角线分布，充满动感",
        "framing": "框架式构图，利用前景元素框住主体，增加层次感",
        "leading_lines": "引导线构图，利用线条将视线导向主体"
    }

    LIGHTING_DESC = {
        "natural": "自然日光，明亮柔和，色彩还原真实，有清晰的方向性阴影",
        "warm": "暖色温光，金黄或橙色调，温馨舒适，适合室内和傍晚",
        "cool": "冷色温光，蓝白或青色调，清新安静，适合夜晚或室内荧光灯",
        "dramatic": "戏剧性侧光或顶光，强烈明暗对比，高反差，营造情绪张力",
        "soft": "柔光扩散，无明确阴影，整体柔和淡雅，梦幻朦胧"
    }

    NEGATIVE_PROMPT = (
        "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
        "deformed, disfigured, ugly, duplicate, mutilated, out of frame, extra limbs, "
        "bad proportions, gross proportions, poorly drawn face, poorly drawn hands, "
        "cloned face, malformed limbs, fused fingers, too many fingers, long neck, "
        "incoherent background"
    )

    def enhance_prompt(
        self,
        style_name: str,
        desc: str,
        dialogue: str = "",
        shot_type: str = "Medium",
        characters_detail: str = "",
        composition: str = "rule_of_thirds",
        lighting: str = "natural",
        quality_hints: str = "masterpiece, best quality, highres"
    ) -> Dict[str, str]:
        parts = []
        parts.append(f"{style_name}，{quality_hints}")
        shot_desc = self.SHOT_DESC.get(shot_type, self.SHOT_DESC["Medium"])
        parts.append(shot_desc)
        comp_desc = self.COMPOSITION_DESC.get(composition, self.COMPOSITION_DESC["rule_of_thirds"])
        parts.append(comp_desc)
        light_desc = self.LIGHTING_DESC.get(lighting, self.LIGHTING_DESC["natural"])
        parts.append(light_desc)
        if desc:
            parts.append(desc)
        if characters_detail:
            parts.append(f"人物特征：{characters_detail}")
        if dialogue:
            parts.append(f"对话/氛围：{dialogue}")

        positive_prompt = "，".join(filter(None, parts))
        quality_score = len([p for p in parts if p])

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": self.NEGATIVE_PROMPT,
            "quality_score": quality_score
        }

    def get_shot_suggestions(self) -> Dict[str, str]:
        return self.SHOT_DESC.copy()

    def get_composition_suggestions(self) -> Dict[str, str]:
        return self.COMPOSITION_DESC.copy()

    def get_lighting_suggestions(self) -> Dict[str, str]:
        return self.LIGHTING_DESC.copy()
