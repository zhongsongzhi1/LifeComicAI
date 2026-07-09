from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PromptVersion

PRESETS = [
    {
        "style_key": "miyazaki",
        "style_name": "宫崎骏风",
        "mood": "温馨",
        "dialogue_level": "少",
        "cinematic": 1,
        "prompt_template": (
            "你是一位漫画导演，请将以下分镜脚本调整为「宫崎骏风格」。\n\n"
            "核心要求：\n"
            "- 画面色调：柔和、温暖、自然光\n"
            "- 氛围：温馨、治愈、带一点怀旧\n"
            "- 对白：尽量少，用画面和表情传达情绪\n"
            "- 细节：注意天空、风、花草等自然元素的描绘\n"
            "- 镜头：偏好中远景和全景，少用特写\n\n"
            "原始分镜：\n{storyboard_json}\n\n"
            "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
        ),
    },
    {
        "style_key": "cinematic",
        "style_name": "电影感",
        "mood": "中性",
        "dialogue_level": "中",
        "cinematic": 1,
        "prompt_template": (
            "你是一位漫画导演，请将以下分镜脚本调整为「电影感风格」。\n\n"
            "核心要求：\n"
            "- 画面色调：电影级调色，强调光影对比\n"
            "- 氛围：叙事感强，有戏剧张力\n"
            "- 对白：自然流畅，富有节奏感\n"
            "- 细节：注意构图和景深，营造电影画面感\n"
            "- 镜头：灵活运用各种镜头语言\n\n"
            "原始分镜：\n{storyboard_json}\n\n"
            "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
        ),
    },
    {
        "style_key": "slice",
        "style_name": "日常风",
        "mood": "轻松",
        "dialogue_level": "多",
        "cinematic": 0,
        "prompt_template": (
            "你是一位漫画导演，请将以下分镜脚本调整为「日常风格」。\n\n"
            "核心要求：\n"
            "- 画面色调：明亮、清新、生活化\n"
            "- 氛围：轻松、愉快、日常感\n"
            "- 对白：丰富自然，贴近日常对话\n"
            "- 细节：注重生活场景和人物互动\n"
            "- 镜头：多用中景和近景，突出人物表情\n\n"
            "原始分镜：\n{storyboard_json}\n\n"
            "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
        ),
    },
    {
        "style_key": "manga",
        "style_name": "日系漫画",
        "mood": "热血",
        "dialogue_level": "中",
        "cinematic": 1,
        "prompt_template": (
            "你是一位漫画导演，请将以下分镜脚本调整为「日系漫画风格」。\n\n"
            "核心要求：\n"
            "- 画面色调：高对比度，富有冲击力\n"
            "- 氛围：热血、燃、有速度感\n"
            "- 对白：简洁有力，富有感染力\n"
            "- 细节：注重动作场面和特效线\n"
            "- 镜头：多用特写和动态镜头\n\n"
            "原始分镜：\n{storyboard_json}\n\n"
            "请输出调整后的完整分镜脚本，保持 JSON 格式不变。"
        ),
    },
]


async def seed_prompt_versions(session: AsyncSession):
    result = await session.execute(select(PromptVersion.style_key))
    existing_keys = {row[0] for row in result.fetchall()}

    for preset in PRESETS:
        if preset["style_key"] not in existing_keys:
            pv = PromptVersion(
                agent_name="director",
                style_key=preset["style_key"],
                style_name=preset["style_name"],
                mood=preset["mood"],
                dialogue_level=preset["dialogue_level"],
                cinematic=preset["cinematic"],
                prompt_template=preset["prompt_template"],
                version=1,
                is_active=1,
            )
            session.add(pv)

    await session.commit()
