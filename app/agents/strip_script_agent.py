import asyncio
import logging
from typing import Any, Dict, List

from app.agents.base import BaseAgent

# 默认镜头模版（按场景类型动态选择）
SHOT_TEMPLATES = {
    "eating": {
        "shots": ["establishing", "medium", "full_shot", "close_up", "detail", "medium"],
        "compositions": ["leading_lines", "center", "rule_of_thirds", "center", "framing", "diagonal"],
        "lightings": ["natural", "warm", "warm", "soft", "warm", "warm"],
    },
    "working": {
        "shots": ["establishing", "medium", "medium", "close_up", "full_shot", "medium"],
        "compositions": ["leading_lines", "rule_of_thirds", "center", "center", "diagonal", "rule_of_thirds"],
        "lightings": ["natural", "cool", "cool", "dramatic", "natural", "warm"],
    },
    "outdoor": {
        "shots": ["establishing", "full_shot", "medium", "close_up", "medium", "full_shot"],
        "compositions": ["leading_lines", "rule_of_thirds", "diagonal", "center", "framing", "rule_of_thirds"],
        "lightings": ["natural", "natural", "warm", "soft", "dramatic", "warm"],
    },
    "relaxing": {
        "shots": ["establishing", "medium", "full_shot", "close_up", "medium", "medium"],
        "compositions": ["leading_lines", "center", "rule_of_thirds", "center", "diagonal", "framing"],
        "lightings": ["soft", "warm", "warm", "soft", "warm", "warm"],
    },
    "general": {
        "shots": ["establishing", "medium", "medium", "close_up", "medium", "medium"],
        "compositions": ["leading_lines", "rule_of_thirds", "diagonal", "center", "framing", "rule_of_thirds"],
        "lightings": ["natural", "warm", "warm", "soft", "dramatic", "warm"],
    },
}

logger = logging.getLogger(__name__)

STRIP_SCRIPT_SYSTEM_PROMPT = """你是专业漫画分镜师。基于照片分析创作{num_panels}格漫画脚本（竖版长图）。

=== 核心规则 ===
1. 严格贴合照片：每格画面描述和对话必须基于对应照片的真实场景、人物动作、表情情绪，禁止编造与照片不符的内容
2. 对话必须符合场景：如果照片中人物在吃饭，对话必须与吃饭相关；如果在工作，对话必须与工作相关
3. characters数组：【重要】所有出场人物的纯外貌特征描述，必须从照片分析的 people 数组中提取，保持一致
   - characters[].name：人物代号（如"小明""小红""眼镜男""长发女生"），用于指代
   - characters[].char_desc：【仅】包含该人物纯外貌特征（性别年龄、发型发色、五官特征、皮肤肤色、体型体态、服装颜色款式、配饰），【绝对禁止】包含任何场景、地点、动作、食物、物品、表情、情绪描述！
   - characters[].photo_people_idx：对应照片分析中 people 数组的索引（0起），用于关联人脸位置
4. description（50字内）：画面描述，重点写人物动作和表情，不包含场景（场景由source_photo_idx自动关联）
5. dialogue（20字内）：必须贴合当前格的场景和动作，幽默/温馨/生活化，有梗
6. source_photo_idx：每格参考哪张照片（0起），确保场景元素来自该照片
7. speaker_id：当前格说话的人物索引（0起），对应 characters 数组的索引，如果是旁白填 -1
8. present_chars：当前格出场的人物索引数组（0起），对应 characters 数组
9. bubble_type：气泡类型，根据对话内容选择
   - dialogue：普通对话（默认）
   - thought：内心想法/思考
   - narration：旁白/解说
   - shout：大喊/惊叫/强调
10. emotion：说话人的情绪，用于选择气泡位置和样式
    - 可选项：happy/sad/angry/surprised/neutral/excited/thoughtful

=== 镜头语言规则（重要！专业漫画必须镜头多样化） ===
11. shot_type：{num_panels}格必须使用至少3种不同景别，禁止全部使用medium！
    - 可选项：establishing/full_shot/medium/close_up/detail
    - 第1格必须用establishing交代环境，倒数第2或第3格必须用close_up制造情绪高潮
    - 推荐分布：{num_panels}格 = 1个establishing + 1个close_up + 其余用medium和full_shot交替
12. composition：每格用不同构图，禁止全用rule_of_thirds
    - 可选项：rule_of_thirds/center/diagonal/framing/leading_lines
    - 推荐：establishing用leading_lines，close_up用center，medium用rule_of_thirds或diagonal
13. lighting：根据场景氛围选光影，不要全用warm
    - 可选项：natural/warm/cool/dramatic/soft
    - 推荐：白天户外用natural，室内用餐用warm，夜晚用cool，情绪转折用dramatic
14. 叙事节奏：establishing(铺垫)→medium(发展)→close_up(转折/笑点)→medium(收尾)

输出纯JSON：
{{
  "title":"标题(10字)",
  "characters":[
    {{"name":"人物1","char_desc":"外貌描述","photo_people_idx":0}}
  ],
  "panels":[
    {{
      "panel_num":1,
      "source_photo_idx":0,
      "shot_type":"establishing",
      "composition":"leading_lines",
      "lighting":"natural",
      "description":"画面描述",
      "dialogue":"对白",
      "speaker_id":0,
      "present_chars":[0,1],
      "bubble_type":"dialogue",
      "emotion":"happy"
    }}
  ]
}}"""


VALID_SHOT_TYPES = {"establishing", "full_shot", "medium", "close_up", "detail"}
VALID_COMPOSITIONS = {"rule_of_thirds", "center", "diagonal", "framing", "leading_lines"}
VALID_LIGHTINGS = {"natural", "warm", "cool", "dramatic", "soft"}
VALID_BUBBLE_TYPES = {"dialogue", "thought", "narration", "shout"}
VALID_EMOTIONS = {"happy", "sad", "angry", "surprised", "neutral", "excited", "thoughtful"}

class StripScriptAgent(BaseAgent):
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, **kwargs) -> Dict[str, Any]:
        photo_analyses: List[Dict[str, Any]] = kwargs.get("photo_analyses", [])
        num_panels: int = kwargs.get("num_panels", 6)
        num_panels = max(3, min(6, num_panels))
        if not photo_analyses:
            return self._default_script([], num_panels)
        user_prompt = self._build_user_prompt(photo_analyses, num_panels)
        system_prompt = STRIP_SCRIPT_SYSTEM_PROMPT.format(num_panels=num_panels)
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_provider.chat_json(
                    messages, temperature=0.7,
                    max_tokens=400 + num_panels * 150,
                )
            )
            if self._validate_result(result, num_panels):
                return result
            logger.warning("StripScript LLM returned invalid structure, using fallback")
            return self._default_script(photo_analyses, num_panels)
        except Exception as e:
            logger.error(f"StripScript generation failed: {e}")
            return self._default_script(photo_analyses, num_panels)

    @staticmethod
    def _build_user_prompt(photo_analyses: List[Dict[str, Any]], num_panels: int) -> str:
        parts = [f"请根据以下 {len(photo_analyses)} 张照片分析，生成 {num_panels}格漫画脚本："]
        for i, a in enumerate(photo_analyses):
            people_info = []
            for j, p in enumerate(a.get("people", [])):
                people_info.append(
                    f"人物{j+1}：{p.get('traits','')}，服装：{p.get('clothing','')}，位置：{p.get('position_rel','')}，朝向：{p.get('facing','')}，动作：{p.get('action','')}，情绪：{p.get('emotion','')}"
                )
            parts.append(
                f"照片{i+1}：场景描述：{a.get('scene_desc','')}；地点：{a.get('location','')}；动作：{a.get('action','')}；光线：{a.get('lighting','')}；色调：{a.get('color_tone','')}；情绪：{a.get('emotion','')}；物品：{'、'.join(a.get('objects',[])[:5])}；服装：{'、'.join(a.get('clothing',[])[:5])}；人物：{' | '.join(people_info)}"
            )
        return "\n".join(parts)

    @staticmethod
    def _validate_result(result: Any, num_panels: int) -> bool:
        if not isinstance(result, dict):
            return False
        panels = result.get("panels")
        if not isinstance(panels, list) or len(panels) < num_panels:
            return False

        # 兼容两种格式：新格式 characters[] 数组，老格式 character_desc 字符串
        characters = result.get("characters")
        char_desc = result.get("character_desc", "")

        # 如果只有老格式 character_desc，转成 characters 数组（单角色）
        if not isinstance(characters, list) or len(characters) == 0:
            if char_desc and len(char_desc) >= 5:
                characters = [{"name": "主角", "char_desc": char_desc, "photo_people_idx": 0}]
                result["characters"] = characters
            else:
                return False

        # 验证 characters 数组
        valid_chars = []
        for c in characters:
            if not isinstance(c, dict):
                continue
            cd = c.get("char_desc", "")
            if not cd or len(cd) < 5:
                continue
            # 确保有 name 和 photo_people_idx
            if "name" not in c:
                c["name"] = f"角色{len(valid_chars)+1}"
            if "photo_people_idx" not in c:
                c["photo_people_idx"] = len(valid_chars)
            valid_chars.append(c)
        if not valid_chars:
            return False
        result["characters"] = valid_chars
        # 同时保留 character_desc 字段（取第一个角色的描述）用于向后兼容
        result["character_desc"] = valid_chars[0]["char_desc"]

        for p in panels[:num_panels]:
            if not isinstance(p, dict):
                return False
            if not p.get("description") or not p.get("dialogue"):
                return False
            src_idx = p.get("source_photo_idx")
            if not isinstance(src_idx, int) or src_idx < -1:
                return False
            # 校验镜头字段合法性：兼容外部返回 'shot' 或 'shot_type'，大小写和短横线处理
            st_raw = p.get("shot_type") or p.get("shot") or "medium"
            st = str(st_raw).lower().replace("-", "_").replace(" ", "_")
            if st not in VALID_SHOT_TYPES:
                st = "medium"
            p["shot_type"] = st

            comp_raw = p.get("composition", "rule_of_thirds")
            comp = str(comp_raw).lower()
            if comp not in VALID_COMPOSITIONS:
                comp = "rule_of_thirds"
            p["composition"] = comp

            lt_raw = p.get("lighting", "warm")
            lt = str(lt_raw).lower()
            if lt not in VALID_LIGHTINGS:
                lt = "warm"
            p["lighting"] = lt

            # 规范化 speaker_id（对应 characters 数组索引）
            spk_raw = p.get("speaker_id", 0)
            if not isinstance(spk_raw, int):
                try:
                    spk_raw = int(spk_raw)
                except (ValueError, TypeError):
                    spk_raw = 0
            # 边界检查
            if spk_raw >= len(valid_chars):
                spk_raw = 0
            p["speaker_id"] = spk_raw

            # 规范化 present_chars（当前格出场角色）
            present = p.get("present_chars")
            if not isinstance(present, list):
                # 默认所有角色都出场
                present = list(range(len(valid_chars)))
            # 过滤无效索引
            present = [idx for idx in present if isinstance(idx, int) and 0 <= idx < len(valid_chars)]
            if not present:
                present = [0]
            p["present_chars"] = present

            # 规范化 bubble_type
            bt_raw = p.get("bubble_type", "dialogue")
            bt = str(bt_raw).lower()
            if bt not in VALID_BUBBLE_TYPES:
                bt = "dialogue"
            p["bubble_type"] = bt

            # 规范化 emotion
            em_raw = p.get("emotion", "neutral")
            em = str(em_raw).lower()
            if em not in VALID_EMOTIONS:
                em = "neutral"
            p["emotion"] = em

        # 接受 LLM 返回的结构（已做字段规范化），不强制要求镜头多样性
        return True

    @staticmethod
    def _default_script(photo_analyses: List[Dict[str, Any]], num_panels: int) -> Dict[str, Any]:
        # 构建 characters 数组（从第一张照片的 people 中提取）
        characters = []
        if photo_analyses:
            first_analysis = photo_analyses[0]
            for idx, p in enumerate(first_analysis.get("people", [])):
                traits = p.get("traits", "")
                clothing = p.get("clothing", "")
                desc_parts = []
                if traits:
                    desc_parts.append(traits)
                if clothing:
                    desc_parts.append(f"穿着{clothing}")
                if desc_parts:
                    char_desc = "，".join(desc_parts)
                else:
                    char_desc = "一个普通人"
                characters.append({
                    "name": f"角色{idx+1}",
                    "char_desc": char_desc,
                    "photo_people_idx": idx,
                })

        if not characters:
            characters = [{"name": "主角", "char_desc": "一个年轻的普通人", "photo_people_idx": 0}]

        char_desc = characters[0]["char_desc"]

        action_keywords = ["吃", "饭", "餐", "喝", "面", "食", "厨房", "餐厅", "桌", "碗", "筷"]
        work_keywords = ["工作", "电脑", "办公", "文件", "会议", "写", "打字", "键盘", "屏幕"]
        outdoor_keywords = ["公园", "户外", "散步", "走", "运动", "跑步", "街", "路"]
        relax_keywords = ["休息", "睡觉", "躺", "坐", "沙发", "看", "手机", "书", "电视"]
        happy_keywords = ["笑", "开心", "高兴", "快乐", "微笑"]
        surprised_keywords = ["惊讶", "吃惊", "意外", "震惊"]

        def _match_keywords(text: str, keywords: list) -> bool:
            if not text:
                return False
            for kw in keywords:
                if kw in text:
                    return True
            return False

        def _get_scene_type(analysis):
            text = analysis.get("action", "") + analysis.get("scene_desc", "") + analysis.get("location", "")
            if _match_keywords(text, action_keywords):
                return "eating"
            elif _match_keywords(text, work_keywords):
                return "working"
            elif _match_keywords(text, outdoor_keywords):
                return "outdoor"
            elif _match_keywords(text, relax_keywords):
                return "relaxing"
            else:
                return "general"

        def _build_defaults(num_p: int):
            num_photos = len(photo_analyses) if photo_analyses else 1
            descs = []
            dialogues = []
            srcs = []

            eating_dialogues = [
                "今天的饭菜好香啊~", "嗯~味道真不错！", "这个味道好特别！",
                "等等，这个有点辣...", "不过真的很过瘾！", "吃饱啦，满足！"
            ]
            eating_descs = [
                "主角坐在餐桌前，面前摆着美味的饭菜",
                "主角正在享用美食，表情很满足",
                "主角尝到了独特的味道，眼睛一亮",
                "主角吃到了辣的东西，表情有点惊讶",
                "主角克服了辣味，吃得很开心",
                "主角吃完了饭，露出满意的笑容",
            ]
            working_dialogues = [
                "今天工作好多啊...", "这个方案终于搞定了！", "效率不错嘛~",
                "咦？这里好像有问题", "赶紧修改一下", "下班啦，明天继续~"
            ]
            working_descs = [
                "主角坐在办公桌前，看着电脑屏幕",
                "主角专注地工作，终于完成了任务",
                "主角看着完成的工作，露出满意的表情",
                "主角发现工作中的小问题，眉头微皱",
                "主角认真地修改问题，表情专注",
                "主角结束工作，放松地伸了个懒腰",
            ]
            outdoor_dialogues = [
                "今天天气真舒服~", "这里的风景好美！", "空气真清新！",
                "哇，这里好热闹啊", "玩得真开心！", "走走逛逛真开心~"
            ]
            outdoor_descs = [
                "主角走在户外，享受着美好的天气",
                "主角欣赏着周围的风景，心情愉悦",
                "主角深呼吸，享受新鲜空气",
                "主角发现了有趣的事情，眼睛一亮",
                "主角参与到活动中，笑得很开心",
                "主角愉快地结束了散步，满脸笑容",
            ]
            relaxing_dialogues = [
                "终于可以休息一下了~", "这个真好看！", "看得太入迷了！",
                "哎呀，时间过得好快", "再看一会儿~", "休息够了，明天加油！"
            ]
            relaxing_descs = [
                "主角舒适地休息，放松身心",
                "主角专注地看着手机或书，很投入",
                "主角被内容吸引，看得津津有味",
                "主角发现时间过得很快，有点惊讶",
                "主角决定再多看一会儿，意犹未尽",
                "主角休息完毕，精神饱满地准备继续",
            ]
            general_dialogues = [
                "今天过得真充实~", "发生了一件有趣的事！", "原来是这样啊~",
                "真有意思！", "学到了新知识", "生活真美好！"
            ]
            general_descs = [
                "主角出现在照片场景中，表情轻松愉快",
                "主角遇到了有趣的事情，露出好奇的表情",
                "主角明白了什么，若有所思地点点头",
                "主角觉得很有趣，开心地笑了",
                "主角若有所思，脸上带着微笑",
                "主角会心一笑，温馨收尾",
            ]

            scene_map = {
                "eating": (eating_descs, eating_dialogues),
                "working": (working_descs, working_dialogues),
                "outdoor": (outdoor_descs, outdoor_dialogues),
                "relaxing": (relaxing_descs, relaxing_dialogues),
                "general": (general_descs, general_dialogues),
            }

            for i in range(num_p):
                src_idx = min(i % num_photos, num_photos - 1)
                srcs.append(src_idx)
                if photo_analyses and src_idx < len(photo_analyses):
                    pa = photo_analyses[src_idx]
                    scene_type = _get_scene_type(pa)
                    scene_descs, scene_dialogues = scene_map[scene_type]
                    scene_desc_from_photo = pa.get("scene_desc", "")
                    if scene_desc_from_photo and len(scene_desc_from_photo) > 10:
                        base_desc = scene_descs[i % len(scene_descs)]
                        custom_desc = base_desc + "，" + scene_desc_from_photo[:60]
                        descs.append(custom_desc)
                    else:
                        descs.append(scene_descs[i % len(scene_descs)])
                    dialogues.append(scene_dialogues[i % len(scene_dialogues)])
                else:
                    descs.append(general_descs[i % len(general_descs)])
                    dialogues.append(general_dialogues[i % len(general_dialogues)])

            return descs, dialogues, srcs

        default_descs, default_dialogues, default_src = _build_defaults(num_panels)

        # 根据第一张照片的场景类型选择镜头模版
        scene_type = _get_scene_type(photo_analyses[0]) if photo_analyses else "general"
        template = SHOT_TEMPLATES.get(scene_type, SHOT_TEMPLATES["general"])
        shot_types = template["shots"]
        compositions = template["compositions"]
        lightings = template["lightings"]

        panels = []
        num_chars = len(characters)
        for i in range(num_panels):
            # 所有角色默认都出场
            present = list(range(num_chars))
            # 说话人轮流转
            speaker = i % num_chars
            panels.append({
                "panel_num": i + 1,
                "source_photo_idx": default_src[i],
                "shot_type": shot_types[i % len(shot_types)],
                "composition": compositions[i % len(compositions)],
                "lighting": lightings[i % len(lightings)],
                "description": default_descs[i],
                "dialogue": default_dialogues[i],
                "speaker_id": speaker,
                "present_chars": present,
                "bubble_type": "dialogue",
                "emotion": "happy",
            })
        return {"title": "生活小确幸", "character_desc": char_desc, "characters": characters, "panels": panels}
