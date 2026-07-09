# 快速条漫生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现"照片上传→30-60秒生成2×3条漫长图"功能，单张QuickStrip表记录结果，6格数据内存流转不持久化。

**Architecture:** 新建独立快速通道 `StripService`，不复用现有5-Agent PDF流水线。流水线：照片保存→并行视觉分析→单次LLM生6格脚本→asyncio并行生6张图→PIL拼长图+气泡。API为同步等待（HTTP连接保持），30-60秒返回。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 Async, Pydantic v2, Pillow (PIL), asyncio, Qianfan musesteamer-air-image, DashScope Qwen VL

## Global Constraints

- Python >= 3.10, 使用现有依赖栈（Pillow 通过 reportlab 间接安装）
- 主键使用 Integer, autoincrement（与现有模型一致）
- 视觉分析使用现有 QianfanProvider（实际是 DashScope Qwen VL，通过 configure(api_key, base_url) 配置）
- 图片生成使用现有 QianfanImageProvider，并行生成6张（semaphore=3 限制并发）
- 条漫输出目录：`{COMICS_DIR}/strips/{strip_id}/`
- 单格图片尺寸：竖版 864x1152（最终长图：1752宽 × 3480高）
- 照片数量限制：2-6 张
- 网格固定：2列×3行（共6格）
- 不添加代码注释
- 所有新增代码遵循现有代码风格
- 中文界面提示
- **只创建一张表 quick_strips**，不创建 strip_panels 表

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/db/models.py` | 修改 | 仅添加 QuickStrip 模型（无 StripPanel） |
| `app/agents/strip_script_agent.py` | 新建 | 单次LLM：照片分析→6格脚本+对白+角色描述 |
| `app/agents/strip_layout_agent.py` | 新建 | PIL后处理：气泡+边框+拼2×3长图 |
| `app/agents/vision_agent.py` | 修改 | 添加 run_async 方法（run_in_executor包装同步调用） |
| `app/llm/qianfan_image_provider.py` | 修改 | 添加 generate_async 方法 |
| `app/services/strip_service.py` | 新建 | 编排：并行视觉分析+并行生图+组装 |
| `app/api/strips.py` | 新建 | API路由：POST上传/GET查询/GET图片 |
| `app/agents/__init__.py` | 修改 | 导出新Agent |
| `main.py` | 修改 | 挂载strips路由+启动时创建目录 |
| `tests/test_strip_script.py` | 新建 | StripScriptAgent测试 |
| `tests/test_strip_layout.py` | 新建 | StripLayoutAgent测试 |
| `tests/test_strip_service.py` | 新建 | StripService测试 |
| `tests/test_strip_api.py` | 新建 | API端点测试 |

---

### Task 1: 数据模型 - QuickStrip（单表，极简）

**Files:**
- Modify: `app/db/models.py`
- Test: `tests/test_strip_models.py`

- [ ] **Step 1: 在 app/db/models.py 末尾追加 QuickStrip 模型**

在现有模型（PromptVersion）之后追加：

```python
class QuickStrip(Base):
    __tablename__ = "quick_strips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, default="")
    status = Column(String(20), nullable=False, default="processing")
    image_path = Column(String(500), nullable=False, default="")
    error_msg = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, server_default=func.now())
```

注意：不添加 StripPanel 模型，不添加 relationship（panels数据不持久化）。

- [ ] **Step 2: 编写模型测试**

创建 `tests/test_strip_models.py`：

```python
from app.db.models import QuickStrip


def test_quick_strip_defaults():
    qs = QuickStrip()
    assert qs.status == "processing"
    assert qs.title == ""
    assert qs.image_path == ""
    assert qs.error_msg == ""


def test_quick_strip_creation():
    qs = QuickStrip(id=1, title="测试条漫", status="completed", image_path="/tmp/strip.png")
    assert qs.id == 1
    assert qs.title == "测试条漫"
    assert qs.status == "completed"
```

- [ ] **Step 3: 运行测试确认通过**

Run: `pytest tests/test_strip_models.py -v`
Expected: 2 passed

- [ ] **Step 4: 提交**

```bash
git add app/db/models.py tests/test_strip_models.py
git commit -m "feat: add QuickStrip model for quick comic strip generation (single table)"
```

---

### Task 2: StripScriptAgent - 单次LLM生成6格脚本

**Files:**
- Create: `app/agents/strip_script_agent.py`
- Modify: `app/agents/__init__.py`
- Test: `tests/test_strip_script.py`

**Interfaces:**
- Consumes: `llm_provider` (OpenAIProvider，需有 `chat_json(messages, temperature=...)` 方法)
- Produces:
  - `class StripScriptAgent(BaseAgent)`
  - `async def run(self, **kwargs) -> Dict[str, Any]`
  - 输入 kwargs: `photo_analyses: List[dict]`
  - 返回: `{"title": str, "character_desc": str, "panels": [{"panel_num": int, "source_photo_idx": int, "description": str, "dialogue": str, "shot": str}]}`（6个panel）

- [ ] **Step 1: 读取 app/agents/base.py 和 app/agents/dialogue_agent.py 确认现有模式**

- [ ] **Step 2: 编写失败测试**

创建 `tests/test_strip_script.py`：

```python
from unittest.mock import MagicMock
from app.agents.strip_script_agent import StripScriptAgent


def _mock_provider(result=None):
    p = MagicMock()
    p.chat_json = MagicMock(return_value=result or {
        "title": "测试标题",
        "character_desc": "戴眼镜的男生",
        "panels": [
            {"panel_num": i+1, "source_photo_idx": 0, "description": f"格{i+1}描述", "dialogue": f"对白{i+1}", "shot": "Medium"}
            for i in range(6)
        ]
    })
    return p


def test_returns_6_panels():
    import asyncio
    agent = StripScriptAgent(llm_provider=_mock_provider())
    result = asyncio.run(agent.run(photo_analyses=[
        {"scene_desc": "餐厅", "people": [{"traits": "戴眼镜男生"}], "action": "吃饭"}
    ]))
    assert result["title"] == "测试标题"
    assert len(result["panels"]) == 6
    for p in result["panels"]:
        assert p["dialogue"]
        assert p["description"]


def test_fallback_on_llm_failure():
    import asyncio
    p = MagicMock()
    p.chat_json = MagicMock(side_effect=Exception("LLM error"))
    agent = StripScriptAgent(llm_provider=p)
    result = asyncio.run(agent.run(photo_analyses=[{"scene_desc": "公园", "action": "散步"}, {"scene_desc": "咖啡店", "action": "喝咖啡"}]))
    assert len(result["panels"]) == 6
    assert result["title"]


def test_fallback_on_incomplete_panels():
    import asyncio
    p = MagicMock()
    p.chat_json = MagicMock(return_value={"panels": [{"panel_num": 1, "description": "x", "dialogue": "y"}]})
    agent = StripScriptAgent(llm_provider=p)
    result = asyncio.run(agent.run(photo_analyses=[{"scene_desc": "办公室"}]))
    assert len(result["panels"]) == 6


def test_build_prompt_contains_photo_info():
    agent = StripScriptAgent(llm_provider=_mock_provider())
    prompt = agent._build_user_prompt([
        {"scene_desc": "餐厅里摆满菜", "people": [{"traits": "戴黑框眼镜男生"}], "action": "叹气", "emotion": "疲惫", "location": "餐厅", "clothing": ["浅蓝色衬衫"]},
    ])
    assert "餐厅" in prompt
    assert "眼镜" in prompt
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_strip_script.py -v`
Expected: FAIL

- [ ] **Step 4: 实现 StripScriptAgent**

创建 `app/agents/strip_script_agent.py`：

```python
import logging
from typing import Any, Dict, List

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

STRIP_SCRIPT_SYSTEM_PROMPT = """你是一个专业条漫编剧。根据用户提供的照片场景分析，创作一个2列3行（共6格）的六格漫画风格条漫脚本。

核心要求：
1. 从照片中提取核心人物外貌特征（发型、服装、配饰如眼镜等），在character_desc中详细描述，确保6格画风一致
2. 每格包含画面描述（中文，50字以内，包含人物外貌、动作、表情、场景）
3. 每格包含对白（中文，30字以内，幽默/温馨/生活化，要有梗）
4. 6格有起承转合：开头格(1-2)铺垫，中间格(3-4)发展/转折，结尾格(5-6)punchline/温馨收尾
5. shot类型：Wide/Medium/Close-up/Action/Ending
6. 画面风格：半色调网点漫画风，粗黑线条，温暖色调

输出严格JSON格式，不要输出任何其他文字：
{
  "title": "条漫标题（10字以内）",
  "character_desc": "人物外貌统一描述，用于所有格生图保持一致性",
  "panels": [
    {"panel_num": 1, "source_photo_idx": 0, "description": "画面描述", "dialogue": "对白", "shot": "Medium"}
  ]
}"""


class StripScriptAgent(BaseAgent):
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, **kwargs) -> Dict[str, Any]:
        photo_analyses: List[Dict[str, Any]] = kwargs.get("photo_analyses", [])
        if not photo_analyses:
            return self._default_script([])
        user_prompt = self._build_user_prompt(photo_analyses)
        try:
            messages = [
                {"role": "system", "content": STRIP_SCRIPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.8)
            if self._validate_result(result):
                return result
            logger.warning("StripScript LLM returned invalid structure, using fallback")
            return self._default_script(photo_analyses)
        except Exception as e:
            logger.error(f"StripScript generation failed: {e}")
            return self._default_script(photo_analyses)

    @staticmethod
    def _build_user_prompt(photo_analyses: List[Dict[str, Any]]) -> str:
        parts = ["以下是用户上传照片的场景分析，请基于这些照片创作6格条漫：\n"]
        for i, a in enumerate(photo_analyses):
            people_str = ""
            people = a.get("people", [])
            if people:
                traits = [p.get("traits", "") for p in people if p.get("traits")]
                people_str = "；".join(traits)
            clothing = a.get("clothing", [])
            clothing_str = "、".join(clothing) if clothing else ""
            parts.append(
                f"照片{i+1}：场景：{a.get('scene_desc', '')}；地点：{a.get('location', '')}；"
                f"人物：{people_str}；穿着：{clothing_str}；"
                f"动作：{a.get('action', '')}；情绪：{a.get('emotion', '')}"
            )
        parts.append("\n请创作6格幽默/温馨条漫，确保人物外貌在所有格中一致。")
        return "\n".join(parts)

    @staticmethod
    def _validate_result(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        panels = result.get("panels")
        if not isinstance(panels, list) or len(panels) < 6:
            return False
        for p in panels[:6]:
            if not isinstance(p, dict):
                return False
            if not p.get("description") or not p.get("dialogue"):
                return False
        return True

    @staticmethod
    def _default_script(photo_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        default_descs = [
            "主角出现在场景中，表情轻松愉快",
            "主角发现了什么有趣的事情，眼睛一亮",
            "主角兴奋地做出反应，动作夸张",
            "事情出现了意想不到的转折",
            "主角露出无奈或惊讶的表情",
            "主角会心一笑，温馨收尾",
        ]
        default_dialogues = ["今天天气真好啊~", "咦？那是什么？", "太棒了！", "等等，这不对吧……", "啊这……", "哈哈，生活就是这样嘛~"]
        panels = []
        for i in range(6):
            src_idx = min(i, len(photo_analyses) - 1) if photo_analyses else 0
            panels.append({
                "panel_num": i + 1, "source_photo_idx": src_idx,
                "description": default_descs[i], "dialogue": default_dialogues[i], "shot": "Medium",
            })
        return {"title": "生活小确幸", "character_desc": "一个年轻的普通人", "panels": panels}
```

- [ ] **Step 5: 在 app/agents/__init__.py 中添加 StripScriptAgent 导入**

读取现有文件，追加：`from .strip_script_agent import StripScriptAgent`

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_strip_script.py -v`
Expected: 4 passed

- [ ] **Step 7: 提交**

```bash
git add app/agents/strip_script_agent.py app/agents/__init__.py tests/test_strip_script.py
git commit -m "feat: add StripScriptAgent - single LLM call generates 6-panel strip script"
```

---

### Task 3: StripLayoutAgent - PIL拼长图+对白气泡

**Files:**
- Create: `app/agents/strip_layout_agent.py`
- Modify: `app/agents/__init__.py`（追加 StripLayoutAgent 导入）
- Test: `tests/test_strip_layout.py`

**Interfaces:**
- Produces:
  - `class StripLayoutAgent(BaseAgent)`
  - `async def run(self, **kwargs) -> Dict[str, Any]`
  - 输入: `panels: List[dict]`（每个含 image_path, dialogue）, `strip_dir: str`
  - 返回: `{"image_path": str, "width": int, "height": int}`
  - 常量: PANEL_W=864, PANEL_H=1152, COLS=2, ROWS=3, BORDER=6, OUTER_BORDER=8

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_strip_layout.py`：

```python
import os
from PIL import Image
from app.agents.strip_layout_agent import StripLayoutAgent, PANEL_W, PANEL_H, COLS, ROWS, BORDER, OUTER_BORDER


def _make_panels(tmp_path):
    panels = []
    for i in range(6):
        img = Image.new("RGB", (PANEL_W, PANEL_H), color=(200 + i * 8, 180, 150))
        p = tmp_path / f"panel_{i+1}.png"
        img.save(str(p))
        panels.append({"panel_num": i + 1, "image_path": str(p), "dialogue": f"对白内容{i+1}哈"})
    return panels


def test_compose_creates_long_image(tmp_path):
    import asyncio
    panels = _make_panels(tmp_path)
    result = asyncio.run(StripLayoutAgent().run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])
    img = Image.open(result["image_path"])
    expected_w = PANEL_W * COLS + BORDER * (COLS - 1) + OUTER_BORDER * 2
    expected_h = PANEL_H * ROWS + BORDER * (ROWS - 1) + OUTER_BORDER * 2
    assert img.width == expected_w
    assert img.height == expected_h


def test_compose_handles_missing_images(tmp_path):
    import asyncio
    panels = _make_panels(tmp_path)
    panels[2]["image_path"] = "/nonexistent.png"
    result = asyncio.run(StripLayoutAgent().run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])


def test_compose_with_empty_dialogue(tmp_path):
    import asyncio
    panels = _make_panels(tmp_path)
    panels[0]["dialogue"] = ""
    result = asyncio.run(StripLayoutAgent().run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_strip_layout.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 StripLayoutAgent**

创建 `app/agents/strip_layout_agent.py`：

```python
import logging
import os
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PANEL_W = 864
PANEL_H = 1152
COLS = 2
ROWS = 3
BORDER = 6
OUTER_BORDER = 8
BUBBLE_PAD = 20
BUBBLE_RADIUS = 18
BUBBLE_FONT_SIZE = 28
WATERMARK = "LifeComicAI"


class StripLayoutAgent(BaseAgent):
    def __init__(self):
        pass

    async def run(self, **kwargs) -> Dict[str, Any]:
        panels: List[Dict[str, Any]] = kwargs.get("panels", [])
        strip_dir: str = kwargs.get("strip_dir", "")
        os.makedirs(strip_dir, exist_ok=True)

        output_path = os.path.join(strip_dir, "strip.png")
        total_w = PANEL_W * COLS + BORDER * (COLS - 1) + OUTER_BORDER * 2
        total_h = PANEL_H * ROWS + BORDER * (ROWS - 1) + OUTER_BORDER * 2

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(0, 0, 0), width=OUTER_BORDER)
        font = self._load_font(BUBBLE_FONT_SIZE)

        for idx, panel in enumerate(panels[:COLS * ROWS]):
            col = idx % COLS
            row = idx // COLS
            x = OUTER_BORDER + col * (PANEL_W + BORDER)
            y = OUTER_BORDER + row * (PANEL_H + BORDER)
            img_path = panel.get("image_path", "")
            if img_path and os.path.exists(img_path):
                try:
                    pimg = Image.open(img_path).convert("RGB").resize((PANEL_W, PANEL_H), Image.LANCZOS)
                    canvas.paste(pimg, (x, y))
                except Exception as e:
                    logger.warning(f"Failed to load panel image {img_path}: {e}")
                    self._draw_placeholder(draw, x, y)
            else:
                self._draw_placeholder(draw, x, y)
            draw.rectangle([x, y, x + PANEL_W - 1, y + PANEL_H - 1], outline=(0, 0, 0), width=4)
            dialogue = panel.get("dialogue", "")
            if dialogue:
                bubble = self._create_bubble(dialogue, PANEL_W - 80, font)
                if bubble:
                    bx, by = x + 40, y + 30
                    if by + bubble.height > y + PANEL_H - 20:
                        by = y + PANEL_H - bubble.height - 20
                    canvas.paste(bubble, (bx, by), bubble)

        self._draw_watermark(draw, total_w, total_h, font)
        canvas.save(output_path, "PNG")
        logger.info(f"Strip composed: {output_path} ({total_w}x{total_h})")
        return {"image_path": output_path, "width": total_w, "height": total_h}

    def _create_bubble(self, text: str, max_width: int, font: ImageFont.ImageFont) -> Image.Image:
        lines = self._wrap_text(text, font, max_width - BUBBLE_PAD * 2)
        if not lines:
            return None
        bbox = font.getbbox("测Ag")
        line_h = bbox[3] - bbox[1] + 8
        text_w = max(font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines)
        bw = text_w + BUBBLE_PAD * 2
        bh = line_h * len(lines) + BUBBLE_PAD * 2 + 12
        bubble = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)
        bd.rounded_rectangle([0, 0, bw - 1, bh - 15], radius=BUBBLE_RADIUS, fill=(255, 255, 255, 240), outline=(0, 0, 0), width=3)
        bd.polygon([(bw // 2 - 12, bh - 15), (bw // 2 + 12, bh - 15), (bw // 2, bh)], fill=(255, 255, 255, 240))
        bd.line([(bw // 2 - 12, bh - 15), (bw // 2, bh), (bw // 2 + 12, bh - 15)], fill=(0, 0, 0), width=3)
        yt = BUBBLE_PAD
        for line in lines:
            lb = font.getbbox(line)
            lw = lb[2] - lb[0]
            bd.text(((bw - lw) // 2, yt), line, fill=(0, 0, 0), font=font)
            yt += line_h
        return bubble

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        lines, cur = [], ""
        for ch in text:
            t = cur + ch
            if font.getbbox(t)[2] - font.getbbox(t)[0] <= max_width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines[:4]

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"]:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _draw_placeholder(draw: ImageDraw.ImageDraw, x: int, y: int):
        draw.rectangle([x, y, x + PANEL_W - 1, y + PANEL_H - 1], fill=(240, 240, 240))

    @staticmethod
    def _draw_watermark(draw: ImageDraw.ImageDraw, w: int, h: int, font: ImageFont.ImageFont):
        try:
            bb = font.getbbox(WATERMARK)
            fw, fh = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((w - fw - 20, h - fh - 12), WATERMARK, fill=(180, 180, 180), font=font)
        except Exception:
            pass
```

- [ ] **Step 4: 在 app/agents/__init__.py 中追加 StripLayoutAgent 导入**

`from .strip_layout_agent import StripLayoutAgent`

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_strip_layout.py -v`
Expected: 3 passed

- [ ] **Step 6: 提交**

```bash
git add app/agents/strip_layout_agent.py app/agents/__init__.py tests/test_strip_layout.py
git commit -m "feat: add StripLayoutAgent - PIL compose 2x3 strip with speech bubbles and borders"
```

---

### Task 4: VisionAgent async + QianfanImageProvider async + StripService 编排

**Files:**
- Modify: `app/agents/vision_agent.py`（添加 run_async 方法）
- Modify: `app/llm/qianfan_image_provider.py`（添加 generate_async 方法）
- Create: `app/services/strip_service.py`
- Test: `tests/test_strip_service.py`

**Interfaces:**
- Produces:
  - `VisionAgent.run_async(photo_id, image_path)` async方法
  - `QianfanImageProvider.generate_async(prompt, size)` async方法
  - `class StripService`
  - `async def generate_strip(self, photo_paths: List[str], session: AsyncSession) -> dict`

- [ ] **Step 1: 在 VisionAgent 添加 run_async**

读取 `app/agents/vision_agent.py`，在现有 `run` 方法后添加：

```python
    async def run_async(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_run, photo_id, image_path)

    def _sync_run(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        try:
            analysis = self.qianfan_provider.analyze_image(image_path)
            return {
                "photo_id": photo_id,
                "scene_desc": analysis.get("scene_desc", ""),
                "weather": analysis.get("weather", ""),
                "location": analysis.get("location", ""),
                "action": analysis.get("action", ""),
                "emotion": analysis.get("emotion", ""),
                "clothing": analysis.get("clothing", []),
                "people": analysis.get("people", []),
                "objects": analysis.get("objects", []),
            }
        except Exception as e:
            logger.error(f"Vision analysis failed for photo {photo_id}: {e}")
            return {"photo_id": photo_id, "scene_desc": "", "weather": "", "location": "", "action": "", "emotion": "", "clothing": [], "people": [], "objects": []}
```

- [ ] **Step 2: 在 QianfanImageProvider 添加 generate_async**

在 `app/llm/qianfan_image_provider.py` 的 `generate_with_retry` 方法后添加：

```python
    async def generate_async(
        self,
        prompt: str,
        size: str = "864x1152",
        model: str = "musesteamer-air-image",
    ) -> Optional[dict]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_with_retry, prompt, size, model, 2)
```

- [ ] **Step 3: 编写失败测试**

创建 `tests/test_strip_service.py`：

```python
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.strip_service import StripService


@pytest.fixture
def mock_session():
    s = AsyncMock()
    s.execute = AsyncMock()
    s.flush = AsyncMock()
    s.commit = AsyncMock()
    s.refresh = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_generate_strip_validates_photo_count(tmp_path, mock_session):
    svc = StripService(
        vision_agent=MagicMock(), script_agent=MagicMock(),
        image_provider=MagicMock(), layout_agent=MagicMock(),
        comics_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="2-6"):
        await svc.generate_strip(photo_paths=["/a.jpg"], session=mock_session)


@pytest.mark.asyncio
async def test_generate_strip_full_pipeline(tmp_path, mock_session):
    photo = tmp_path / "test.jpg"
    photo.write_bytes(b"fake")
    photo2 = tmp_path / "test2.jpg"
    photo2.write_bytes(b"fake")

    vision = MagicMock()
    vision.run_async = AsyncMock(return_value={
        "scene_desc": "餐厅吃饭", "people": [{"traits": "戴眼镜男生"}],
        "action": "用餐", "emotion": "开心", "location": "餐厅", "clothing": ["蓝衬衫"], "objects": [],
    })
    script = MagicMock()
    script.run = AsyncMock(return_value={
        "title": "聚餐日常", "character_desc": "戴眼镜男生",
        "panels": [{"panel_num": i+1, "source_photo_idx": 0, "description": f"格{i+1}", "dialogue": f"对白{i+1}", "shot": "Medium"} for i in range(6)],
    })
    img_prov = MagicMock()
    img_prov.generate_async = AsyncMock(return_value={"urls": ["https://example.com/img.png"]})
    layout = MagicMock()
    layout.run = AsyncMock(return_value={"image_path": str(tmp_path / "strip.png"), "width": 1752, "height": 3480})

    with patch("app.services.strip_service.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.content = b"fakedata"
        mock_resp.raise_for_status = MagicMock()
        mock_req.get.return_value = mock_resp
        with patch("app.services.strip_service.Image.new") as mock_new:
            mock_img = MagicMock()
            mock_img.save = MagicMock()
            mock_new.return_value = mock_img
            svc = StripService(vision_agent=vision, script_agent=script, image_provider=img_prov, layout_agent=layout, comics_dir=str(tmp_path))
            mock_session.refresh.side_effect = lambda x: setattr(x, "id", 1) or setattr(x, "status", "completed")
            result = await svc.generate_strip(photo_paths=[str(photo), str(photo2)], session=mock_session)

    assert result["strip_id"] == 1
    assert result["title"] == "聚餐日常"
    assert result["status"] == "completed"
    assert result["image_path"].endswith("strip.png")
    assert len(result["panels"]) == 6
    mock_session.commit.assert_called()
```

- [ ] **Step 4: 运行测试确认失败**

Run: `pytest tests/test_strip_service.py -v`
Expected: FAIL

- [ ] **Step 5: 实现 StripService**

创建 `app/services/strip_service.py`：

```python
import asyncio
import logging
import os
from typing import Any, Dict, List

import requests
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.strip_layout_agent import PANEL_H, PANEL_W
from app.db.models import QuickStrip

logger = logging.getLogger(__name__)


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
        try:
            analyses = await self._analyze_photos(photo_paths)
            script = await self.script_agent.run(photo_analyses=analyses)
            strip.title = script.get("title", "生活条漫")
            char_desc = script.get("character_desc", "")
            panels_script = script.get("panels", [])[:6]
            panel_results = await self._generate_panels(panels_script, char_desc, out_dir)
            compose = await self.layout_agent.run(panels=panel_results, strip_dir=out_dir)
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

    async def _generate_panels(self, panels: List[Dict[str, Any]], char_desc: str, out_dir: str) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(3)
        async def gen_one(p):
            async with sem:
                pn = p["panel_num"]
                desc, dialogue = p.get("description", ""), p.get("dialogue", "")
                prompt = f"{char_desc}。{desc}。漫画风格，粗黑线稿，半色调网点，温暖色调，高质量竖版插画"
                try:
                    r = await self.image_provider.generate_async(prompt, size="864x1152")
                    if r and r.get("urls"):
                        lp = os.path.join(out_dir, f"panel_{pn}.png")
                        if await self._download(r["urls"][0], lp):
                            p["image_path"] = lp
                            return p
                except Exception as e:
                    logger.warning(f"Panel {pn} failed: {e}")
                lp = os.path.join(out_dir, f"panel_{pn}.png")
                self._placeholder(lp)
                p["image_path"] = lp
                return p
        return await asyncio.gather(*[gen_one(dict(p)) for p in panels])

    @staticmethod
    async def _download(url: str, path: str) -> bool:
        import asyncio
        loop = asyncio.get_event_loop()
        def _dl():
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
            except Exception as e:
                logger.warning(f"Download failed: {e}")
                return False
        return await loop.run_in_executor(None, _dl)

    @staticmethod
    def _placeholder(path: str):
        Image.new("RGB", (PANEL_W, PANEL_H), (245, 245, 245)).save(path, "PNG")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_strip_service.py -v`
Expected: 2 passed

- [ ] **Step 7: 提交**

```bash
git add app/agents/vision_agent.py app/llm/qianfan_image_provider.py app/services/strip_service.py tests/test_strip_service.py
git commit -m "feat: add StripService with parallel photo analysis and image generation"
```

---

### Task 5: API端点 - strips router

**Files:**
- Create: `app/api/strips.py`
- Test: `tests/test_strip_api.py`

**Interfaces:**
- Produces:
  - `POST /api/v1/strip/generate` - multipart上传照片（2-6张），同步等待返回
  - `GET /api/v1/strip/{strip_id}` - 查询状态
  - `GET /api/v1/strip/{strip_id}/image` - 返回长图PNG文件

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_strip_api.py`：

```python
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _img_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), (200, 180, 160)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _make_app(monkeypatch, tmp_path):
    from app.api import strips as sm
    from app.db.models import Base
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    import asyncio
    async def init():
        async with engine.begin() as c:
            await c.run_sync(Base.metadata.create_all)
    asyncio.run(init())
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async def override_session():
        async with sf() as s:
            yield s
    sm.get_session_override = override_session
    class FakeSvc:
        async def generate_strip(self, photo_paths, session):
            sp = tmp_path / "strips" / "1"
            sp.mkdir(parents=True, exist_ok=True)
            strip_p = sp / "strip.png"
            Image.new("RGB", (100, 100), (255, 0, 0)).save(str(strip_p))
            return {"strip_id": 1, "title": "测试", "status": "completed", "image_path": str(strip_p), "panels": []}
    monkeypatch.setattr(sm, "_build_strip_service", lambda: FakeSvc())
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(sm.strips_router)
    return app


def test_generate_endpoint(monkeypatch, tmp_path):
    client = TestClient(_make_app(monkeypatch, tmp_path))
    files = [("files", ("a.jpg", _img_bytes(), "image/jpeg")), ("files", ("b.jpg", _img_bytes(), "image/jpeg"))]
    r = client.post("/api/v1/strip/generate", files=files)
    assert r.status_code == 200
    d = r.json()
    assert d["code"] == 0
    assert d["data"]["strip_id"] == 1
    assert d["data"]["status"] == "completed"
    assert "image_url" in d["data"]


def test_generate_too_few(monkeypatch, tmp_path):
    client = TestClient(_make_app(monkeypatch, tmp_path))
    r = client.post("/api/v1/strip/generate", files=[])
    assert r.status_code == 422


def test_get_strip(monkeypatch, tmp_path):
    client = TestClient(_make_app(monkeypatch, tmp_path))
    r = client.get("/api/v1/strip/1")
    assert r.status_code in (200, 404)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_strip_api.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 strips API**

创建 `app/api/strips.py`：

```python
import logging
import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.strip_layout_agent import StripLayoutAgent
from app.agents.strip_script_agent import StripScriptAgent
from app.agents.vision_agent import VisionAgent
from app.config.config import get_settings
from app.db.database import get_session
from app.db.models import QuickStrip
from app.llm.openai_provider import OpenAIProvider
from app.llm.qianfan_image_provider import QianfanImageProvider
from app.llm.qianfan_provider import QianfanProvider
from app.schemas.common import APIResponse
from app.services.strip_service import StripService

logger = logging.getLogger(__name__)
get_session_override = None


def _get_session():
    if get_session_override:
        return get_session_override()
    return get_session()


def _build_strip_service() -> StripService:
    settings = get_settings()
    qianfan = QianfanProvider(access_key=settings.QIANFAN_ACCESS_KEY, secret_key=settings.QIANFAN_SECRET_KEY)
    qianfan.configure(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    openai_p = OpenAIProvider(
        api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME, temperature=0.8, top_p=settings.LLM_TOP_P,
    )
    img_p = QianfanImageProvider(access_key=settings.QIANFAN_ACCESS_KEY, secret_key=settings.QIANFAN_SECRET_KEY)
    return StripService(
        vision_agent=VisionAgent(qianfan_provider=qianfan),
        script_agent=StripScriptAgent(llm_provider=openai_p),
        image_provider=img_p,
        layout_agent=StripLayoutAgent(),
        comics_dir=settings.COMICS_DIR,
    )


strips_router = APIRouter(prefix="/api/v1/strip", tags=["strip"])


@strips_router.post("/generate", response_model=APIResponse[dict])
async def generate_strip(
    files: List[UploadFile] = File(...),
    svc: StripService = Depends(lambda: _build_strip_service()),
    session: AsyncSession = Depends(_get_session),
):
    if not (2 <= len(files) <= 6):
        raise HTTPException(status_code=400, detail="请上传2-6张照片")
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
    upload_dir = os.path.join(get_settings().UPLOAD_DIR, "strip_photos")
    os.makedirs(upload_dir, exist_ok=True)
    photo_paths = []
    try:
        for f in files:
            if f.content_type not in allowed:
                raise HTTPException(status_code=400, detail=f"不支持的格式: {f.content_type}")
            ext = os.path.splitext(f.filename or "p.jpg")[1].lower() or ".jpg"
            if ext not in allowed_ext:
                ext = ".jpg"
            fp = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
            with open(fp, "wb") as out:
                shutil.copyfileobj(f.file, out)
            photo_paths.append(fp)
        result = await svc.generate_strip(photo_paths=photo_paths, session=session)
        return APIResponse(code=0, message="success", data={
            "strip_id": result["strip_id"], "status": result["status"],
            "title": result["title"],
            "image_url": f"/api/v1/strip/{result['strip_id']}/image",
            "panels": result.get("panels", []),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strip gen error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"条漫生成失败: {str(e)[:200]}")


@strips_router.get("/{strip_id}", response_model=APIResponse[dict])
async def get_strip(strip_id: int, session: AsyncSession = Depends(_get_session)):
    r = await session.execute(select(QuickStrip).where(QuickStrip.id == strip_id))
    s = r.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="条漫不存在")
    return APIResponse(code=0, data={
        "strip_id": s.id, "title": s.title, "status": s.status,
        "image_url": f"/api/v1/strip/{s.id}/image" if s.status == "completed" else None,
        "error_msg": s.error_msg,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    })


@strips_router.get("/{strip_id}/image")
async def get_strip_image(strip_id: int, session: AsyncSession = Depends(_get_session)):
    r = await session.execute(select(QuickStrip).where(QuickStrip.id == strip_id))
    s = r.scalar_one_or_none()
    if not s or not s.image_path or not os.path.exists(s.image_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(s.image_path, media_type="image/png")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_strip_api.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add app/api/strips.py tests/test_strip_api.py
git commit -m "feat: add strip API endpoints - upload photos, generate and serve strip image"
```

---

### Task 6: 主文件集成

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 修改 main.py**

在 import 区域添加：
```python
from app.api.strips import strips_router
```

在 `app.include_router(health_router)` 之前添加：
```python
app.include_router(strips_router)
```

确保 `import os` 在顶部（检查是否已有）。在 startup_event 末尾添加目录创建：
```python
    try:
        settings = get_settings()
        os.makedirs(settings.COMICS_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.COMICS_DIR, "strips"), exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Directory creation failed: {e}")
```

- [ ] **Step 2: 运行全部单元测试确认不破坏现有功能**

Run: `pytest tests/ -v --ignore=tests/test_comic_integration.py --ignore=tests/test_integration.py -x 2>&1 | Select-Object -Last 50`
Expected: 所有单元测试通过

- [ ] **Step 3: 提交**

```bash
git add main.py
git commit -m "feat: integrate strips router and ensure storage directories on startup"
```

---

### Task 7: 回归验证

- [ ] **Step 1: 运行全量测试**

Run: `pytest tests/ -v --ignore=tests/test_comic_integration.py --ignore=tests/test_integration.py 2>&1 | Select-Object -Last 20`
Expected: 所有单元测试通过

- [ ] **Step 2: 检查 imports 无循环依赖**

Run: `python -c "from main import app; print('App loaded OK')"`
Expected: "App loaded OK"

- [ ] **Step 3: 提交（如有修复）**

```bash
git add -A
git commit -m "chore: final verification for quick strip feature"
```
