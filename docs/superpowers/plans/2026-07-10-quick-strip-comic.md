# 快速条漫生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现"照片上传→30-60秒生成2×3条漫长图"功能，包含对白气泡PIL后处理、并行生图、单次LLM脚本生成。

**Architecture:** 新建独立的快速通道 `StripService`，不复用现有5-Agent PDF流水线。流水线：照片保存→并行视觉分析→单次LLM生6格脚本→asyncio并行生6张图→PIL拼长图+气泡。API为同步等待（HTTP连接保持），30-60秒返回。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 Async, Pydantic v2, Pillow (PIL), asyncio, Qianfan musesteamer-air-image, DashScope Qwen VL (已有QianfanProvider)

## Global Constraints

- Python >= 3.10, 使用现有依赖栈（Pillow 已通过 reportlab 间接安装，无需新增依赖）
- 主键使用 Integer, autoincrement（与现有模型一致）
- 视觉分析使用 DashScope Qwen VL（通过现有 QianfanProvider 类，虽然名叫 QianfanProvider 但实际调用 DashScope 兼容接口）
- 图片生成使用千帆 musesteamer-air-image（复用现有 QianfanImageProvider）
- 条漫输出目录：`{COMICS_DIR}/strips/{strip_id}/`
- 单格图片尺寸：竖版 864x1152（最终长图 1728x3456）
- 照片数量限制：2-6 张
- 总超时限制：90 秒
- 网格固定：2列×3行（共6格），如果照片少于6张则有重复或创意扩展
- 不添加代码注释
- 所有新增代码遵循现有代码风格（从 app/agents/ 和 app/services/ 学习模式）
- 中文界面提示

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/db/models.py` | 修改 | 添加 QuickStrip、StripPanel 模型 |
| `app/agents/strip_script_agent.py` | 新建 | 单次LLM：照片分析→6格脚本+对白+角色描述 |
| `app/agents/strip_layout_agent.py` | 新建 | PIL后处理：气泡+边框+拼2×3长图 |
| `app/services/strip_service.py` | 新建 | 编排：并行视觉分析+并行生图+组装 |
| `app/api/strips.py` | 新建 | API路由：POST/GET条漫端点 |
| `app/agents/__init__.py` | 修改 | 导出新Agent |
| `main.py` | 修改 | 挂载strips路由+静态文件目录 |
| `tests/test_strip_script.py` | 新建 | StripScriptAgent测试 |
| `tests/test_strip_layout.py` | 新建 | StripLayoutAgent测试 |
| `tests/test_strip_service.py` | 新建 | StripService测试 |
| `tests/test_strip_api.py` | 新建 | API端点测试 |

---

### Task 1: 数据模型 - QuickStrip 和 StripPanel

**Files:**
- Modify: `app/db/models.py`
- Test: `tests/test_strip_models.py`

**Interfaces:**
- Consumes: 无（纯模型定义）
- Produces: `class QuickStrip(Base)`, `class StripPanel(Base)`，供后续 Service/API 使用

- [ ] **Step 1: 检查现有 models.py 末尾，确认插入位置**

读取 `app/db/models.py` 末尾，找到 Comic/PromptVersion 模型之后的位置。

Run: `pytest tests/ -v --co -q 2>&1 | Select-Object -Last 5`（确认现有测试可收集）

- [ ] **Step 2: 在 app/db/models.py 末尾添加 QuickStrip 和 StripPanel 模型**

在现有模型之后追加：

```python
class QuickStrip(Base):
    __tablename__ = "quick_strips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, default="")
    status = Column(String(20), nullable=False, default="processing")
    image_path = Column(String(500), nullable=False, default="")
    grid_cols = Column(Integer, nullable=False, default=2)
    grid_rows = Column(Integer, nullable=False, default=3)
    error_msg = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, server_default=func.now())

    panels = relationship("StripPanel", back_populates="strip", lazy="selectin", order_by="StripPanel.panel_num")


class StripPanel(Base):
    __tablename__ = "strip_panels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strip_id = Column(Integer, ForeignKey("quick_strips.id"), nullable=False)
    panel_num = Column(Integer, nullable=False)
    image_path = Column(String(500), nullable=False, default="")
    image_prompt = Column(Text, nullable=False, default="")
    dialogue = Column(Text, nullable=False, default="")
    narration = Column(Text, nullable=False, default="")
    source_photo_idx = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())

    strip = relationship("QuickStrip", back_populates="panels")
```

- [ ] **Step 3: 检查 Story 模型是否需要关系（不需要，QuickStrip独立于Story）**

不需要修改 Story 模型，QuickStrip 是独立功能，不关联 Story/Album。

- [ ] **Step 4: 编写模型测试**

创建 `tests/test_strip_models.py`：

```python
import pytest
from app.db.models import QuickStrip, StripPanel


def test_quick_strip_defaults():
    qs = QuickStrip()
    assert qs.status == "processing"
    assert qs.grid_cols == 2
    assert qs.grid_rows == 3
    assert qs.title == ""
    assert qs.image_path == ""
    assert qs.error_msg == ""


def test_strip_panel_defaults():
    sp = StripPanel(panel_num=1)
    assert sp.panel_num == 1
    assert sp.image_path == ""
    assert sp.dialogue == ""
    assert sp.source_photo_idx == 0


def test_quick_strip_relationship():
    qs = QuickStrip(id=1, title="测试条漫")
    panel = StripPanel(strip_id=1, panel_num=1, dialogue="你好")
    qs.panels = [panel]
    assert len(qs.panels) == 1
    assert qs.panels[0].dialogue == "你好"
    assert panel.strip == qs
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_strip_models.py -v`
Expected: 3 passed

- [ ] **Step 6: 提交**

```bash
git add app/db/models.py tests/test_strip_models.py
git commit -m "feat: add QuickStrip and StripPanel models for quick comic strip generation"
```

---

### Task 2: StripScriptAgent - 单次LLM生成6格脚本

**Files:**
- Create: `app/agents/strip_script_agent.py`
- Modify: `app/agents/__init__.py`
- Test: `tests/test_strip_script.py`

**Interfaces:**
- Consumes: `llm_provider` (OpenAIProvider，需有 `chat_json(messages, temperature=...)` 方法，和其他Agent一致)
- Produces:
  - `class StripScriptAgent(BaseAgent)`
  - `async def run(self, **kwargs) -> Dict[str, Any]`
  - 输入 kwargs: `photo_analyses: List[dict]`（每张照片的视觉分析结果列表）
  - 返回: `{"title": str, "character_desc": str, "panels": [{"panel_num": int, "source_photo_idx": int, "description": str, "dialogue": str, "shot": str}]}`

- [ ] **Step 1: 先看现有 Agent 的基类和模式**

读取 `app/agents/base.py` 和一个现有 Agent（如 `app/agents/dialogue_agent.py`）确认接口模式。

- [ ] **Step 2: 编写失败测试**

创建 `tests/test_strip_script.py`：

```python
import pytest
from unittest.mock import MagicMock, AsyncMock
from app.agents.strip_script_agent import StripScriptAgent


def _make_mock_provider(result=None):
    provider = MagicMock()
    provider.chat_json = MagicMock(return_value=result or {
        "title": "测试标题",
        "character_desc": "戴眼镜的男生",
        "panels": [
            {"panel_num": i+1, "source_photo_idx": 0, "description": f"格{i+1}描述", "dialogue": f"对白{i+1}", "shot": "Medium"}
            for i in range(6)
        ]
    })
    return provider


def test_strip_script_returns_6_panels():
    provider = _make_mock_provider()
    agent = StripScriptAgent(llm_provider=provider)
    import asyncio
    result = asyncio.run(agent.run(photo_analyses=[
        {"scene_desc": "餐厅", "people": [{"traits": "戴眼镜男生"}], "action": "吃饭"}
    ]))
    assert result["title"] == "测试标题"
    assert result["character_desc"] == "戴眼镜的男生"
    assert len(result["panels"]) == 6
    for p in result["panels"]:
        assert "panel_num" in p
        assert "description" in p
        assert "dialogue" in p
        assert p["dialogue"] != ""


def test_strip_script_fallback_on_llm_failure():
    provider = MagicMock()
    provider.chat_json = MagicMock(side_effect=Exception("LLM error"))
    agent = StripScriptAgent(llm_provider=provider)
    import asyncio
    result = asyncio.run(agent.run(photo_analyses=[
        {"scene_desc": "公园", "action": "散步"},
        {"scene_desc": "咖啡店", "action": "喝咖啡"},
    ]))
    assert len(result["panels"]) == 6
    assert result["title"] != ""


def test_strip_script_fallback_on_invalid_json():
    provider = MagicMock()
    provider.chat_json = MagicMock(return_value={"panels": [{"panel_num": 1}]})
    agent = StripScriptAgent(llm_provider=provider)
    import asyncio
    result = asyncio.run(agent.run(photo_analyses=[
        {"scene_desc": "办公室", "action": "写代码"},
    ]))
    assert len(result["panels"]) == 6


def test_build_prompt_contains_photo_info():
    provider = _make_mock_provider()
    agent = StripScriptAgent(llm_provider=provider)
    analyses = [
        {"scene_desc": "餐厅里摆满菜", "people": [{"traits": "戴黑框眼镜男生"}], "action": "叹气", "emotion": "疲惫"},
    ]
    prompt = agent._build_user_prompt(analyses)
    assert "餐厅" in prompt
    assert "眼镜" in prompt
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_strip_script.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'app.agents.strip_script_agent'）

- [ ] **Step 4: 实现 StripScriptAgent**

创建 `app/agents/strip_script_agent.py`：

```python
import json
import logging
from typing import Any, Dict, List

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

STRIP_SCRIPT_SYSTEM_PROMPT = """你是一个专业条漫编剧。根据用户提供的照片场景分析，创作一个2列3行（共6格）的四格/六格漫画风格条漫脚本。

核心要求：
1. 从照片中提取核心人物的外貌特征（发型、服装、配饰如眼镜等），在character_desc中详细描述，确保6格画风一致
2. 每格必须包含画面描述（中文，50字以内，包含人物外貌、动作、表情、场景）
3. 每格必须包含对白（中文，30字以内，幽默/温馨/生活化，要有梗）
4. 6格要有起承转合：开头格(1-2)铺垫场景，中间格(3-4)发展/转折，结尾格(5-6)有punchline/温馨收尾
5. shot类型：Wide(全景)/Medium(中景)/Close-up(特写)/Action(动作)/Ending(结尾)
6. 风格：半色调网点漫画风，粗黑线条，温暖色调

输出严格JSON格式，不要输出任何其他文字：
{
  "title": "条漫标题（10字以内）",
  "character_desc": "人物外貌统一描述，用于所有格生图时保持一致性",
  "panels": [
    {
      "panel_num": 1,
      "source_photo_idx": 0,
      "description": "画面描述",
      "dialogue": "对白文字",
      "shot": "Medium"
    }
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
            scene = a.get("scene_desc", "")
            people = a.get("people", [])
            people_str = ""
            if people:
                traits = [p.get("traits", "") for p in people if p.get("traits")]
                people_str = "；".join(traits)
            action = a.get("action", "")
            emotion = a.get("emotion", "")
            location = a.get("location", "")
            clothing = a.get("clothing", [])
            clothing_str = "、".join(clothing) if clothing else ""
            parts.append(
                f"照片{i+1}：场景：{scene}；地点：{location}；"
                f"人物：{people_str}；穿着：{clothing_str}；"
                f"动作：{action}；情绪：{emotion}"
            )
        parts.append("\n请创作6格幽默/温馨的条漫，确保人物外貌在所有格中一致。")
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
        panels = []
        default_descs = [
            "主角出现在场景中，表情轻松愉快",
            "主角发现了什么有趣的事情，眼睛一亮",
            "主角兴奋地做出反应，动作夸张",
            "事情出现了意想不到的转折",
            "主角露出无奈或惊讶的表情",
            "主角会心一笑，温馨收尾",
        ]
        default_dialogues = [
            "今天天气真好啊~",
            "咦？那是什么？",
            "太棒了！",
            "等等，这不对吧……",
            "啊这……",
            "哈哈，生活就是这样嘛~",
        ]
        for i in range(6):
            src_idx = min(i, len(photo_analyses) - 1) if photo_analyses else 0
            panels.append({
                "panel_num": i + 1,
                "source_photo_idx": src_idx,
                "description": default_descs[i],
                "dialogue": default_dialogues[i],
                "shot": "Medium",
            })
        return {
            "title": "生活小确幸",
            "character_desc": "一个年轻的普通人",
            "panels": panels,
        }
```

- [ ] **Step 5: 在 app/agents/__init__.py 中导出**

读取现有 `app/agents/__init__.py`，在现有导出后追加 `StripScriptAgent`：

```python
from .strip_script_agent import StripScriptAgent
from .strip_layout_agent import StripLayoutAgent
```
（注意：StripLayoutAgent 将在 Task 3 创建，这里先添加 StripScriptAgent 的导入，StripLayoutAgent 的导入在 Task 3 时再添加。）

先读取现有文件再精确编辑。

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
- Test: `tests/test_strip_layout.py`

**Interfaces:**
- Consumes: 无外部API依赖，纯PIL本地处理
- Produces:
  - `class StripLayoutAgent(BaseAgent)`
  - `async def run(self, **kwargs) -> Dict[str, Any]`
  - 输入 kwargs: `panels: List[dict]`（每个panel含 image_path, dialogue）, `strip_dir: str`, `character_desc: str`
  - 返回: `{"image_path": str, "width": int, "height": int}`
  - 关键常量: PANEL_W=864, PANEL_H=1152, COLS=2, ROWS=3, BORDER=6, OUTER_BORDER=8

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_strip_layout.py`：

```python
import os
import pytest
from PIL import Image
from app.agents.strip_layout_agent import StripLayoutAgent


@pytest.fixture
def sample_panels_with_images(tmp_path):
    panels = []
    for i in range(6):
        img = Image.new("RGB", (864, 1152), color=(200 + i * 10, 180, 150))
        p = tmp_path / f"panel_{i+1}.png"
        img.save(str(p))
        panels.append({
            "panel_num": i + 1,
            "image_path": str(p),
            "dialogue": f"对白内容{i+1}",
        })
    return panels, tmp_path


def test_compose_creates_long_image(sample_panels_with_images):
    panels, tmp_path = sample_panels_with_images
    agent = StripLayoutAgent()
    import asyncio
    result = asyncio.run(agent.run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])
    img = Image.open(result["image_path"])
    expected_w = 864 * 2 + 6 + 8 * 2
    expected_h = 1152 * 3 + 6 * 2 + 8 * 2
    assert img.width == expected_w
    assert img.height == expected_h


def test_compose_handles_missing_images(sample_panels_with_images):
    panels, tmp_path = sample_panels_with_images
    panels[2]["image_path"] = "/nonexistent/path.png"
    agent = StripLayoutAgent()
    import asyncio
    result = asyncio.run(agent.run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])
    img = Image.open(result["image_path"])
    assert img.width > 0


def test_compose_with_empty_dialogue(sample_panels_with_images):
    panels, tmp_path = sample_panels_with_images
    panels[0]["dialogue"] = ""
    agent = StripLayoutAgent()
    import asyncio
    result = asyncio.run(agent.run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])


def test_bubble_drawing(sample_panels_with_images):
    panels, tmp_path = sample_panels_with_images
    agent = StripLayoutAgent()
    bubble = agent._create_bubble("测试对白文字", 400)
    assert bubble.width > 0
    assert bubble.height > 0
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
                    panel_img = Image.open(img_path).convert("RGB")
                    panel_img = panel_img.resize((PANEL_W, PANEL_H), Image.LANCZOS)
                    canvas.paste(panel_img, (x, y))
                except Exception as e:
                    logger.warning(f"Failed to load panel image {img_path}: {e}")
                    self._draw_placeholder(draw, x, y, panel.get("description", ""))
            else:
                self._draw_placeholder(draw, x, y, panel.get("description", ""))

            draw.rectangle(
                [x, y, x + PANEL_W - 1, y + PANEL_H - 1],
                outline=(0, 0, 0), width=4
            )

            dialogue = panel.get("dialogue", "")
            if dialogue:
                bubble = self._create_bubble(dialogue, PANEL_W - 80, font)
                if bubble:
                    bx = x + 40
                    by = y + 30
                    if by + bubble.height > y + PANEL_H - 20:
                        by = y + PANEL_H - bubble.height - 20
                    canvas.paste(bubble, (bx, by), bubble)

        self._draw_watermark(draw, total_w, total_h, font)

        canvas.save(output_path, "PNG", quality=95)
        logger.info(f"Strip composed: {output_path} ({total_w}x{total_h})")

        return {
            "image_path": output_path,
            "width": total_w,
            "height": total_h,
        }

    def _create_bubble(self, text: str, max_width: int, font: ImageFont.ImageFont) -> Image.Image:
        lines = self._wrap_text(text, font, max_width - BUBBLE_PAD * 2)
        if not lines:
            return None

        bbox = font.getbbox("测Ag")
        line_height = bbox[3] - bbox[1] + 8
        text_w = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
        bubble_w = text_w + BUBBLE_PAD * 2
        bubble_h = line_height * len(lines) + BUBBLE_PAD * 2 + 10

        bubble = Image.new("RGBA", (bubble_w, bubble_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)
        bd.rounded_rectangle(
            [0, 0, bubble_w - 1, bubble_h - 15],
            radius=BUBBLE_RADIUS,
            fill=(255, 255, 255, 240),
            outline=(0, 0, 0, 255),
            width=3,
        )
        bd.polygon([
            (bubble_w // 2 - 12, bubble_h - 15),
            (bubble_w // 2 + 12, bubble_h - 15),
            (bubble_w // 2, bubble_h),
        ], fill=(255, 255, 255, 240), outline=(0, 0, 0, 255))
        bd.line([
            (bubble_w // 2 - 12, bubble_h - 15),
            (bubble_w // 2, bubble_h),
            (bubble_w // 2 + 12, bubble_h - 15),
        ], fill=(0, 0, 0, 255), width=3)

        y_text = BUBBLE_PAD
        for line in lines:
            line_bbox = font.getbbox(line)
            lw = line_bbox[2] - line_bbox[0]
            bd.text(((bubble_w - lw) // 2, y_text), line, fill=(0, 0, 0, 255), font=font)
            y_text += line_height

        return bubble

    @staticmethod
    def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
        return lines[:4]

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    @staticmethod
    def _draw_placeholder(draw: ImageDraw.ImageDraw, x: int, y: int, desc: str):
        draw.rectangle([x, y, x + PANEL_W - 1, y + PANEL_H - 1], fill=(240, 240, 240))

    def _draw_watermark(self, draw: ImageDraw.ImageDraw, w: int, h: int, font: ImageFont.ImageFont):
        try:
            bbox = font.getbbox(WATERMARK)
            fw = bbox[2] - bbox[0]
            fh = bbox[3] - bbox[1]
            draw.text((w - fw - 20, h - fh - 12), WATERMARK, fill=(180, 180, 180), font=font)
        except Exception:
            pass
```

- [ ] **Step 4: 更新 app/agents/__init__.py，添加 StripLayoutAgent 导入**

确保导入行存在：`from .strip_layout_agent import StripLayoutAgent`

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_strip_layout.py -v`
Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
git add app/agents/strip_layout_agent.py app/agents/__init__.py tests/test_strip_layout.py
git commit -m "feat: add StripLayoutAgent - PIL compose 2x3 strip with speech bubbles"
```

---

### Task 4: VisionAgent/QianfanImageProvider async支持 + StripService 编排

**Files:**
- Modify: `app/agents/vision_agent.py`（添加 async 方法，因为 run() 里调用的是同步 analyze_image，会阻塞事件循环）
- Modify: `app/llm/qianfan_image_provider.py`（添加 async 包装方法）
- Create: `app/services/strip_service.py`
- Test: `tests/test_strip_service.py`

**Interfaces:**
- Consumes:
  - `VisionAgent.run_async(photo_id, image_path)` (新增async方法)
  - `QianfanImageProvider.generate_async(prompt, size)` (新增async方法)
  - `StripScriptAgent.run(photo_analyses=...)`
  - `StripLayoutAgent.run(panels=..., strip_dir=...)`
  - `AsyncSession` for DB
  - `settings.COMICS_DIR`
- Produces:
  - `class StripService`
  - `async def generate_strip(self, photo_paths: List[str], session: AsyncSession) -> dict`
  - 返回: `{"strip_id": int, "title": str, "image_path": str, "panels": [...]}`

- [ ] **Step 1: 为 VisionAgent 添加 async 方法**

在 `app/agents/vision_agent.py` 的 VisionAgent 类中添加 run_async 方法（现有 run() 是 async 但内部调用同步 analyze_image，需要用 run_in_executor 包装以避免阻塞）：

在现有 `run` 方法后添加：

```python
    async def run_async(self, photo_id: int, image_path: str) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_analyze, photo_id, image_path)

    def _sync_analyze(self, photo_id: int, image_path: str) -> Dict[str, Any]:
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
            return {
                "photo_id": photo_id,
                "scene_desc": "", "weather": "", "location": "",
                "action": "", "emotion": "", "clothing": [], "people": [], "objects": [],
            }
```

- [ ] **Step 2: 为 QianfanImageProvider 添加 async 包装**

在 `app/llm/qianfan_image_provider.py` 类末尾添加：

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

- [ ] **Step 3: 编写 StripService 失败测试**

创建 `tests/test_strip_service.py`：

```python
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image
from app.services.strip_service import StripService


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _get_session_override():
    return None


@pytest.mark.asyncio
async def test_generate_strip_creates_record_and_returns_result(tmp_path, mock_session):
    photo = tmp_path / "test.jpg"
    photo.write_bytes(b"fake_image_data")

    vision_agent = MagicMock()
    vision_agent.run_async = AsyncMock(return_value={
        "scene_desc": "餐厅吃饭", "people": [{"traits": "戴眼镜男生"}],
        "action": "用餐", "emotion": "开心", "location": "餐厅", "clothing": ["蓝衬衫"],
    })

    script_agent = MagicMock()
    script_agent.run = AsyncMock(return_value={
        "title": "聚餐日常",
        "character_desc": "戴黑框眼镜穿蓝衬衫的男生",
        "panels": [
            {"panel_num": i+1, "source_photo_idx": 0, "description": f"格{i+1}", "dialogue": f"对白{i+1}", "shot": "Medium"}
            for i in range(6)
        ]
    })

    image_provider = MagicMock()
    image_provider.generate_async = AsyncMock(return_value={"urls": ["https://example.com/img.png"]})

    layout_agent = MagicMock()
    layout_agent.run = AsyncMock(return_value={
        "image_path": str(tmp_path / "strip.png"),
        "width": 1750, "height": 3480,
    })

    with patch("app.services.strip_service.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.content = b"fake"
        mock_resp.raise_for_status = MagicMock()
        mock_req.get.return_value = mock_resp
        with patch("app.services.strip_service.Image.new"):
            svc = StripService(
                vision_agent=vision_agent,
                script_agent=script_agent,
                image_provider=image_provider,
                layout_agent=layout_agent,
                comics_dir=str(tmp_path),
            )
            mock_session.refresh.side_effect = lambda x: setattr(x, "id", 1) or setattr(x, "status", "completed")
            result = await svc.generate_strip(
                photo_paths=[str(photo), str(tmp_path / "test2.jpg")],
                session=mock_session,
            )
    assert result["strip_id"] == 1
    assert result["title"] == "聚餐日常"
    assert result["status"] == "completed"
    assert result["image_path"] == str(tmp_path / "strip.png")
    assert len(result["panels"]) == 6
    mock_session.add.assert_called()
    mock_session.commit.assert_called()


@pytest.mark.asyncio
async def test_generate_strip_validates_photo_count(tmp_path, mock_session):
    svc = StripService(
        vision_agent=MagicMock(), script_agent=MagicMock(),
        image_provider=MagicMock(), layout_agent=MagicMock(),
        comics_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="2-6"):
        await svc.generate_strip(photo_paths=[str(tmp_path / "a.jpg")], session=mock_session)
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
import uuid
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.agents.strip_layout_agent import PANEL_H, PANEL_W
from app.db.models import QuickStrip, StripPanel

logger = logging.getLogger(__name__)


class StripService:
    def __init__(
        self,
        vision_agent: BaseAgent,
        script_agent: BaseAgent,
        image_provider,
        layout_agent: BaseAgent,
        comics_dir: str,
    ):
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

        strips_dir = os.path.join(self.comics_dir, "strips", str(strip_id))
        os.makedirs(strips_dir, exist_ok=True)

        try:
            photo_analyses = await self._analyze_photos(photo_paths)

            script_result = await self.script_agent.run(photo_analyses=photo_analyses)
            title = script_result.get("title", "生活条漫")
            character_desc = script_result.get("character_desc", "")
            panels_script = script_result.get("panels", [])[:6]

            strip.title = title

            panel_results = await self._generate_panels(
                panels_script, character_desc, strips_dir
            )

            db_panels = []
            for p in panel_results:
                db_panel = StripPanel(
                    strip_id=strip_id,
                    panel_num=p["panel_num"],
                    image_path=p.get("image_path", ""),
                    image_prompt=p.get("image_prompt", ""),
                    dialogue=p.get("dialogue", ""),
                    source_photo_idx=p.get("source_photo_idx", 0),
                )
                db_panels.append(db_panel)
                session.add(db_panel)

            compose_result = await self.layout_agent.run(panels=panel_results, strip_dir=strips_dir)

            strip.status = "completed"
            strip.image_path = compose_result["image_path"]
            await session.commit()
            await session.refresh(strip)

            return {
                "strip_id": strip_id,
                "title": title,
                "status": "completed",
                "image_path": compose_result["image_path"],
                "panels": [
                    {
                        "panel_num": p["panel_num"],
                        "image_path": p.get("image_path", ""),
                        "dialogue": p.get("dialogue", ""),
                    }
                    for p in panel_results
                ],
            }

        except Exception as e:
            logger.error(f"Strip generation failed for strip {strip_id}: {e}", exc_info=True)
            strip.status = "failed"
            strip.error_msg = str(e)[:500]
            await session.commit()
            raise

    async def _analyze_photos(self, photo_paths: List[str]) -> List[Dict[str, Any]]:
        tasks = [
            self.vision_agent.run_async(photo_id=i, image_path=path)
            for i, path in enumerate(photo_paths)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        analyses = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"Photo {i} analysis failed: {r}")
                analyses.append({
                    "scene_desc": "生活场景", "people": [], "action": "",
                    "emotion": "", "location": "", "clothing": [], "objects": [],
                })
            else:
                analyses.append(r)
        return analyses

    async def _generate_panels(
        self,
        panels_script: List[Dict[str, Any]],
        character_desc: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(3)

        async def gen_one(panel_info: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                panel_num = panel_info["panel_num"]
                desc = panel_info.get("description", "")
                dialogue = panel_info.get("dialogue", "")
                prompt = self._build_image_prompt(character_desc, desc, dialogue)
                try:
                    result = await self.image_provider.generate_async(prompt, size="864x1152")
                    if result and result.get("urls"):
                        url = result["urls"][0]
                        local_path = os.path.join(output_dir, f"panel_{panel_num}.png")
                        ok = await self._download_image_async(url, local_path)
                        if ok:
                            panel_info["image_path"] = local_path
                            panel_info["image_prompt"] = prompt
                            return panel_info
                except Exception as e:
                    logger.warning(f"Panel {panel_num} generation failed: {e}")

                panel_info["image_path"] = ""
                panel_info["image_prompt"] = prompt
                placeholder = os.path.join(output_dir, f"panel_{panel_num}.png")
                self._create_placeholder(placeholder, desc)
                panel_info["image_path"] = placeholder
                return panel_info

        tasks = [gen_one(p) for p in panels_script]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _build_image_prompt(character_desc: str, desc: str, dialogue: str) -> str:
        return (
            f"{character_desc}。{desc}。"
            f"漫画风格，粗黑线稿，半色调网点，温暖色调，高质量插画，"
            f"竖版构图，适合2×3条漫分镜"
        )

    @staticmethod
    async def _download_image_async(url: str, local_path: str) -> bool:
        import asyncio
        loop = asyncio.get_event_loop()
        def _download():
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                return True
            except Exception as e:
                logger.warning(f"Download failed {url}: {e}")
                return False
        return await loop.run_in_executor(None, _download)

    @staticmethod
    def _create_placeholder(path: str, desc: str):
        img = Image.new("RGB", (PANEL_W, PANEL_H), (245, 245, 245))
        img.save(path, "PNG")
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_strip_service.py -v`
Expected: 测试通过（可能需要调整 mock 细节，如果失败则根据错误修复）

- [ ] **Step 7: 提交**

```bash
git add app/llm/qianfan_provider.py app/llm/qianfan_image_provider.py app/services/strip_service.py tests/test_strip_service.py
git commit -m "feat: add StripService orchestrator with parallel photo analysis and image generation"
```

---

### Task 5: API端点 - strips router

**Files:**
- Create: `app/api/strips.py`
- Test: `tests/test_strip_api.py`

**Interfaces:**
- Consumes: StripService, get_session, get_settings
- Produces:
  - `POST /api/v1/strip/generate` - multipart上传照片，同步等待返回结果
  - `GET /api/v1/strip/{strip_id}` - 查询条漫状态
  - `GET /api/v1/strip/{strip_id}/image` - 返回长图文件

- [ ] **Step 1: 编写API失败测试**

创建 `tests/test_strip_api.py`：

```python
import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_test_image_bytes():
    img = Image.new("RGB", (100, 100), (200, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _get_app(monkeypatch, tmp_path):
    from app.api import strips as strips_mod
    from app.db.database import get_session
    from app.db.models import QuickStrip, StripPanel, Base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async def init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    import asyncio
    asyncio.run(init())

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_session():
        async with session_factory() as s:
            yield s

    async def fake_generate_strip(photo_paths, session):
        return {
            "strip_id": 1, "title": "测试", "status": "completed",
            "image_path": str(tmp_path / "strip.png"),
            "panels": [{"panel_num": 1, "image_path": "", "dialogue": "你好"}],
        }

    strips_mod.get_session_override = override_get_session

    class FakeSvc:
        async def generate_strip(self, photo_paths, session):
            return await fake_generate_strip(photo_paths, session)

    def fake_build():
        return FakeSvc()

    monkeypatch.setattr(strips_mod, "_build_strip_service", fake_build)

    from fastapi import FastAPI
    app = FastAPI()
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(strips_mod.strips_router)
    return app


def test_generate_strip_endpoint(monkeypatch, tmp_path):
    app = _get_app(monkeypatch, tmp_path)
    client = TestClient(app)
    files = [("files", ("test.jpg", _make_test_image_bytes(), "image/jpeg"))]
    resp = client.post("/api/v1/strip/generate", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == 0
    assert data["data"]["status"] == "completed"
    assert data["data"]["strip_id"] == 1


def test_generate_strip_too_few_photos(monkeypatch, tmp_path):
    app = _get_app(monkeypatch, tmp_path)
    client = TestClient(app)
    resp = client.post("/api/v1/strip/generate", files=[])
    assert resp.status_code == 400


def test_get_strip_endpoint(monkeypatch, tmp_path):
    app = _get_app(monkeypatch, tmp_path)
    client = TestClient(app)
    resp = client.get("/api/v1/strip/1")
    assert resp.status_code == 200


def test_static_image_served(monkeypatch, tmp_path):
    img_path = tmp_path / "strips" / "1"
    img_path.mkdir(parents=True, exist_ok=True)
    strip_file = img_path / "strip.png"
    Image.new("RGB", (100, 100), (255, 0, 0)).save(str(strip_file))

    async def fake_get(photo_paths, session):
        return {
            "strip_id": 1, "title": "测试", "status": "completed",
            "image_path": str(strip_file),
            "panels": [],
        }
    app = _get_app(monkeypatch, tmp_path)
    from app.api import strips as strips_mod
    orig_build = strips_mod._build_strip_service
    class FakeSvc2:
        async def generate_strip(self, photo_paths, session):
            return await fake_get(photo_paths, session)
    monkeypatch.setattr(strips_mod, "_build_strip_service", lambda: FakeSvc2())
    client = TestClient(app)
    files = [("files", ("a.jpg", _make_test_image_bytes(), "image/jpeg")), ("files", ("b.jpg", _make_test_image_bytes(), "image/jpeg"))]
    resp = client.post("/api/v1/strip/generate", files=files)
    assert resp.status_code == 200
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
    qianfan = QianfanProvider(
        access_key=settings.QIANFAN_ACCESS_KEY,
        secret_key=settings.QIANFAN_SECRET_KEY,
    )
    qianfan.configure(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

    openai_provider = OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=0.8,
        top_p=settings.LLM_TOP_P,
    )
    image_provider = QianfanImageProvider(
        access_key=settings.QIANFAN_ACCESS_KEY,
        secret_key=settings.QIANFAN_SECRET_KEY,
    )
    return StripService(
        vision_agent=VisionAgent(qianfan_provider=qianfan),
        script_agent=StripScriptAgent(llm_provider=openai_provider),
        image_provider=image_provider,
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

    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    photo_paths = []
    upload_dir = os.path.join(get_settings().UPLOAD_DIR, "strip_photos")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        for f in files:
            if f.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail=f"不支持的格式: {f.content_type}")
            ext = os.path.splitext(f.filename or "photo.jpg")[1] or ".jpg"
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            fname = f"{uuid.uuid4().hex}{ext}"
            fpath = os.path.join(upload_dir, fname)
            with open(fpath, "wb") as out:
                shutil.copyfileobj(f.file, out)
            photo_paths.append(fpath)

        result = await svc.generate_strip(photo_paths=photo_paths, session=session)
        return APIResponse(
            code=0,
            message="success",
            data={
                "strip_id": result["strip_id"],
                "status": result["status"],
                "title": result["title"],
                "image_url": f"/api/v1/strip/{result['strip_id']}/image",
                "panels": result.get("panels", []),
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Strip generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"条漫生成失败: {str(e)[:200]}")


@strips_router.get("/{strip_id}", response_model=APIResponse[dict])
async def get_strip(
    strip_id: int,
    session: AsyncSession = Depends(_get_session),
):
    result = await session.execute(
        select(QuickStrip).where(QuickStrip.id == strip_id)
    )
    strip = result.scalar_one_or_none()
    if not strip:
        raise HTTPException(status_code=404, detail="条漫不存在")
    return APIResponse(
        code=0,
        data={
            "strip_id": strip.id,
            "title": strip.title,
            "status": strip.status,
            "image_url": f"/api/v1/strip/{strip.id}/image" if strip.status == "completed" else None,
            "error_msg": strip.error_msg,
            "created_at": strip.created_at.isoformat() if strip.created_at else None,
        },
    )


@strips_router.get("/{strip_id}/image")
async def get_strip_image(
    strip_id: int,
    session: AsyncSession = Depends(_get_session),
):
    result = await session.execute(
        select(QuickStrip).where(QuickStrip.id == strip_id)
    )
    strip = result.scalar_one_or_none()
    if not strip or not strip.image_path or not os.path.exists(strip.image_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(strip.image_path, media_type="image/png")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_strip_api.py -v`
Expected: 所有测试通过

- [ ] **Step 5: 提交**

```bash
git add app/api/strips.py tests/test_strip_api.py app/agents/vision_agent.py
git commit -m "feat: add strip API endpoints - upload photos, generate and serve strip image"
```

---

### Task 6: 主文件集成 + 静态文件挂载

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: strips_router
- Produces: 挂载路由，启动时创建目录

- [ ] **Step 1: 修改 main.py 挂载 strips 路由**

读取 `main.py`，在现有 import 区域添加：

```python
from app.api.strips import strips_router
```

在 `app.include_router(health_router)` 之前添加：

```python
app.include_router(strips_router)
```

- [ ] **Step 2: 添加 comics/strips 目录自动创建**

在 startup_event 中，在种子数据之后添加：

```python
    try:
        settings = get_settings()
        os.makedirs(settings.COMICS_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.COMICS_DIR, "strips"), exist_ok=True)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        logger.info("Storage directories ensured")
    except Exception as e:
        logger.error(f"Directory creation failed: {e}")
```

需要确保 `import os` 在顶部。

- [ ] **Step 3: 验证所有测试仍然通过**

Run: `pytest tests/ -v --ignore=tests/test_comic_integration.py -x 2>&1 | Select-Object -Last 30`
Expected: 所有单元测试通过

- [ ] **Step 4: 提交**

```bash
git add main.py
git commit -m "feat: integrate strips router and ensure storage directories on startup"
```

---

### Task 7: 集成测试 + 回归

**Files:**
- Create: `tests/test_strip_integration.py`（可选，标记为 integration）

- [ ] **Step 1: 编写集成测试**

创建 `tests/test_strip_integration.py`：

```python
import os
import pytest
from PIL import Image


@pytest.mark.integration
@pytest.mark.asyncio
async def test_strip_script_agent_real_llm():
    from app.llm.openai_provider import OpenAIProvider
    from app.config.config import get_settings
    from app.agents.strip_script_agent import StripScriptAgent

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        pytest.skip("No API key configured")

    provider = OpenAIProvider(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=0.8,
        top_p=settings.LLM_TOP_P,
    )
    agent = StripScriptAgent(llm_provider=provider)
    result = await agent.run(photo_analyses=[
        {"scene_desc": "程序员在餐厅聚餐", "people": [{"traits": "戴黑框眼镜男生"}], "action": "叹气", "emotion": "疲惫", "location": "餐厅", "clothing": ["浅蓝色衬衫"]},
        {"scene_desc": "桌上摆满了菜", "people": [], "action": "吃饭", "emotion": "饥饿", "location": "餐厅", "clothing": []},
    ])
    assert len(result["panels"]) == 6
    assert result["title"]
    for p in result["panels"]:
        assert p["dialogue"]
        assert p["description"]


@pytest.mark.integration
def test_strip_layout_real_images(tmp_path):
    from app.agents.strip_layout_agent import StripLayoutAgent
    panels = []
    for i in range(6):
        img = Image.new("RGB", (864, 1152), (180 + i * 10, 160, 140))
        p = tmp_path / f"p{i}.png"
        img.save(str(p))
        panels.append({"panel_num": i+1, "image_path": str(p), "dialogue": f"测试对白{i+1}哈哈"})

    import asyncio
    agent = StripLayoutAgent()
    result = asyncio.run(agent.run(panels=panels, strip_dir=str(tmp_path)))
    assert os.path.exists(result["image_path"])
    out = Image.open(result["image_path"])
    assert out.width == 864 * 2 + 6 + 16
    assert out.height == 1152 * 3 + 12 + 16
```

- [ ] **Step 2: 运行集成测试**

Run: `pytest tests/test_strip_integration.py -v -m integration`
Expected: 通过（需要有效API key）

- [ ] **Step 3: 运行全量回归测试**

Run: `pytest tests/ -v --ignore=tests/test_comic_integration.py 2>&1 | Select-Object -Last 40`
Expected: 所有单元测试通过

- [ ] **Step 4: 提交**

```bash
git add tests/test_strip_integration.py
git commit -m "test: add integration tests for strip script and layout agents"
```
