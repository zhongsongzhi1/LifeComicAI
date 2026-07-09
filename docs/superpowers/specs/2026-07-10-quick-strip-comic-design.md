# 快速条漫生成功能设计文档

## 背景与目标

**用户痛点**：现有漫画流水线（Story→多页PDF）耗时 5-6 分钟，移动端用户无法接受。用户期望"拍几张照片上传，1分钟内出一张6宫格条漫长图"。

**核心目标**：
- 输入：用户直接上传 2-6 张生活照片
- 输出：单张 2×3 网格条漫长图（含对白气泡、粗边框、漫画风格）
- 耗时：**30-60 秒**完成
- 体验：API 一次请求返回结果，移动端显示 loading 即可

**非目标**：
- 不做流式逐格推送（简单的 loading 等待即可）
- 不做 PDF（只出长图）
- 不做风格选择（默认统一漫画风，后续可加）
- 不做用户编辑对白（后续迭代）

---

## 架构设计

### 流水线对比

```
现有 PDF 流水线（5-6min，串行）:
  Storyboard(37s) → Dialogue(54s) → Director(94s) → Comic串行逐页(55s×N) → Layout(PDF)

快速条漫流水线（40-60s，并行优化）:
  ① 保存照片 (1s)
  ② Vision 并行分析照片 (Qwen VL, ~20s, 并行×N)
  ③ StripScript Agent (单次LLM, ~25s) — 一次性生成6格分镜+对白+角色描述
  ④ 并行生图 (千帆, ~30-40s, asyncio.gather×6)
  ⑤ PIL 后处理拼长图+气泡 (2s)
```

### 关键速度优化

| 优化项 | 原方案 | 新方案 | 节省 |
|--------|--------|--------|------|
| LLM 调用次数 | 3次串行（分镜+对白+导演） | **1次**合并调用 | ~2分钟 |
| 生图方式 | 串行逐页 | **asyncio.gather 并行6张** | ~4分钟（6张串行vs并行） |
| 风格应用 | Director Agent 单独调整 | prompt 中内嵌风格指令 | ~90s |
| PDF/Layout | ReportLab 生成PDF | PIL 直接拼长图 | ~5s |

### 组件图

```
POST /api/v1/strip/generate (multipart/form-data: files)
    │
    ▼
StripService.generate_strip(files, session)
    │
    ├─ 1. 保存照片到 uploads/strip/{strip_id}/
    ├─ 2. 创建 QuickStrip 记录（status=processing）
    ├─ 3. 并行调用 Qwen VL 分析每张照片 → photo_analyses
    ├─ 4. StripScriptAgent.run(photo_analyses) → strip_script
    │      （单次LLM：输出6格分镜+对白+统一角色外貌描述）
    ├─ 5. 并行生图 asyncio.gather(6 tasks) → 6张本地图片路径
    ├─ 6. StripLayoutAgent.compose(pages) → 长图路径
    │      （PIL：粗边框+对白气泡+网点纹理+拼2×3网格）
    └─ 7. 更新 QuickStrip（status=completed, image_path=...）→ 返回结果
```

---

## 数据模型

### 新增表：quick_strips

```sql
CREATE TABLE quick_strips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       VARCHAR(255) DEFAULT '',        -- 自动生成的标题
    status      VARCHAR(20) DEFAULT 'processing', -- processing/completed/failed
    image_path  VARCHAR(500) DEFAULT '',       -- 最终长图本地路径
    grid_cols   INTEGER DEFAULT 2,            -- 列数
    grid_rows   INTEGER DEFAULT 3,            -- 行数
    error_msg   TEXT DEFAULT '',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 新增表：strip_panels

```sql
CREATE TABLE strip_panels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    strip_id    INTEGER NOT NULL REFERENCES quick_strips(id),
    panel_num   INTEGER NOT NULL,             -- 1-6
    image_path  VARCHAR(500) DEFAULT '',      -- 单格图片本地路径
    image_prompt TEXT DEFAULT '',
    dialogue    TEXT DEFAULT '',              -- 对白文字
    narration   TEXT DEFAULT '',
    source_photo_idx INTEGER DEFAULT 0,      -- 对应第几张上传的照片（0-indexed）
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### strip_script 数据结构（LLM 输出）

```json
{
  "title": "程序员聚餐的崩溃瞬间",
  "character_desc": "一个戴黑框眼镜、穿浅蓝色衬衫的年轻程序员男生，黑色短发",
  "panels": [
    {
      "panel_num": 1,
      "source_photo_idx": 0,
      "description": "程序员坐在餐桌前，周围摆满了菜，他表情疲惫地叹气",
      "dialogue": "终于能从Bug的海洋里喘口气了……这展会，比写代码还烧脑！",
      "shot": "Medium"
    }
  ]
}
```

---

## Agent 设计

### 1. StripScriptAgent（新增）

**职责**：基于照片分析结果，一次 LLM 调用输出完整的 6 格分镜脚本。

**输入**：`photo_analyses: list[dict]`（每张照片的视觉分析结果）
**输出**：`strip_script: dict`（6格分镜+对白+角色描述+标题）

**System Prompt 要点**：
- 你是一个漫画编剧，根据用户提供的照片场景分析，创作一个6格幽默/温馨条漫
- 必须提取照片中的核心人物外貌特征（发型、服装、眼镜等），在所有格中保持一致
- 每格要有画面描述（中文，50字以内）和对白（中文，40字以内）
- 对白要幽默、生活化、有梗
- 开头格铺垫场景，中间格有发展/笑点，结尾格有 punchline
- 第一格和最后一格最好有呼应
- 画面风格：美式漫画线条+日系上色+半色调网点，粗黑边框感
- 输出严格 JSON 格式

**关键优化**：将 Storyboard + Dialogue + Director 三个 Agent 合并为一个 Prompt，一次调用完成。

### 2. StripComicAgent（复用现有 ComicAgent，调整调用方式）

**职责**：根据分镜脚本生成6张漫画图片。

**调整**：
- 生图 prompt 内嵌风格（不再经过 Director Agent）
- 每张图的 prompt 格式：
  ```
  {character_desc}。{description}。{dialogue_context}。
  美式漫画风格，粗黑线稿，半色调网点，温暖色调，高质量，2×3条漫分镜
  ```
- 使用 `asyncio.gather` 并行生成6张图，每张图独立 retry
- 图片尺寸：竖版 `864x1152`（每格竖版，拼2×3）

### 3. StripLayoutAgent（新增，PIL后处理）

**职责**：将6张图拼成一张2×3长图，叠加对白气泡、粗边框、网点纹理。

**布局规格**（以单格 432×576 px 为例，输出 2×3 = 864×1728）：

```
┌──────────────┬──────────────┐
│   panel 1    │   panel 2    │  ← 粗黑分隔线 (6px)
│  [对白气泡]  │  [对白气泡]  │
├──────────────┼──────────────┤
│   panel 3    │   panel 4    │
│  [对白气泡]  │  [对白气泡]  │
├──────────────┼──────────────┤
│   panel 5    │   panel 6    │
│  [对白气泡]  │  [对白气泡]  │
└──────────────┴──────────────┘
```

**后处理步骤**：
1. 加载6张图，统一缩放到单格尺寸
2. 绘制半色调网点纹理叠加（可选，轻微不透明度 0.08）
3. 用 PIL 绘制对白气泡：
   - 白色圆角矩形（圆角半径 15px，边框 3px 黑色）
   - 小三角尾巴指向说话人方向
   - 中文字体（使用系统黑体/SimHei，文字居中，自动换行）
4. 拼接 2×3 网格，格间粗黑线 6px
5. 外边框 8px 黑色
6. 右下角小字 "LifeComicAI" 水印

---

## API 设计

### POST /api/v1/strip/generate

**Content-Type**: `multipart/form-data`

**请求参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | File[] | 是 | 2-6 张照片（jpg/jpeg/png） |

采用**同步等待**方式（HTTP 请求保持连接直到完成，目标 30-60 秒）：

**成功响应**：
```json
{
  "code": 0,
  "data": {
    "strip_id": 1,
    "status": "completed",
    "title": "程序员聚餐的崩溃瞬间",
    "image_url": "/static/strips/1/strip.png",
    "panels": [
      {"panel_num": 1, "dialogue": "终于能从Bug的海洋里喘口气了...", "image_url": "/static/strips/1/panel_1.png"}
    ]
  }
}
```

错误响应：
```json
{
  "code": 500,
  "message": "图片生成失败，请重试"
}
```

### GET /api/v1/strip/{strip_id}
查询条漫状态（用于用户刷新页面时恢复）。

### GET /api/v1/strip/{strip_id}/image
直接返回长图文件（FileResponse）。

---

## 文件结构

```
app/
├── agents/
│   ├── strip_script_agent.py     # 新增：单次LLM生成6格脚本
│   ├── strip_layout_agent.py     # 新增：PIL拼长图+气泡
│   └── __init__.py               # 修改：导出新Agent
├── services/
│   └── strip_service.py          # 新增：条漫编排服务
├── api/
│   └── strips.py                 # 新增：条漫API端点
├── db/
│   └── models.py                 # 修改：添加 QuickStrip + StripPanel
└── config/
    └── config.py                 # 不需修改（复用COMICS_DIR/UPLOAD_DIR）

tests/
├── test_strip_script_agent.py    # 新增
├── test_strip_layout_agent.py    # 新增
├── test_strip_service.py         # 新增
└── test_strip_api.py             # 新增
```

---

## 视觉分析复用

现有 [VisionAgent](file:///f:/pythonProtect/LifeComicAI/app/agents/vision_agent.py) 使用 Qianfan 分析单张照片。由于项目约束（Qianfan SDK SQLite bug），需改为使用 **DashScope Qwen VL**（HTTP 直调），和现有项目其他视觉分析保持一致。

实际上现有 StoryService 已经通过 `qianfan_provider.analyze_image` 调视觉，但根据项目记忆有 SQLite bug，需要确认是否已有替代方案。如果有现成的 Qwen VL HTTP provider，直接复用。

**照片并行分析**：使用 `asyncio.gather` 同时分析多张照片（但 Qwen VL 是同步 HTTP，需要用 `loop.run_in_executor` 包装）。

---

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| 照片数量 <2 或 >6 | 返回 400 |
| 照片格式不支持 | 返回 400 |
| LLM 生成脚本失败 | 重试1次，仍失败返回500 |
| 部分图片生成失败（<3张失败） | 失败的格子用占位图+该格描述文字代替 |
| 全部图片生成失败 | 返回500 |
| 超时（>90秒） | 返回 timeout 错误 |

---

## 测试策略

### 单元测试（Mock）
1. **StripScriptAgent**: Mock LLM provider，验证输出 JSON 结构正确、包含6格
2. **StripLayoutAgent**: 用 Pillow 生成纯色测试图，验证拼长图文件存在且尺寸正确
3. **StripService**: Mock 所有 Agent，验证数据库记录正确、并行调用正确
4. **API**: Mock service，验证 multipart 上传、错误响应

### 集成测试（真实API）
1. 用 2-3 张测试照片，端到端验证生成条漫长图
2. 验证耗时在 90 秒以内
3. 验证输出图片尺寸正确、气泡文字可见

---

## 实现顺序（Task List）

1. **数据模型**：添加 QuickStrip + StripPanel 表
2. **StripScriptAgent**：单次 LLM 生成6格脚本（TDD）
3. **StripLayoutAgent**：PIL 拼长图+气泡（TDD）
4. **StripService**：编排服务，并行生图+照片分析（TDD）
5. **API端点**：strips router + 静态文件服务
6. **集成测试**：端到端验证
7. **回归测试**：确保不破坏现有功能
