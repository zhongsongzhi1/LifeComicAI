from typing import Optional


class DialogueCustomizer:
    """简单的对白风格化与长度控制工具。

    功能：
    - 根据 `tone`（语气）添加情感提示或轻微改写
    - 根据 `max_sentences` 截断对白长度
    - 支持可选 `style` 标签附加
    这个实现保持轻量、可测试，并适合作为后续复杂重写的钩子。
    """

    def customize(
        self,
        dialogue: str,
        tone: Optional[str] = None,
        max_sentences: int = 2,
        style: Optional[str] = None,
    ) -> str:
        if not dialogue:
            return ""

        # 简单分句（基于常见句末符号）
        segments = []
        current = []
        for ch in dialogue:
            current.append(ch)
            if ch in ("。", "！", "?", "?", "!", "."):
                seg = "".join(current).strip()
                if seg:
                    segments.append(seg)
                current = []
        if current:
            tail = "".join(current).strip()
            if tail:
                segments.append(tail)

        # 截断到 max_sentences
        trimmed = segments[:max_sentences]
        out = "".join(trimmed).strip()

        # 根据 tone 做轻微改写或附加标注
        if tone:
            tone = tone.lower()
            if tone in ("humorous", "humor", "幽默"):
                out = out + "（略带幽默）"
            elif tone in ("dramatic", "drama", "戏剧性"):
                out = "（紧张）" + out
            elif tone in ("warm", "温情"):
                out = out + "（温柔地）"

        # style 附加为括号标签
        if style:
            out = f"{out}（风格：{style}）"

        return out
