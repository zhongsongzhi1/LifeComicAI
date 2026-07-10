import logging
import math
import os
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PANEL_W = 864
PANEL_H = 1152
COLS = 2
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
        n = len(panels)
        rows = math.ceil(n / COLS)
        total_w = PANEL_W * COLS + BORDER * (COLS - 1) + OUTER_BORDER * 2
        total_h = PANEL_H * rows + BORDER * (rows - 1) + OUTER_BORDER * 2

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(0, 0, 0), width=OUTER_BORDER)
        font = self._load_font(BUBBLE_FONT_SIZE)

        for idx, panel in enumerate(panels):
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
        logger.info(f"Strip composed: {output_path} ({total_w}x{total_h}, {n} panels)")
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
