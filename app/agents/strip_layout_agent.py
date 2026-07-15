import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PANEL_W = 864
PANEL_H = 1152
COLS = 2
BORDER = 6
OUTER_BORDER = 8
BUBBLE_PAD = 24
BUBBLE_RADIUS = 20
BUBBLE_FONT_SIZE = 28
BUBBLE_MAX_WIDTH = PANEL_W - 100
BUBBLE_MARGIN = 40
WATERMARK = "LifeComicAI"

# 气泡锚点位置（相对于 panel 的 9 个位置）
ANCHOR_POSITIONS = {
    "top_left": (BUBBLE_MARGIN, BUBBLE_MARGIN),
    "top_center": (PANEL_W // 2, BUBBLE_MARGIN),
    "top_right": (PANEL_W - BUBBLE_MARGIN, BUBBLE_MARGIN),
    "middle_left": (BUBBLE_MARGIN, PANEL_H // 2),
    "center": (PANEL_W // 2, PANEL_H // 2),
    "middle_right": (PANEL_W - BUBBLE_MARGIN, PANEL_H // 2),
    "bottom_left": (BUBBLE_MARGIN, PANEL_H - BUBBLE_MARGIN),
    "bottom_center": (PANEL_W // 2, PANEL_H - BUBBLE_MARGIN),
    "bottom_right": (PANEL_W - BUBBLE_MARGIN, PANEL_H - BUBBLE_MARGIN),
}


class StripLayoutAgent(BaseAgent):
    def __init__(self):
        pass

    async def run(self, **kwargs) -> Dict[str, Any]:
        panels: List[Dict[str, Any]] = kwargs.get("panels", [])
        strip_dir: str = kwargs.get("strip_dir", "")
        photo_paths: List[str] = kwargs.get("photo_paths", [])
        title: str = kwargs.get("title", "")
        os.makedirs(strip_dir, exist_ok=True)

        output_path = os.path.join(strip_dir, "strip.png")
        n = len(panels)
        rows = math.ceil(n / COLS)

        # 顶部区域高度（缩略图 + 标题 + 间距）
        thumb_size = 120
        thumb_gap = 16
        header_padding_top = 40
        header_padding_bottom = 30
        title_height = 50 if title else 0
        header_height = (
            header_padding_top
            + (thumb_size if photo_paths else 0)
            + (title_height + 10 if title else 0)
            + header_padding_bottom
        )

        total_w = PANEL_W * COLS + BORDER * (COLS - 1) + OUTER_BORDER * 2
        total_h = header_height + PANEL_H * rows + BORDER * (rows - 1) + OUTER_BORDER * 2

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=(0, 0, 0), width=OUTER_BORDER)
        font = self._load_font(BUBBLE_FONT_SIZE)
        title_font = self._load_font(36)

        # ===== 绘制顶部缩略图 =====
        current_y = header_padding_top
        if photo_paths:
            num_photos = len(photo_paths)
            total_thumbs_w = num_photos * thumb_size + (num_photos - 1) * thumb_gap
            start_x = (total_w - total_thumbs_w) // 2

            for i, ppath in enumerate(photo_paths):
                tx = start_x + i * (thumb_size + thumb_gap)
                ty = current_y
                if os.path.exists(ppath):
                    try:
                        thumb = Image.open(ppath).convert("RGB")
                        # 按比例裁剪成正方形
                        w, h = thumb.size
                        if w > h:
                            left = (w - h) // 2
                            right = left + h
                            thumb = thumb.crop((left, 0, right, h))
                        else:
                            top = (h - w) // 2
                            bottom = top + w
                            thumb = thumb.crop((0, top, w, bottom))
                        thumb = thumb.resize((thumb_size, thumb_size), Image.LANCZOS)
                        # 圆角裁剪
                        rounded = self._round_thumb(thumb, radius=20)
                        canvas.paste(rounded, (tx, ty), rounded)
                    except Exception as e:
                        logger.warning(f"Failed to load thumbnail {ppath}: {e}")
                        self._draw_thumb_placeholder(draw, tx, ty, thumb_size)
                else:
                    self._draw_thumb_placeholder(draw, tx, ty, thumb_size)

                # 缩略图底部的序号小圆点
                dot_y = ty + thumb_size + 12
                dot_x = tx + thumb_size // 2
                dot_r = 6
                draw.ellipse([dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r], fill=(200, 200, 200))
            current_y += thumb_size + 25

        # ===== 绘制标题 =====
        if title:
            bbox = title_font.getbbox(title)
            tw = bbox[2] - bbox[0]
            tx = (total_w - tw) // 2
            draw.text((tx, current_y), title, fill=(0, 0, 0), font=title_font)
            # 标题下划线装饰
            line_y = current_y + title_height - 5
            draw.line([total_w * 0.15, line_y, total_w * 0.85, line_y], fill=(220, 220, 220), width=2)
            current_y += title_height

        # ===== 绘制 comics panels =====
        panels_start_y = header_height

        for idx, panel in enumerate(panels):
            col = idx % COLS
            row = idx // COLS
            x = OUTER_BORDER + col * (PANEL_W + BORDER)
            y = panels_start_y + row * (PANEL_H + BORDER)

            # 绘制 panel 图片
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

            # 绘制对话气泡
            dialogue = panel.get("dialogue", "")
            if dialogue:
                bubble_type = panel.get("bubble_type", "dialogue")
                face_info = panel.get("face_info")
                emotion = panel.get("emotion", "neutral")

                bubble, anchor_key = self._create_smart_bubble(
                    text=dialogue,
                    font=font,
                    bubble_type=bubble_type,
                    face_info=face_info,
                    emotion=emotion,
                )
                if bubble:
                    bx, by = self._calc_bubble_position(bubble, anchor_key, x, y)
                    canvas.paste(bubble, (bx, by), bubble)

        self._draw_watermark(draw, total_w, total_h, font)
        canvas.save(output_path, "PNG")
        logger.info(f"Strip composed: {output_path} ({total_w}x{total_h}, {n} panels)")
        return {"image_path": output_path, "width": total_w, "height": total_h}

    # ========== 智能气泡系统 ==========

    def _create_smart_bubble(
        self,
        text: str,
        font: ImageFont.ImageFont,
        bubble_type: str = "dialogue",
        face_info: Optional[Dict[str, str]] = None,
        emotion: str = "neutral",
    ) -> Tuple[Optional[Image.Image], str]:
        """创建智能气泡，返回气泡图片和锚点位置 key。"""
        max_width = BUBBLE_MAX_WIDTH
        lines = self._wrap_text(text, font, max_width - BUBBLE_PAD * 2)
        if not lines:
            return None, "top_left"

        bbox = font.getbbox("测Ag")
        line_h = bbox[3] - bbox[1] + 10
        text_w = max(font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines)
        bw = text_w + BUBBLE_PAD * 2
        bh = line_h * len(lines) + BUBBLE_PAD * 2

        # 根据人脸位置决定气泡锚点和尾巴方向
        tail_side, anchor_key = self._decide_bubble_position(face_info, bw, bh)

        # 根据气泡类型创建不同样式
        if bubble_type == "thought":
            bubble = self._create_thought_bubble(bw, bh, lines, font, line_h, tail_side)
        elif bubble_type == "narration":
            bubble = self._create_narration_bubble(bw, bh, lines, font, line_h)
        elif bubble_type == "shout":
            bubble = self._create_shout_bubble(bw, bh, lines, font, line_h, tail_side, emotion)
        else:
            bubble = self._create_dialogue_bubble(bw, bh, lines, font, line_h, tail_side, emotion)

        return bubble, anchor_key

    def _decide_bubble_position(
        self,
        face_info: Optional[Dict[str, str]],
        bubble_w: int,
        bubble_h: int,
    ) -> Tuple[str, str]:
        """根据人脸位置决定气泡位置和尾巴方向。
        返回 (tail_side, anchor_key)
        tail_side: top/bottom/left/right - 尾巴在气泡的哪一边
        anchor_key: 气泡锚点位置 key
        """
        if not face_info:
            return "bottom", "top_left"

        face_pos = face_info.get("face_position", "center_middle")
        facing = face_info.get("facing", "front")

        # 解析人脸位置
        parts = face_pos.split("_")
        if len(parts) != 2:
            h_pos, v_pos = "center", "middle"
        else:
            h_pos, v_pos = parts[0], parts[1]

        # 规则：气泡放在人脸的对侧
        # 水平方向：人脸在左 → 气泡在右；人脸在右 → 气泡在左
        # 垂直方向：人脸在上 → 气泡在下；人脸在下 → 气泡在上

        # 优先水平放置（左右两侧空间较大）
        if h_pos == "left":
            # 人脸在左边，气泡放右边
            if v_pos == "top":
                tail_side = "left"
                anchor_key = "top_right"
            elif v_pos == "bottom":
                tail_side = "left"
                anchor_key = "bottom_right"
            else:
                tail_side = "left"
                anchor_key = "middle_right"
        elif h_pos == "right":
            # 人脸在右边，气泡放左边
            if v_pos == "top":
                tail_side = "right"
                anchor_key = "top_left"
            elif v_pos == "bottom":
                tail_side = "right"
                anchor_key = "bottom_left"
            else:
                tail_side = "right"
                anchor_key = "middle_left"
        else:
            # 人脸在中间，优先放顶部或底部
            if v_pos == "top":
                # 人脸在上，气泡放下边
                tail_side = "top"
                anchor_key = "bottom_center"
            elif v_pos == "bottom":
                # 人脸在下，气泡放上边
                tail_side = "bottom"
                anchor_key = "top_center"
            else:
                # 人脸在正中间，默认放顶部
                tail_side = "bottom"
                anchor_key = "top_center"

        return tail_side, anchor_key

    def _calc_bubble_position(
        self, bubble: Image.Image, anchor_key: str, panel_x: int, panel_y: int
    ) -> Tuple[int, int]:
        """根据锚点计算气泡在 canvas 上的实际位置。"""
        bw, bh = bubble.size
        anchor_x_rel, anchor_y_rel = ANCHOR_POSITIONS.get(anchor_key, (BUBBLE_MARGIN, BUBBLE_MARGIN))

        # 根据锚点调整气泡偏移（气泡的哪个角对准锚点）
        if "left" in anchor_key:
            bx = panel_x + anchor_x_rel
        elif "right" in anchor_key:
            bx = panel_x + anchor_x_rel - bw
        else:
            bx = panel_x + anchor_x_rel - bw // 2

        if "top" in anchor_key:
            by = panel_y + anchor_y_rel
        elif "bottom" in anchor_key:
            by = panel_y + anchor_y_rel - bh
        else:
            by = panel_y + anchor_y_rel - bh // 2

        # 边界检查，确保气泡在 panel 内
        bx = max(panel_x + 10, min(bx, panel_x + PANEL_W - bw - 10))
        by = max(panel_y + 10, min(by, panel_y + PANEL_H - bh - 10))

        return bx, by

    # ========== 气泡样式 ==========

    def _create_dialogue_bubble(
        self, bw: int, bh: int, lines: List[str], font: ImageFont.ImageFont,
        line_h: int, tail_side: str, emotion: str
    ) -> Image.Image:
        """普通对话气泡：圆角矩形 + 尖尾巴。"""
        # 尾巴额外空间
        tail_size = 20
        if tail_side in ("top", "bottom"):
            total_h = bh + tail_size
            total_w = bw
        else:
            total_w = bw + tail_size
            total_h = bh

        bubble = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)

        # 根据情绪微调气泡颜色
        fill_color = (255, 255, 255, 245)
        outline_color = (0, 0, 0, 255)
        if emotion == "happy":
            fill_color = (255, 253, 240, 245)
        elif emotion == "sad":
            fill_color = (240, 245, 255, 245)
        elif emotion == "angry":
            fill_color = (255, 240, 240, 245)
        elif emotion == "surprised":
            fill_color = (240, 255, 250, 245)

        # 绘制气泡主体和尾巴
        if tail_side == "bottom":
            # 主体在上，尾巴朝下
            bd.rounded_rectangle(
                [0, 0, bw - 1, bh - 1],
                radius=BUBBLE_RADIUS, fill=fill_color, outline=outline_color, width=3
            )
            # 尾巴在底部中间
            tail_x = bw // 2
            bd.polygon(
                [(tail_x - 14, bh - 4), (tail_x + 14, bh - 4), (tail_x, total_h - 1)],
                fill=fill_color
            )
            bd.line(
                [(tail_x - 14, bh - 4), (tail_x, total_h - 1), (tail_x + 14, bh - 4)],
                fill=outline_color, width=3
            )
            text_offset_y = 0
            text_offset_x = 0
        elif tail_side == "top":
            # 主体在下，尾巴朝上
            bd.rounded_rectangle(
                [0, tail_size, bw - 1, tail_size + bh - 1],
                radius=BUBBLE_RADIUS, fill=fill_color, outline=outline_color, width=3
            )
            tail_x = bw // 2
            bd.polygon(
                [(tail_x - 14, tail_size + 4), (tail_x + 14, tail_size + 4), (tail_x, 0)],
                fill=fill_color
            )
            bd.line(
                [(tail_x - 14, tail_size + 4), (tail_x, 0), (tail_x + 14, tail_size + 4)],
                fill=outline_color, width=3
            )
            text_offset_y = tail_size
            text_offset_x = 0
        elif tail_side == "left":
            # 主体在右，尾巴朝左
            bd.rounded_rectangle(
                [tail_size, 0, tail_size + bw - 1, bh - 1],
                radius=BUBBLE_RADIUS, fill=fill_color, outline=outline_color, width=3
            )
            tail_y = bh // 2
            bd.polygon(
                [(tail_size + 4, tail_y - 14), (tail_size + 4, tail_y + 14), (0, tail_y)],
                fill=fill_color
            )
            bd.line(
                [(tail_size + 4, tail_y - 14), (0, tail_y), (tail_size + 4, tail_y + 14)],
                fill=outline_color, width=3
            )
            text_offset_x = tail_size
            text_offset_y = 0
        else:  # right
            # 主体在左，尾巴朝右
            bd.rounded_rectangle(
                [0, 0, bw - 1, bh - 1],
                radius=BUBBLE_RADIUS, fill=fill_color, outline=outline_color, width=3
            )
            tail_y = bh // 2
            bd.polygon(
                [(bw - 4, tail_y - 14), (bw - 4, tail_y + 14), (total_w - 1, tail_y)],
                fill=fill_color
            )
            bd.line(
                [(bw - 4, tail_y - 14), (total_w - 1, tail_y), (bw - 4, tail_y + 14)],
                fill=outline_color, width=3
            )
            text_offset_x = 0
            text_offset_y = 0

        # 绘制文字
        yt = BUBBLE_PAD + text_offset_y
        for line in lines:
            lb = font.getbbox(line)
            lw = lb[2] - lb[0]
            bd.text(
                ((bw - lw) // 2 + text_offset_x, yt),
                line, fill=(0, 0, 0, 255), font=font
            )
            yt += line_h

        return bubble

    def _create_thought_bubble(
        self, bw: int, bh: int, lines: List[str], font: ImageFont.ImageFont,
        line_h: int, tail_side: str
    ) -> Image.Image:
        """思考气泡：云朵形状 + 小圆圈尾巴。"""
        tail_size = 40
        if tail_side in ("top", "bottom"):
            total_h = bh + tail_size
            total_w = bw
        else:
            total_w = bw + tail_size
            total_h = bh

        bubble = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)

        fill_color = (255, 255, 255, 235)
        outline_color = (100, 100, 100, 255)

        # 云朵形主体（用多个重叠椭圆模拟云朵）
        if tail_side == "bottom":
            # 主体在上
            body_x0, body_y0 = 0, 0
            body_x1, body_y1 = bw - 1, bh - 1
        elif tail_side == "top":
            body_x0, body_y0 = 0, tail_size
            body_x1, body_y1 = bw - 1, tail_size + bh - 1
        elif tail_side == "left":
            body_x0, body_y0 = tail_size, 0
            body_x1, body_y1 = tail_size + bw - 1, bh - 1
        else:
            body_x0, body_y0 = 0, 0
            body_x1, body_y1 = bw - 1, bh - 1

        # 绘制云朵主体（多个椭圆叠加）
        # 大椭圆
        bd.ellipse(
            [body_x0, body_y0 + 10, body_x1, body_y1 - 10],
            fill=fill_color, outline=outline_color, width=2
        )
        # 顶部凸起
        bd.ellipse(
            [body_x0 + bw * 0.15, body_y0, body_x0 + bw * 0.45, body_y0 + bh * 0.35],
            fill=fill_color, outline=outline_color, width=2
        )
        bd.ellipse(
            [body_x0 + bw * 0.55, body_y0, body_x0 + bw * 0.85, body_y0 + bh * 0.3],
            fill=fill_color, outline=outline_color, width=2
        )
        # 底部凸起
        bd.ellipse(
            [body_x0 + bw * 0.1, body_y1 - bh * 0.3, body_x0 + bw * 0.4, body_y1],
            fill=fill_color, outline=outline_color, width=2
        )
        bd.ellipse(
            [body_x0 + bw * 0.6, body_y1 - bh * 0.35, body_x0 + bw * 0.9, body_y1],
            fill=fill_color, outline=outline_color, width=2
        )

        # 小圆圈尾巴
        if tail_side == "bottom":
            cx = bw // 2
            for i, (r, off) in enumerate([(10, bh + 8), (7, bh + 25), (5, bh + 38)]):
                bd.ellipse([cx - r, off - r, cx + r, off + r], fill=fill_color, outline=outline_color, width=2)
            text_offset_x, text_offset_y = 0, 0
        elif tail_side == "top":
            cx = bw // 2
            for i, (r, off) in enumerate([(10, tail_size - 8), (7, tail_size - 25), (5, tail_size - 38)]):
                bd.ellipse([cx - r, off - r, cx + r, off + r], fill=fill_color, outline=outline_color, width=2)
            text_offset_x, text_offset_y = 0, tail_size
        elif tail_side == "left":
            cy = bh // 2
            for i, (r, off) in enumerate([(10, tail_size - 8), (7, tail_size - 25), (5, tail_size - 38)]):
                bd.ellipse([off - r, cy - r, off + r, cy + r], fill=fill_color, outline=outline_color, width=2)
            text_offset_x, text_offset_y = tail_size, 0
        else:  # right
            cy = bh // 2
            for i, (r, off) in enumerate([(10, bw + 8), (7, bw + 25), (5, bw + 38)]):
                bd.ellipse([off - r, cy - r, off + r, cy + r], fill=fill_color, outline=outline_color, width=2)
            text_offset_x, text_offset_y = 0, 0

        # 绘制文字
        yt = BUBBLE_PAD + text_offset_y
        for line in lines:
            lb = font.getbbox(line)
            lw = lb[2] - lb[0]
            bd.text(
                ((bw - lw) // 2 + text_offset_x, yt),
                line, fill=(0, 0, 0, 255), font=font
            )
            yt += line_h

        return bubble

    def _create_narration_bubble(
        self, bw: int, bh: int, lines: List[str], font: ImageFont.ImageFont, line_h: int
    ) -> Image.Image:
        """旁白气泡：矩形黑框，无尾巴。"""
        bubble = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)

        # 黑色背景 + 白色文字
        bd.rectangle([0, 0, bw - 1, bh - 1], fill=(20, 20, 20, 240), outline=(255, 255, 255, 255), width=2)

        # 绘制文字（白色）
        yt = BUBBLE_PAD
        for line in lines:
            lb = font.getbbox(line)
            lw = lb[2] - lb[0]
            bd.text(
                ((bw - lw) // 2, yt),
                line, fill=(255, 255, 255, 255), font=font
            )
            yt += line_h

        return bubble

    def _create_shout_bubble(
        self, bw: int, bh: int, lines: List[str], font: ImageFont.ImageFont,
        line_h: int, tail_side: str, emotion: str
    ) -> Image.Image:
        """喊话气泡：锯齿边 + 尖尾巴。"""
        tail_size = 24
        if tail_side in ("top", "bottom"):
            total_h = bh + tail_size
            total_w = bw
        else:
            total_w = bw + tail_size
            total_h = bh

        bubble = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bubble)

        fill_color = (255, 255, 220, 245)
        outline_color = (0, 0, 0, 255)

        # 绘制锯齿边（用小三角形模拟）
        if tail_side == "bottom":
            body_x0, body_y0 = 0, 0
            body_x1, body_y1 = bw - 1, bh - 1
        elif tail_side == "top":
            body_x0, body_y0 = 0, tail_size
            body_x1, body_y1 = bw - 1, tail_size + bh - 1
        elif tail_side == "left":
            body_x0, body_y0 = tail_size, 0
            body_x1, body_y1 = tail_size + bw - 1, bh - 1
        else:
            body_x0, body_y0 = 0, 0
            body_x1, body_y1 = bw - 1, bh - 1

        # 简化版：圆角矩形 + 粗边框，模拟爆炸感
        bd.rounded_rectangle(
            [body_x0, body_y0, body_x1, body_y1],
            radius=10, fill=fill_color, outline=outline_color, width=5
        )

        # 添加爆炸锯齿效果（在四角加小三角）
        spike_size = 12
        # 顶部锯齿
        for i in range(4, int(bw / spike_size) - 2):
            sx = body_x0 + i * spike_size
            if tail_side == "top" and bh * 0.3 < sx < bw * 0.7:
                continue
            bd.polygon(
                [(sx, body_y0), (sx + spike_size // 2, body_y0 - 8), (sx + spike_size, body_y0)],
                fill=fill_color
            )
            bd.line(
                [(sx, body_y0), (sx + spike_size // 2, body_y0 - 8), (sx + spike_size, body_y0)],
                fill=outline_color, width=3
            )
        # 底部锯齿
        for i in range(4, int(bw / spike_size) - 2):
            sx = body_x0 + i * spike_size
            if tail_side == "bottom" and bw * 0.3 < sx < bw * 0.7:
                continue
            bd.polygon(
                [(sx, body_y1), (sx + spike_size // 2, body_y1 + 8), (sx + spike_size, body_y1)],
                fill=fill_color
            )
            bd.line(
                [(sx, body_y1), (sx + spike_size // 2, body_y1 + 8), (sx + spike_size, body_y1)],
                fill=outline_color, width=3
            )

        # 尾巴
        if tail_side == "bottom":
            tail_x = bw // 2
            bd.polygon(
                [(tail_x - 16, bh - 4), (tail_x + 16, bh - 4), (tail_x, total_h - 1)],
                fill=fill_color
            )
            bd.line(
                [(tail_x - 16, bh - 4), (tail_x, total_h - 1), (tail_x + 16, bh - 4)],
                fill=outline_color, width=5
            )
            text_offset_x, text_offset_y = 0, 0
        elif tail_side == "top":
            tail_x = bw // 2
            bd.polygon(
                [(tail_x - 16, tail_size + 4), (tail_x + 16, tail_size + 4), (tail_x, 0)],
                fill=fill_color
            )
            bd.line(
                [(tail_x - 16, tail_size + 4), (tail_x, 0), (tail_x + 16, tail_size + 4)],
                fill=outline_color, width=5
            )
            text_offset_x, text_offset_y = 0, tail_size
        elif tail_side == "left":
            tail_y = bh // 2
            bd.polygon(
                [(tail_size + 4, tail_y - 16), (tail_size + 4, tail_y + 16), (0, tail_y)],
                fill=fill_color
            )
            bd.line(
                [(tail_size + 4, tail_y - 16), (0, tail_y), (tail_size + 4, tail_y + 16)],
                fill=outline_color, width=5
            )
            text_offset_x, text_offset_y = tail_size, 0
        else:  # right
            tail_y = bh // 2
            bd.polygon(
                [(bw - 4, tail_y - 16), (bw - 4, tail_y + 16), (total_w - 1, tail_y)],
                fill=fill_color
            )
            bd.line(
                [(bw - 4, tail_y - 16), (total_w - 1, tail_y), (bw - 4, tail_y + 16)],
                fill=outline_color, width=5
            )
            text_offset_x, text_offset_y = 0, 0

        # 绘制文字（粗体效果用两次偏移模拟）
        yt = BUBBLE_PAD + text_offset_y
        for line in lines:
            lb = font.getbbox(line)
            lw = lb[2] - lb[0]
            tx = (bw - lw) // 2 + text_offset_x
            # 模拟粗体
            bd.text((tx + 1, yt), line, fill=(0, 0, 0, 255), font=font)
            bd.text((tx, yt + 1), line, fill=(0, 0, 0, 255), font=font)
            bd.text((tx, yt), line, fill=(0, 0, 0, 255), font=font)
            yt += line_h

        return bubble

    # ========== 工具方法 ==========

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
    def _round_thumb(img: Image.Image, radius: int = 20) -> Image.Image:
        """将图片裁剪为圆角矩形，返回带 alpha 通道的图片。"""
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
        result = img.convert("RGBA")
        result.putalpha(mask)
        return result

    @staticmethod
    def _draw_thumb_placeholder(draw: ImageDraw.ImageDraw, x: int, y: int, size: int):
        draw.rounded_rectangle([x, y, x + size - 1, y + size - 1], radius=20, fill=(230, 230, 230))

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
