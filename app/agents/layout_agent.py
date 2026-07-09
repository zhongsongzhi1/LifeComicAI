import logging
import os
from typing import Any, Dict, List

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MAX_IMG_W = 500
MAX_IMG_H = 600
MARGIN = 50


class LayoutAgent(BaseAgent):
    """Layout Agent：将漫画图片、对白等组装成 PDF。"""

    def __init__(self, comics_dir: str):
        self.comics_dir = comics_dir

    async def run(self, **kwargs) -> Dict[str, Any]:
        comic_id = kwargs.get("comic_id", 0)
        story_title = kwargs.get("story_title", "")
        style_name = kwargs.get("style_name", "")
        pages: List[Dict[str, Any]] = kwargs.get("pages", [])
        created_at_str = kwargs.get("created_at_str", "")

        output_dir = os.path.join(self.comics_dir, str(comic_id))
        os.makedirs(output_dir, exist_ok=True)

        pdf_path = os.path.join(output_dir, "comic.pdf")

        try:
            c = canvas.Canvas(pdf_path, pagesize=A4)
            self._draw_cover(c, story_title, created_at_str, style_name)
            for page_info in pages:
                self._draw_page(c, page_info)
            self._draw_end_page(c)
            c.save()
            return {"pdf_path": pdf_path, "success": True}
        except Exception:
            logger.exception("Failed to generate PDF for comic %s", comic_id)
            return {"pdf_path": "", "success": False}

    # ---------- cover ----------

    def _draw_cover(self, c: canvas.Canvas, title: str, date_str: str, style: str):
        """绘制封面页。"""
        y_center = PAGE_H / 2
        # 标题
        c.setFont("Helvetica-Bold", 24)
        self._draw_centered_text(c, title, y_center + 20)
        # 日期
        if date_str:
            c.setFont("Helvetica", 14)
            self._draw_centered_text(c, date_str, y_center - 30)
        # 风格
        if style:
            c.setFont("Helvetica", 14)
            self._draw_centered_text(c, f"风格：{style}", y_center - 55)
        c.showPage()

    # ---------- content page ----------

    def _draw_page(self, c: canvas.Canvas, page_info: Dict[str, Any]):
        """绘制一页内容：图片 + 对白 + 页码。"""
        page_num = page_info.get("page", 0)
        dialogue = page_info.get("dialogue", "")
        image_url = page_info.get("image_url", "")

        # --- 图片 ---
        img_y_top = PAGE_H - MARGIN
        if image_url and os.path.exists(image_url):
            try:
                img = Image.open(image_url)
                iw, ih = img.size
                # 按比例缩放
                scale = min(MAX_IMG_W / iw, MAX_IMG_H / ih, 1.0)
                dw, dh = iw * scale, ih * scale
                img_x = (PAGE_W - dw) / 2
                img_y = img_y_top - dh
                c.drawImage(image_url, img_x, img_y, width=dw, height=dh)
                dialogue_y = img_y - 20
            except Exception:
                logger.exception("Failed to draw image %s", image_url)
                self._draw_placeholder(c, img_y_top, MAX_IMG_H)
                dialogue_y = img_y_top - MAX_IMG_H - 20
        else:
            self._draw_placeholder(c, img_y_top, MAX_IMG_H)
            dialogue_y = img_y_top - MAX_IMG_H - 20

        # --- 对白 ---
        if dialogue:
            self._draw_dialogue(c, dialogue, dialogue_y)

        # --- 页码 ---
        c.setFont("Helvetica", 10)
        c.drawRightString(PAGE_W - MARGIN, MARGIN, str(page_num))

        c.showPage()

    # ---------- end page ----------

    def _draw_end_page(self, c: canvas.Canvas):
        """绘制结尾页。"""
        c.setFont("Helvetica", 18)
        self._draw_centered_text(c, "—— Powered by LifeComicAI ——", PAGE_H / 2)
        c.showPage()

    # ---------- helpers ----------

    @staticmethod
    def _draw_centered_text(c: canvas.Canvas, text: str, y: float):
        """在 PDF 页面中居中绘制单行文字。"""
        text_width = c.stringWidth(text)
        x = (PAGE_W - text_width) / 2
        c.drawString(x, y, text)

    @staticmethod
    def _draw_placeholder(c: canvas.Canvas, top_y: float, height: float):
        """绘制"图片生成失败"占位框与文字。"""
        box_y = top_y - height
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(1)
        c.rect(MARGIN, box_y, PAGE_W - 2 * MARGIN, height)
        c.setFont("Helvetica", 14)
        msg = "图片生成失败"
        tw = c.stringWidth(msg, "Helvetica", 14)
        c.drawString((PAGE_W - tw) / 2, box_y + height / 2 - 7, msg)

    @staticmethod
    def _draw_dialogue(c: canvas.Canvas, text: str, start_y: float):
        """绘制对白文字，自动换行。"""
        c.setFont("Helvetica", 12)
        max_width = PAGE_W - 2 * MARGIN
        # 按字符粗略估算每行最大字符数（中文字符按 1.5 倍宽度估算）
        char_width = c.stringWidth("测", "Helvetica", 12)  # 中文字符宽度近似
        if char_width < 1:
            char_width = 6
        max_chars = max(1, int(max_width / char_width))
        lines = LayoutAgent._wrap_text(text, max_chars)

        line_height = 16
        y = start_y
        for line in lines:
            if y < MARGIN:
                break
            tw = c.stringWidth(line, "Helvetica", 12)
            c.drawString((PAGE_W - tw) / 2, y, line)
            y -= line_height

    @staticmethod
    def _wrap_text(text: str, max_chars: int) -> List[str]:
        """将文本按 max_chars 字符宽度换行。"""
        if not text:
            return []
        lines = []
        current = ""
        for ch in text:
            current += ch
            if len(current) >= max_chars:
                lines.append(current)
                current = ""
        if current:
            lines.append(current)
        return lines
