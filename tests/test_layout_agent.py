import os
import pytest
from PIL import Image
from app.agents.layout_agent import LayoutAgent


@pytest.fixture
def test_images_dir(tmp_path):
    """创建临时测试图片目录。"""
    img_dir = tmp_path / "1"
    img_dir.mkdir()
    # 创建两张 100x100 的纯色 PNG 测试图片
    img1 = Image.new("RGB", (100, 100), color=(255, 100, 100))
    img1.save(img_dir / "page_1.png")
    img2 = Image.new("RGB", (100, 100), color=(100, 100, 255))
    img2.save(img_dir / "page_2.png")
    return tmp_path


@pytest.fixture
def sample_pages(test_images_dir):
    return [
        {
            "page": 1,
            "shot": "Wide",
            "desc": "公园全景",
            "dialogue": "",
            "source_scene_seq": 0,
            "image_url": str(test_images_dir / "1" / "page_1.png")
        },
        {
            "page": 2,
            "shot": "Medium",
            "desc": "小明微笑",
            "dialogue": "今天真开心！",
            "source_scene_seq": 1,
            "image_url": str(test_images_dir / "1" / "page_2.png")
        }
    ]


@pytest.mark.asyncio
async def test_layout_generates_pdf(test_images_dir, sample_pages):
    """测试生成 PDF，断言文件存在且大小 > 1000 字节。"""
    agent = LayoutAgent(str(test_images_dir))
    result = await agent.run(
        comic_id=1,
        story_title="测试故事",
        style_name="宫崎骏风",
        pages=sample_pages,
        created_at_str="2026-07-09"
    )
    assert result["success"] is True
    assert result["pdf_path"] != ""
    assert os.path.exists(result["pdf_path"])
    assert os.path.getsize(result["pdf_path"]) > 1000


@pytest.mark.asyncio
async def test_layout_handles_missing_image(test_images_dir):
    """测试包含空 image_url 的页面，PDF 仍能生成。"""
    pages = [
        {
            "page": 1,
            "shot": "Wide",
            "desc": "封面",
            "dialogue": "",
            "source_scene_seq": 0,
            "image_url": ""
        },
        {
            "page": 2,
            "shot": "Medium",
            "desc": "内容页",
            "dialogue": "你好",
            "source_scene_seq": 1,
            "image_url": str(test_images_dir / "1" / "page_1.png")
        }
    ]
    agent = LayoutAgent(str(test_images_dir))
    result = await agent.run(
        comic_id=2,
        story_title="缺图测试",
        style_name="日常风",
        pages=pages,
        created_at_str="2026-07-09"
    )
    assert result["success"] is True
    assert os.path.exists(result["pdf_path"])
    assert os.path.getsize(result["pdf_path"]) > 1000


@pytest.mark.asyncio
async def test_layout_empty_pages(tmp_path):
    """空页面列表，PDF 仍生成（只有封面和结尾页）。"""
    agent = LayoutAgent(str(tmp_path))
    result = await agent.run(
        comic_id=3,
        story_title="空故事",
        style_name="日常风",
        pages=[],
        created_at_str="2026-07-09"
    )
    assert result["success"] is True
    assert os.path.exists(result["pdf_path"])
    assert os.path.getsize(result["pdf_path"]) > 1000
