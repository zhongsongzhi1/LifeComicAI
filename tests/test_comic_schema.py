from app.schemas.comic import (
    ComicGenerateRequest, ComicResponse, ComicPageResponse,
    GenerationTaskResponse, StylePresetResponse,
    ComicPagesResponse, ComicTasksResponse,
)


class TestComicGenerateRequest:
    def test_default(self):
        r = ComicGenerateRequest(story_id=1)
        assert r.story_id == 1
        assert r.style_key == "slice"

    def test_custom_style(self):
        r = ComicGenerateRequest(story_id=1, style_key="miyazaki")
        assert r.style_key == "miyazaki"


class TestComicResponse:
    def test_from_attributes(self):
        c = ComicResponse.model_validate({
            "id": 1,
            "story_id": 1,
            "style": "miyazaki",
            "status": "processing",
            "total_pages": 0,
            "pdf_path": "",
            "image_partial": False,
            "dialogue_partial": False,
        })
        assert c.id == 1


class TestComicPageResponse:
    def test_comic_page_response(self):
        p = ComicPageResponse.model_validate({
            "id": 1,
            "comic_id": 1,
            "page_num": 1,
            "shot_type": "Wide",
        })
        assert p.id == 1
        assert p.comic_id == 1
        assert p.page_num == 1
        assert p.shot_type == "Wide"


class TestGenerationTaskResponse:
    def test_generation_task_response(self):
        t = GenerationTaskResponse.model_validate({
            "id": 1,
            "comic_id": 1,
            "agent_name": "storyboard",
            "status": "done",
            "error_message": "",
            "retry_count": 0,
            "duration_ms": 1000,
        })
        assert t.agent_name == "storyboard"


class TestStylePresetResponse:
    def test_style_preset_response(self):
        s = StylePresetResponse(
            key="miyazaki",
            name="宫崎骏风",
            mood="温馨",
            dialogue_level="少",
            cinematic=True,
        )
        assert s.key == "miyazaki"
        assert s.name == "宫崎骏风"
        assert s.mood == "温馨"
        assert s.dialogue_level == "少"
        assert s.cinematic is True


class TestComicPagesResponse:
    def test_comic_pages_response(self):
        r = ComicPagesResponse(story_id=1, comic_id=1, pages=[])
        assert r.pages == []


class TestComicTasksResponse:
    def test_comic_tasks_response(self):
        r = ComicTasksResponse(comic_id=1, status="processing", tasks=[])
        assert r.tasks == []
