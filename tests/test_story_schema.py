from app.schemas.story import (
    StoryGenerateRequest, StoryResponse, StoryGraphResponse,
    SceneResponse, CharacterResponse, ActionResponse, EmotionResponse,
    SceneDetailResponse,
)


class TestStoryGenerateRequest:
    def test_valid_request(self):
        r = StoryGenerateRequest(album_id=1)
        assert r.album_id == 1


class TestStoryResponse:
    def test_story_response(self):
        s = StoryResponse(
            id=1,
            album_id=1,
            title="东京一日",
            summary="测试故事",
            status="processing",
            scene_count=3,
        )
        d = s.model_dump()
        assert d["id"] == 1
        assert d["status"] == "processing"


class TestSceneResponse:
    def test_scene_response(self):
        s = SceneResponse(
            id=1,
            story_id=1,
            seq_num=1,
            time_at="08:30",
            location="酒店",
            summary="吃早餐",
        )
        assert s.seq_num == 1
        assert s.location == "酒店"


class TestCharacterResponse:
    def test_character_response(self):
        c = CharacterResponse(
            id=1,
            name="我",
            traits=["长发", "眼镜"],
            role="主角",
        )
        assert c.name == "我"
        assert len(c.traits) == 2


class TestActionResponse:
    def test_action_response(self):
        a = ActionResponse(verb="吃", object="早餐")
        assert a.verb == "吃"
        assert a.object == "早餐"


class TestEmotionResponse:
    def test_emotion_response(self):
        e = EmotionResponse(type="期待", intensity=0.7)
        assert e.type == "期待"
        assert e.intensity == 0.7


class TestStoryGraphResponse:
    def test_story_graph_response(self):
        g = StoryGraphResponse(
            story_id=1,
            story_title="东京一日",
            story_summary="测试故事",
            scenes=[
                SceneResponse(
                    id=1, story_id=1, seq_num=1, time_at="08:30",
                    location="酒店", summary="吃早餐",
                    characters=[
                        CharacterResponse(id=1, name="我", traits=["长发"], role="主角"),
                    ],
                    actions=[ActionResponse(verb="吃", object="早餐")],
                    emotions=[EmotionResponse(type="期待", intensity=0.7)],
                    source_photos=[1, 2],
                ),
            ],
            global_characters=[
                CharacterResponse(id=1, name="我", traits=["长发", "眼镜"], role="主角"),
            ],
        )
        d = g.model_dump()
        assert d["story_title"] == "东京一日"
        assert len(d["scenes"]) == 1
        assert len(d["global_characters"]) == 1
        assert d["scenes"][0]["source_photos"] == [1, 2]


class TestSceneDetailResponse:
    def test_scene_detail_response(self):
        s = SceneDetailResponse(
            id=1,
            story_id=1,
            seq_num=1,
            time_at="08:30",
            location="酒店",
            summary="吃早餐",
            characters=[],
            actions=[],
            emotions=[],
            source_photos=[
                {"id": 1, "album_id": 1, "path": "/uploads/1/test.jpg", "taken_at": "2026-07-07 08:30:00"},
            ],
        )
        d = s.model_dump()
        assert len(d["source_photos"]) == 1
        assert d["source_photos"][0]["path"] == "/uploads/1/test.jpg"
