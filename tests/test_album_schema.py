from app.schemas.album import (
    AlbumCreate, AlbumResponse, TimelineEntry, TimelineResponse, PhotoResponse,
)


class TestAlbumCreate:
    def test_valid_create(self):
        a = AlbumCreate(title="东京一日", date="2026-07-07")
        assert a.title == "东京一日"
        assert a.date == "2026-07-07"


class TestAlbumResponse:
    def test_album_response(self):
        a = AlbumResponse(
            id=1,
            title="东京一日",
            date="2026-07-07",
            photo_count=5,
        )
        d = a.model_dump()
        assert d["id"] == 1
        assert d["photo_count"] == 5


class TestTimelineEntry:
    def test_timeline_entry(self):
        e = TimelineEntry(seq=1, time="08:30", photos=[1, 2], label="早餐")
        assert e.seq == 1
        assert len(e.photos) == 2
        assert e.label == "早餐"


class TestTimelineResponse:
    def test_timeline_response(self):
        t = TimelineResponse(
            date="2026-07-07",
            entries=[
                TimelineEntry(seq=1, time="08:30", photos=[1], label="早餐"),
                TimelineEntry(seq=2, time="10:00", photos=[2], label="东京塔"),
            ],
            photo_count=2,
        )
        d = t.model_dump()
        assert d["date"] == "2026-07-07"
        assert len(d["entries"]) == 2
        assert d["photo_count"] == 2


class TestPhotoResponse:
    def test_photo_response(self):
        p = PhotoResponse(id=1, album_id=1, path="/uploads/1/test.jpg", taken_at="2026-07-07 10:00:00")
        assert p.id == 1
        assert p.path == "/uploads/1/test.jpg"
