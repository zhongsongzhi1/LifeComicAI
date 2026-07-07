from unittest.mock import MagicMock

import pytest


class TestAlbumAgent:
    def test_sort_photos_by_exif_time(self):
        """验证照片按 EXIF 时间排序"""
        from app.agents.album_agent import AlbumAgent

        agent = AlbumAgent(llm_provider=None)
        photos = [
            {"id": 1, "path": "/p1.jpg", "taken_at": "2026-07-07 10:00:00"},
            {"id": 2, "path": "/p2.jpg", "taken_at": "2026-07-07 08:00:00"},
            {"id": 3, "path": "/p3.jpg", "taken_at": "2026-07-07 12:00:00"},
        ]

        result = agent._sort_by_time(photos)
        assert result[0]["id"] == 2
        assert result[1]["id"] == 1
        assert result[2]["id"] == 3

    def test_cluster_photos_by_time_gap(self):
        """验证按时间间隔聚类（间隔 > 2 小时则分簇）"""
        from app.agents.album_agent import AlbumAgent

        agent = AlbumAgent(llm_provider=None)
        photos = [
            {"id": 1, "path": "/p1.jpg", "taken_at": "2026-07-07 08:00:00"},
            {"id": 2, "path": "/p2.jpg", "taken_at": "2026-07-07 08:30:00"},
            {"id": 3, "path": "/p3.jpg", "taken_at": "2026-07-07 12:00:00"},
            {"id": 4, "path": "/p4.jpg", "taken_at": "2026-07-07 12:15:00"},
        ]

        clusters = agent._cluster_by_time(photos, gap_minutes=120)
        assert len(clusters) == 2
        assert len(clusters[0]) == 2
        assert len(clusters[1]) == 2

    @pytest.mark.asyncio
    async def test_run_generates_timeline(self):
        """验证完整 run 方法生成时间线"""
        mock_provider = MagicMock()
        mock_provider.chat_json.return_value = {
            "timeline": [
                {"cluster_id": 0, "label": "早餐"},
                {"cluster_id": 1, "label": "东京塔"},
            ]
        }

        from app.agents.album_agent import AlbumAgent

        agent = AlbumAgent(llm_provider=mock_provider)
        photos = [
            {"id": 1, "path": "/p1.jpg", "taken_at": "2026-07-07 08:00:00"},
            {"id": 2, "path": "/p2.jpg", "taken_at": "2026-07-07 08:30:00"},
            {"id": 3, "path": "/p3.jpg", "taken_at": "2026-07-07 12:00:00"},
        ]

        result = await agent.run(photos=photos)

        assert isinstance(result, dict)
        assert result["date"] == "2026-07-07"
        assert len(result["entries"]) == 2
        assert result["entries"][0]["label"] == "早餐"
