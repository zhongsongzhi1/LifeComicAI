import logging
from datetime import datetime
from typing import Any, Dict, List

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

LABELING_PROMPT = """你是一个生活记录助手。以下是同一天内按时间顺序的照片分组。

每个分组包含若干张照片（按拍照时间相近聚在一起）。请为每个分组生成一个简短的中文标签（2-6个字），
概括这个时间段的活动。同时为整个时间线生成一个日期标签。

输入格式（JSON）：
{
  "clusters": [
    {"cluster_id": 0, "photo_count": 3, "start_time": "08:00", "end_time": "08:30"},
    {"cluster_id": 1, "photo_count": 2, "start_time": "12:00", "end_time": "12:15"}
  ]
}

输出格式（仅输出 JSON，不要其他文字）：
{
  "date_label": "YYYY-MM-DD 星期X",
  "timeline": [
    {"cluster_id": 0, "label": "早餐时光"},
    {"cluster_id": 1, "label": "午餐聚会"}
  ]
}"""


class AlbumAgent(BaseAgent):
    GAP_MINUTES = 120

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    async def run(self, photos: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not photos:
            return {"date": "", "entries": [], "photo_count": 0}

        sorted_photos = self._sort_by_time(photos)
        date = self._extract_date(sorted_photos)
        clusters = self._cluster_by_time(sorted_photos, gap_minutes=self.GAP_MINUTES)
        labels = await self._generate_labels(clusters)
        entries = self._build_entries(clusters, labels)

        return {"date": date, "entries": entries, "photo_count": len(photos)}

    def _sort_by_time(self, photos):
        return sorted(photos, key=lambda p: p.get("taken_at", ""))

    def _extract_date(self, photos):
        taken = photos[0].get("taken_at", "")
        return taken[:10] if taken else ""

    def _cluster_by_time(self, photos, gap_minutes):
        if not photos:
            return []

        clusters = []
        current_cluster = [photos[0]]

        for i in range(1, len(photos)):
            prev_time = self._parse_time(photos[i - 1].get("taken_at", ""))
            curr_time = self._parse_time(photos[i].get("taken_at", ""))
            if prev_time and curr_time:
                diff = (curr_time - prev_time).total_seconds() / 60
                if diff > gap_minutes:
                    clusters.append(current_cluster)
                    current_cluster = [photos[i]]
                else:
                    current_cluster.append(photos[i])
            else:
                current_cluster.append(photos[i])

        clusters.append(current_cluster)
        return clusters

    @staticmethod
    def _parse_time(taken_at):
        if not taken_at:
            return None
        try:
            return datetime.strptime(taken_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    async def _generate_labels(self, clusters):
        if not self.llm_provider:
            return {i: f"片段{i + 1}" for i in range(len(clusters))}

        cluster_info = []
        for i, cluster in enumerate(clusters):
            times = [p.get("taken_at", "") for p in cluster]
            start_time = times[0][-8:-3] if times[0] else "??:??"
            end_time = times[-1][-8:-3] if times[-1] else "??:??"
            cluster_info.append({
                "cluster_id": i,
                "photo_count": len(cluster),
                "start_time": start_time,
                "end_time": end_time,
            })

        try:
            messages = [
                {"role": "system", "content": LABELING_PROMPT},
                {"role": "user", "content": f'{{"clusters": {cluster_info}}}'},
            ]
            result = self.llm_provider.chat_json(messages, temperature=0.3)
            labels = {}
            for item in result.get("timeline", []):
                labels[item["cluster_id"]] = item.get("label", f"片段{item['cluster_id'] + 1}")
            return labels
        except Exception as e:
            logger.error(f"Label generation failed: {e}")
            return {i: f"片段{i + 1}" for i in range(len(clusters))}

    def _build_entries(self, clusters, labels):
        entries = []
        for i, cluster in enumerate(clusters):
            times = [p.get("taken_at", "") for p in cluster]
            time_str = times[0][-8:-3] if times[0] else "??:??"
            entries.append({
                "seq": i + 1,
                "time": time_str,
                "photos": [p["id"] for p in cluster],
                "label": labels.get(i, f"片段{i + 1}"),
            })
        return entries
