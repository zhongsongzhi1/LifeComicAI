import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Character, SceneCharacter, Scene

logger = logging.getLogger(__name__)


class CharacterService:
    """Character 查询服务。"""

    async def get_character(self, character_id: int, session: AsyncSession) -> Optional[dict]:
        """获取角色信息（含出现在哪些故事中）。

        Args:
            character_id: 角色 ID
            session: 数据库会话

        Returns:
            dict or None: CharacterResponse 序列化字典
        """
        query = select(Character).where(Character.id == character_id).options(
            selectinload(Character.scenes).selectinload(SceneCharacter.scene)
        )
        result = await session.execute(query)
        character = result.scalar_one_or_none()

        if not character:
            return None

        # 收集角色出现过的故事
        story_ids = set()
        for sc in character.scenes:
            if sc.scene:
                story_ids.add(sc.scene.story_id)

        return {
            "id": character.id,
            "name": character.name,
            "traits": character.traits or [],
            "role": "",
            "avatar_url": character.avatar_url or "",
            "appeared_in_stories": list(story_ids),
        }
