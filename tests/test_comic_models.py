import pytest
from app.db.models import Comic, ComicPage, GenerationTask, PromptVersion
from app.db.seed_prompts import seed_prompt_versions


@pytest.mark.asyncio
async def test_create_comic(session):
    comic = Comic(story_id=1, style="miyazaki")
    session.add(comic)
    await session.commit()
    await session.refresh(comic)

    assert comic.id is not None
    assert comic.story_id == 1
    assert comic.style == "miyazaki"
    assert comic.status == "processing"


@pytest.mark.asyncio
async def test_create_comic_page(session):
    comic = Comic(story_id=1)
    session.add(comic)
    await session.flush()

    page = ComicPage(
        comic_id=comic.id,
        page_num=1,
        shot_type="Wide",
        image_prompt="test",
        image_url="/test.png",
        dialogue="hello",
        narration="narration",
    )
    session.add(page)
    await session.commit()
    await session.refresh(page)

    assert page.id is not None


@pytest.mark.asyncio
async def test_create_generation_task(session):
    comic = Comic(story_id=1)
    session.add(comic)
    await session.flush()

    task = GenerationTask(comic_id=comic.id, agent_name="storyboard", status="done")
    session.add(task)
    await session.commit()
    await session.refresh(task)

    assert task.id is not None


@pytest.mark.asyncio
async def test_prompt_version_seed(session):
    await seed_prompt_versions(session)

    from sqlalchemy import select
    result = await session.execute(
        select(PromptVersion.style_key).where(PromptVersion.is_active == 1)
    )
    style_keys = {row[0] for row in result.fetchall()}

    assert style_keys == {"miyazaki", "cinematic", "slice", "manga"}
