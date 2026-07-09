from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float,
    ForeignKey, JSON, func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Album(Base):
    __tablename__ = "albums"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, default="")
    date = Column(String(10), nullable=False, default="")
    photo_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())

    photos = relationship("Photo", back_populates="album", lazy="selectin")
    stories = relationship("Story", back_populates="album", lazy="selectin")


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    album_id = Column(Integer, ForeignKey("albums.id"), nullable=False)
    path = Column(String(500), nullable=False)
    taken_at = Column(String(19), nullable=True, default="")

    album = relationship("Album", back_populates="photos")
    analysis = relationship("PhotoAnalysis", back_populates="photo", uselist=False, lazy="selectin")
    scene_photos = relationship("ScenePhoto", back_populates="photo", lazy="selectin")


class PhotoAnalysis(Base):
    __tablename__ = "photo_analysis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(Integer, ForeignKey("photos.id"), nullable=False, unique=True)
    objects = Column(JSON, nullable=True)
    scene_desc = Column(Text, nullable=True, default="")
    weather = Column(String(50), nullable=True, default="")
    location = Column(String(255), nullable=True, default="")
    action = Column(String(100), nullable=True, default="")
    emotion = Column(String(50), nullable=True, default="")
    clothing = Column(JSON, nullable=True)
    people = Column(JSON, nullable=True)
    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    photo = relationship("Photo", back_populates="analysis")


class Story(Base):
    __tablename__ = "stories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    album_id = Column(Integer, ForeignKey("albums.id"), nullable=False)
    title = Column(String(255), nullable=False, default="")
    summary = Column(Text, nullable=True, default="")
    status = Column(String(20), nullable=False, default="processing")
    scene_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())

    album = relationship("Album", back_populates="stories")
    scenes = relationship("Scene", back_populates="story", lazy="selectin",
                          order_by="Scene.seq_num")
    comics = relationship("Comic", back_populates="story", lazy="selectin")


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    seq_num = Column(Integer, nullable=False)
    time_at = Column(String(10), nullable=True, default="")
    location = Column(String(255), nullable=True, default="")
    summary = Column(Text, nullable=True, default="")
    narration = Column(Text, nullable=True, default="")
    dialogue = Column(Text, nullable=True, default="")
    comic_image_url = Column(String(1000), nullable=True, default="")

    story = relationship("Story", back_populates="scenes")
    characters = relationship("SceneCharacter", back_populates="scene", lazy="selectin")
    actions = relationship("SceneAction", back_populates="scene", lazy="selectin")
    emotions = relationship("SceneEmotion", back_populates="scene", lazy="selectin")
    photos = relationship("ScenePhoto", back_populates="scene", lazy="selectin")


class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, default="")
    traits = Column(JSON, nullable=True)
    avatar_url = Column(String(500), nullable=True, default="")
    created_at = Column(DateTime, server_default=func.now())

    scenes = relationship("SceneCharacter", back_populates="character", lazy="selectin")


class SceneCharacter(Base):
    __tablename__ = "scene_characters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=False)
    character_id = Column(Integer, ForeignKey("characters.id"), nullable=False)
    role = Column(String(50), nullable=True, default="")

    scene = relationship("Scene", back_populates="characters")
    character = relationship("Character", back_populates="scenes")


class SceneAction(Base):
    __tablename__ = "scene_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=False)
    verb = Column(String(100), nullable=False, default="")
    object = Column(String(255), nullable=True, default="")

    scene = relationship("Scene", back_populates="actions")


class SceneEmotion(Base):
    __tablename__ = "scene_emotions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=False)
    type = Column(String(50), nullable=False, default="")
    intensity = Column(Float, nullable=True, default=0.0)

    scene = relationship("Scene", back_populates="emotions")


class ScenePhoto(Base):
    __tablename__ = "scene_photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, ForeignKey("scenes.id"), nullable=False)
    photo_id = Column(Integer, ForeignKey("photos.id"), nullable=False)

    scene = relationship("Scene", back_populates="photos")
    photo = relationship("Photo", back_populates="scene_photos")


class Comic(Base):
    __tablename__ = "comics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    story_id = Column(Integer, ForeignKey("stories.id"), nullable=False)
    style = Column(String(50), nullable=False, default="slice")
    status = Column(String(20), nullable=False, default="processing")
    total_pages = Column(Integer, nullable=False, default=0)
    pdf_path = Column(String(500), nullable=False, default="")
    image_partial = Column(Integer, nullable=False, default=0)
    dialogue_partial = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())

    story = relationship("Story", back_populates="comics")
    pages = relationship("ComicPage", back_populates="comic", lazy="selectin",
                         order_by="ComicPage.page_num")
    tasks = relationship("GenerationTask", back_populates="comic", lazy="selectin",
                         order_by="GenerationTask.id")


class ComicPage(Base):
    __tablename__ = "comic_pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    comic_id = Column(Integer, ForeignKey("comics.id"), nullable=False)
    page_num = Column(Integer, nullable=False)
    shot_type = Column(String(20), nullable=False, default="Medium")
    image_prompt = Column(Text, nullable=False, default="")
    image_url = Column(String(500), nullable=False, default="")
    dialogue = Column(Text, nullable=False, default="")
    narration = Column(Text, nullable=False, default="")
    layout_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    comic = relationship("Comic", back_populates="pages")


class GenerationTask(Base):
    __tablename__ = "generation_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    comic_id = Column(Integer, ForeignKey("comics.id"), nullable=False)
    agent_name = Column(String(30), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    input_snapshot = Column(JSON, nullable=True)
    output_snapshot = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=False, default="")
    retry_count = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    comic = relationship("Comic", back_populates="tasks")


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(30), nullable=False, default="director")
    style_key = Column(String(30), nullable=False)
    style_name = Column(String(50), nullable=False, default="")
    mood = Column(String(20), nullable=False, default="温馨")
    dialogue_level = Column(String(10), nullable=False, default="中")
    cinematic = Column(Integer, nullable=False, default=0)
    prompt_template = Column(Text, nullable=False, default="")
    version = Column(Integer, nullable=False, default=1)
    is_active = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, server_default=func.now())
