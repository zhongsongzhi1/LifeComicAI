"""Compatibility shim: expose PromptEnhancer under app.services for existing imports."""
from app.utils.prompt_enhancer import PromptEnhancer

__all__ = ["PromptEnhancer"]
