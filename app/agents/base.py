from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Agent 抽象基类。"""

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        ...
