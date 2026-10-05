"""GPU Lock and Execution Serializer."""
import asyncio
from typing import Optional


class GpuLock:
    """Async lock wrapper to ensure single-process GPU residency without concurrent model collisions."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_owner: Optional[str] = None

    async def acquire(self, model_name: str) -> None:
        await self._lock.acquire()
        self._current_owner = model_name

    def release(self) -> None:
        self._current_owner = None
        if self._lock.locked():
            self._lock.release()

    @property
    def current_owner(self) -> Optional[str]:
        return self._current_owner


# Global in-process lock singleton
gpu_lock = GpuLock()
