import random
import asyncio
import logging
import aiohttp
import affine as af
from collections import deque
from typing import Any, Deque, List, Optional

class BufferedDataset:
    def __init__(
        self,
        dataset_name: str,
        total_size: int,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
        split: str = "train",
        config: str = "default",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 30.0
    ):
        self.dataset_name   = dataset_name
        self.total_size     = total_size
        self.buffer_size    = buffer_size
        self.max_batch      = max_batch
        self.split          = split
        self.config         = config
        self.max_retries    = max_retries
        self.initial_backoff= initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff    = max_backoff

        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None
        self._rng    = random.Random(seed)

    async def fetch_hf(self, offset: int, length: int) -> List[Any]:
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={self.dataset_name}"
            f"&config={self.config}"
            f"&split={self.split}"
            f"&offset={offset}"
            f"&length={length}"
        )
        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            af.logger.debug(f"HF fetch attempt {attempt}: offset={offset}, len={length}")
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=30) as resp:
                    if resp.status == 429:
                        af.logger.debug(f"Rate‑limit hit; sleeping {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * self.backoff_factor, self.max_backoff)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    rows = [r["row"] for r in data.get("rows", [])]
                    af.logger.debug(f"Fetched {len(rows)} rows")
                    return rows
        raise RuntimeError("HF rate-limit retries exhausted")

    async def _fill_buffer(self) -> None:
        af.logger.debug("Starting buffer fill")
        while len(self._buffer) < self.buffer_size:
            batch = self.max_batch
            offset = self._rng.randint(0, max(0, self.total_size - batch))
            try:
                rows = await self.fetch_hf(offset, batch)
            except Exception as e:
                af.logger.debug(f"Fetch error: {e!r}")
                continue
            for item in rows:
                self._buffer.append(item)
        af.logger.debug("Buffer fill complete")

    async def get(self) -> Any:
        async with self._lock:
            if not self._fill_task or self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            if not self._buffer:
                await self._fill_task
            item = self._buffer.popleft()
            if self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            return item

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.get()
