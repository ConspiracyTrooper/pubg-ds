"""httpx-клиент PUBG API: retry на 429 (по Retry-After) + дисковый кэш json-ответов.

Rate limit 10 req/min действует только на служебных endpoint'ах
(/seasons, /players, leaderboards, /samples); /matches/{id} — вне лимита.
"""

import hashlib
import json
import os
import time

from pathlib import Path
from typing import Callable

import httpx

from pubg_ds import config


class PubgClient:
    """Обёртка над httpx.Client с ретраями на 429 и кэшем."""

    def __init__(
        self,
        api_key:     str | None = None,
        cache_dir:   Path | None = None,
        max_retries: int = 6,
        transport:   httpx.BaseTransport | None = None,
        sleep:       Callable[[float], None] = time.sleep,
    ):
        self.cache_dir   = Path(cache_dir) if cache_dir else config.DATA_RAW / "api_cache"
        self.max_retries = max_retries
        self._sleep      = sleep
        key              = api_key if api_key is not None else config.PUBG_API_KEY
        self._client     = httpx.Client(
            base_url=config.PUBG_API_BASE,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/vnd.api+json",
            },
            timeout=30.0,
            transport=transport,
        )

    # Ключ кэша — хэш от пути и отсортированных параметров запроса
    def _cache_path(self, path: str, params: dict | None) -> Path:
        raw = f"{path}?{sorted((params or {}).items())}"
        return self.cache_dir / f"{hashlib.md5(raw.encode()).hexdigest()}.json"

    # Атомарная запись: пишем во временный файл и переименовываем
    def _write_cache(self, cache_file: Path, data: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, cache_file)

    # Пауза перед повтором
    def _backoff(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        return min(2.0 ** attempt * 5.0, 65.0)

    def get(self, path: str, params: dict | None = None, use_cache: bool = True) -> dict:
        """GET с дисковым кэшем, ретраями на 429 и на транзиентные сетевые ошибки."""
        cache_file = self._cache_path(path, params)

        # Уже скачанное не тянем повторно
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        # Ретрай на 429 (ждём сброса лимита) и на таймаут/обрыв связи (транзиент)
        for attempt in range(self.max_retries):
            try:
                resp = self._client.get(path, params=params)
            except httpx.TransportError:
                # Последняя попытка исчерпана — пробрасываем ошибку наружу
                if attempt == self.max_retries - 1:
                    raise
                self._sleep(min(2.0 ** attempt * 2.0, 30.0))
                continue
            if resp.status_code == 429:
                self._sleep(self._backoff(resp, attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            if use_cache:
                self._write_cache(cache_file, data)
            return data

        raise RuntimeError(f"429 не ушёл после {self.max_retries} попыток: {path}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PubgClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
