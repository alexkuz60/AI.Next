from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from pipeline import CoreResult


@dataclass(slots=True)
class FileEntry:
    filename: str
    data: np.ndarray
    nums: int
    orders: int


class FileStore:
    """Small LRU store for uploaded numeric matrices (server side)."""

    def __init__(self, max_items: int = 4) -> None:
        self._items: OrderedDict[str, FileEntry] = OrderedDict()
        self._max_items = max_items

    def put(self, entry: FileEntry) -> str:
        key = uuid4().hex
        self._items[key] = entry
        self._items.move_to_end(key)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)
        return key

    def get(self, key: str | None) -> FileEntry | None:
        if not key:
            return None
        entry = self._items.get(key)
        if entry is not None:
            self._items.move_to_end(key)
        return entry


class CalcCache:
    """LRU cache for computed cores, addressable by id or by parameter key."""

    def __init__(self, max_items: int = 16) -> None:
        self._by_id: OrderedDict[str, CoreResult] = OrderedDict()
        self._by_key: dict[tuple, str] = {}
        self._max_items = max_items

    def put(self, core: CoreResult, key: tuple | None = None) -> str:
        if key is not None and key in self._by_key:
            calc_id = self._by_key[key]
            self._by_id.move_to_end(calc_id)
            return calc_id

        calc_id = uuid4().hex
        self._by_id[calc_id] = core
        if key is not None:
            self._by_key[key] = calc_id

        while len(self._by_id) > self._max_items:
            old_id, _ = self._by_id.popitem(last=False)
            for old_key, value in list(self._by_key.items()):
                if value == old_id:
                    del self._by_key[old_key]
        return calc_id

    def get(self, calc_id: str | None) -> CoreResult | None:
        if not calc_id:
            return None
        core = self._by_id.get(calc_id)
        if core is not None:
            self._by_id.move_to_end(calc_id)
        return core

    def find(self, key: tuple) -> CoreResult | None:
        calc_id = self._by_key.get(key)
        return self.get(calc_id) if calc_id else None
