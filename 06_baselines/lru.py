"""
lru.py
------
LRU (Least Recently Used) reactive cache baseline.

Pipeline:
    1. warm(train_df)              — replay training interactions in order;
                                     insert each item; evict LRU when full.
    2. simulate_requests(node_id,
                         test_df) — hit/miss each test (user, item) pair
                                     against the warmed cache.

Result dict schema (matches Module 5 CacheSimulator):
    node_id, hits, misses, total, hit_rate, cache_size, cached_items
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import pandas as pd


class LRUCache:
    """
    O(1) LRU cache using collections.OrderedDict.

    Items are inserted / moved-to-end on access.
    When the cache is full the oldest (front) item is evicted.
    """

    def __init__(self, cache_size_k: int = 50):
        self.cache_size_k = cache_size_k
        self._cache: OrderedDict[int, None] = OrderedDict()
        self._warmed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _access(self, item_id: int) -> None:
        """Record an access to item_id, evicting LRU item if needed."""
        if item_id in self._cache:
            # Move to most-recently-used end
            self._cache.move_to_end(item_id)
        else:
            # Insert new item
            if len(self._cache) >= self.cache_size_k:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
            self._cache[item_id] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm(self, train_df: pd.DataFrame) -> None:
        """
        Replay training interactions in chronological order to warm the cache.

        Args:
            train_df : DataFrame with at least column 'item_id'.
                       Rows are assumed to be in temporal order (as stored).
        """
        self._cache.clear()
        for item_id in train_df["item_id"].values:
            self._access(int(item_id))
        self._warmed = True

    def simulate_requests(
        self,
        node_id: int,
        test_df: pd.DataFrame,
    ) -> Dict:
        """
        Simulate test requests against the warmed cache.

        Args:
            node_id  : edge node identifier (stored in result)
            test_df  : DataFrame with column 'item_id'

        Returns:
            dict with keys: node_id, hits, misses, total, hit_rate,
                            cache_size, cached_items

        Raises:
            ValueError if warm() has not been called first.
        """
        if not self._warmed:
            raise ValueError(
                "LRUCache.simulate_requests(): call warm() before simulate_requests()."
            )

        cached_set = set(self._cache.keys())
        hits = misses = 0

        for item_id in test_df["item_id"].values:
            if int(item_id) in cached_set:
                hits += 1
            else:
                misses += 1

        total    = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        return {
            "node_id":      node_id,
            "hits":         hits,
            "misses":       misses,
            "total":        total,
            "hit_rate":     hit_rate,
            "cache_size":   len(cached_set),
            "cached_items": list(cached_set),
        }
