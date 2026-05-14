"""
lfu.py
------
LFU (Least Frequently Used / popularity-based) proactive cache baseline.

Pipeline:
    1. warm(train_df)              — count item frequencies in training data;
                                     cache top-K most frequent items.
    2. simulate_requests(node_id,
                         test_df) — hit/miss each test (user, item) pair.

Result dict schema (matches Module 5 CacheSimulator):
    node_id, hits, misses, total, hit_rate, cache_size, cached_items
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class LFUCache:
    """
    Popularity-based proactive cache.

    Caches the top-K most frequently requested items from training data.
    This is the simplest possible proactive baseline — no ML, just frequency.
    """

    def __init__(self, cache_size_k: int = 50):
        self.cache_size_k  = cache_size_k
        self._cached_set:  set  = set()
        self._cached_items: list = []
        self._warmed       = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm(self, train_df: pd.DataFrame) -> None:
        """
        Count item frequencies in training data and cache top-K.

        Args:
            train_df : DataFrame with at least column 'item_id'.
        """
        if train_df.empty:
            self._cached_set   = set()
            self._cached_items = []
            self._warmed       = True
            return

        freq = train_df["item_id"].value_counts()
        k    = min(self.cache_size_k, len(freq))
        top_items = freq.nlargest(k).index.tolist()

        self._cached_items = [int(i) for i in top_items]
        self._cached_set   = set(self._cached_items)
        self._warmed       = True

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
                "LFUCache.simulate_requests(): call warm() before simulate_requests()."
            )

        hits = misses = 0
        for item_id in test_df["item_id"].values:
            if int(item_id) in self._cached_set:
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
            "cache_size":   len(self._cached_set),
            "cached_items": self._cached_items,
        }
