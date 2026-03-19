"""Optional sampling helpers for cross-domain training."""

from __future__ import annotations

from typing import Dict, Iterator

from torch.utils.data import DataLoader


def round_robin_loaders(loaders: Dict[str, DataLoader]) -> Iterator[Dict[str, object]]:
    """
    Round-robin iterator over domain loaders.

    Yields:
        mapping domain -> batch for one synchronized step
    """
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    while True:
        batch_dict = {}
        for name, iterator in list(iterators.items()):
            try:
                batch_dict[name] = next(iterator)
            except StopIteration:
                return
        yield batch_dict