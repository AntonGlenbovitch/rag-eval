from __future__ import annotations

from collections.abc import Hashable, Sequence


ID = Hashable


def _unique_in_order(items: Sequence[ID]) -> list[ID]:
    seen: set[ID] = set()
    unique_items: list[ID] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def recall_at_k(retrieved_ids: Sequence[ID], relevant_ids: Sequence[ID], k: int) -> float:
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    retrieved_top_k = set(_unique_in_order(retrieved_ids)[:k])
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: Sequence[ID], relevant_ids: Sequence[ID], k: int) -> float:
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_top_k = _unique_in_order(retrieved_ids)[:k]
    if not retrieved_top_k:
        return 0.0

    relevant_in_top_k = sum(1 for item in retrieved_top_k if item in relevant_set)
    return relevant_in_top_k / len(retrieved_top_k)


def reciprocal_rank(retrieved_ids: Sequence[ID], relevant_ids: Sequence[ID]) -> float:
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    for rank, item in enumerate(_unique_in_order(retrieved_ids), start=1):
        if item in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: Sequence[Sequence[ID]],
    relevant_lists: Sequence[Sequence[ID]],
) -> float:
    if not retrieved_lists or not relevant_lists:
        return 0.0

    pair_count = min(len(retrieved_lists), len(relevant_lists))
    if pair_count == 0:
        return 0.0

    rr_sum = 0.0
    for retrieved_ids, relevant_ids in zip(retrieved_lists[:pair_count], relevant_lists[:pair_count]):
        rr_sum += reciprocal_rank(retrieved_ids, relevant_ids)

    return rr_sum / pair_count


def compute_retrieval_metrics(
    retrieved_ids: Sequence[ID],
    relevant_ids: Sequence[ID],
    k: int,
) -> dict[str, float]:
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
    }
