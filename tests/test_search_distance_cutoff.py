"""max_distance: no default cutoff, one scale for every metric, and no silent drops."""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.backends import LexicalHit, LexicalResult
from mempalace.searcher import _finalize_candidate_hits, search_memories


def _vector_col(metric: str, distances: list) -> MagicMock:
    col = MagicMock()
    col.distance_metric = metric
    n = len(distances)
    col.query.return_value = {
        "ids": [[f"d{i}" for i in range(n)]],
        "documents": [[f"document number {i}" for i in range(n)]],
        "metadatas": [
            [{"wing": "w", "room": "r", "source_file": f"/p/doc{i}.md"} for i in range(n)]
        ],
        "distances": [distances],
    }
    return col


def _search(col, **kwargs) -> dict:
    with patch("mempalace.searcher.get_collection", return_value=col):
        with patch(
            "mempalace.searcher.get_closets_collection",
            side_effect=RuntimeError("no closets"),
        ):
            return search_memories("document", "/unused", n_results=5, **kwargs)


def _sources(result: dict) -> set:
    return {hit["source_file"] for hit in result["results"]}


def test_a_distant_hit_is_returned_by_default():
    result = _search(_vector_col("cosine", [0.2, 1.7]))

    assert _sources(result) == {"doc0.md", "doc1.md"}
    assert "filtered_by_max_distance" not in result


def test_a_cutoff_drops_and_counts_what_it_dropped():
    result = _search(_vector_col("cosine", [0.2, 1.7]), max_distance=1.0)

    assert _sources(result) == {"doc0.md"}
    assert result["filtered_by_max_distance"] == 1


@pytest.mark.parametrize(
    "metric, distances",
    [
        # l2 1.2 is cosine 0.72 for unit vectors; l2 1.9 is cosine 1.805.
        ("l2", [1.2, 1.9]),
        # ip -0.9 (dot 0.9) is cosine 0.1; ip 0.5 is cosine 1.5.
        ("ip", [-0.9, 0.5]),
    ],
)
def test_the_cutoff_reads_on_the_cosine_scale_for_every_metric(metric, distances):
    result = _search(_vector_col(metric, distances), max_distance=1.0)

    assert _sources(result) == {"doc0.md"}
    assert result["filtered_by_max_distance"] == 1


def _lexical_col(hits: list) -> MagicMock:
    col = MagicMock()
    col.distance_metric = "cosine"
    col.lexical_search.return_value = LexicalResult(hits=hits)
    return col


def _lexical_hit(hit_id: str, source: str) -> LexicalHit:
    return LexicalHit(
        id=hit_id,
        document=f"lexical text for {hit_id}",
        metadata={"wing": "w", "room": "r", "source_file": source, "chunk_index": 0},
        score=3.0,
    )


def _union(col, *, max_distance, stats):
    return _finalize_candidate_hits(
        candidate_strategy="union",
        hits=[],
        drawers_col=col,
        query="lexical text",
        wing=None,
        room=None,
        n_results=5,
        max_distance=max_distance,
        stats=stats,
    )


def test_a_bm25_only_hit_survives_a_failed_distance_lookup():
    col = _lexical_col([_lexical_hit("lex1", "/p/lex1.md")])
    col.get.side_effect = RuntimeError("embeddings unavailable")
    stats: dict = {}

    with patch("mempalace.backends.embedding_wrapper._embed_texts", return_value=[[1.0, 0.0]]):
        hits, error = _union(col, max_distance=0.5, stats=stats)

    assert error is None
    assert [hit["source_file"] for hit in hits] == ["lex1.md"]
    assert hits[0]["distance"] is None
    assert hits[0]["fallback_reason"] == "distance_unavailable"
    assert "filtered_by_max_distance" not in stats


def test_a_bm25_only_hit_past_the_cutoff_is_dropped_and_counted():
    col = _lexical_col([_lexical_hit("near", "/p/near.md"), _lexical_hit("far", "/p/far.md")])
    col.get.return_value = {"ids": ["near", "far"], "embeddings": [[1.0, 0.0], [0.0, 1.0]]}
    stats: dict = {}

    with patch("mempalace.backends.embedding_wrapper._embed_texts", return_value=[[1.0, 0.0]]):
        hits, error = _union(col, max_distance=0.5, stats=stats)

    assert error is None
    assert [hit["source_file"] for hit in hits] == ["near.md"]
    assert hits[0]["distance"] == 0.0
    assert "fallback_reason" not in hits[0]
    assert stats["filtered_by_max_distance"] == 1


def test_a_failed_lexical_fetch_is_reported_not_silent():
    col = MagicMock()
    col.distance_metric = "cosine"
    col.lexical_search.side_effect = RuntimeError("fts down")
    stats: dict = {}

    hits, error = _union(col, max_distance=None, stats=stats)

    assert error is None
    assert hits == []
    assert stats["candidate_strategy_fallback"] == "lexical_search_failed"
