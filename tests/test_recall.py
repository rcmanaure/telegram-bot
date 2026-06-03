"""
Eval harness: recall@5 for the vector search pipeline.

Measures how often the expected chunk appears in the top-5 results for labeled
query→chunk pairs. Run before and after vector search changes to track improvement.

Requires live services:
    docker compose up -d
    docker compose exec api python -m pytest tests/test_recall.py -v -m integration
Or from host with .env:
    .venv/Scripts/python.exe -m pytest tests/test_recall.py -v -m integration

Populate tests/fixtures/eval_pairs.json with real pairs before running.
Sources: UnansweredQuery table + manual curation by bot operator.
"""
import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "eval_pairs.json"
EVAL_TOP_K = 5  # recall@5


def _load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _is_placeholder(pair: dict) -> bool:
    return "EXAMPLE" in pair.get("query", "") or "your-tenant-slug" in pair.get("namespace", "")


def _chunk_matches(chunk: dict, pair: dict) -> bool:
    """Return True if a retrieved chunk satisfies the eval pair criteria."""
    source_match = chunk["source"] == pair["expected_source"]
    if not source_match:
        return False
    content_filter = pair.get("expected_content_contains", "")
    if content_filter:
        return content_filter.lower() in chunk["content"].lower()
    return True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recall_at_5():
    """
    recall@5 = fraction of labeled pairs where the expected chunk
    appears in the top-5 vector search results.

    A score of 1.0 means every query finds its expected chunk.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    fixture = _load_fixture()
    threshold = fixture.get("threshold", 0.0)
    all_pairs = fixture.get("pairs", [])
    pairs = [p for p in all_pairs if not _is_placeholder(p)]

    if not pairs:
        pytest.skip(
            "No eval pairs configured. "
            "Add real pairs to tests/fixtures/eval_pairs.json and re-run."
        )

    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from config import settings
    from rag import retrieve_context

    engine = create_async_engine(settings.database_url, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    hits = 0
    results_log = []

    async with AsyncSessionLocal() as db:
        for pair in pairs:
            query = pair["query"]
            namespace = pair["namespace"]
            expected_source = pair["expected_source"]
            description = pair.get("description", query[:60])

            try:
                chunks = await retrieve_context(db, query, namespace, top_k=EVAL_TOP_K)
            except Exception as e:
                results_log.append(
                    f"  ERROR  [{description}] — retrieve_context raised: {e}"
                )
                continue

            matched = any(_chunk_matches(c, pair) for c in chunks)
            if matched:
                hits += 1
                top_sources = [c["source"] for c in chunks[:3]]
                results_log.append(
                    f"  HIT    [{description}] "
                    f"sim={chunks[0]['similarity']:.3f} top3={top_sources}"
                )
            else:
                top_sources = [(c["source"], round(c["similarity"], 3)) for c in chunks[:3]]
                results_log.append(
                    f"  MISS   [{description}] "
                    f"expected={expected_source!r} got={top_sources}"
                )

    await engine.dispose()

    recall = hits / len(pairs)

    print(f"\n{'='*60}")
    print(f"recall@{EVAL_TOP_K} = {hits}/{len(pairs)} = {recall:.2%}")
    print(f"threshold = {threshold:.2%}")
    print(f"{'='*60}")
    for line in results_log:
        print(line)
    print(f"{'='*60}\n")

    assert recall >= threshold, (
        f"recall@{EVAL_TOP_K} = {recall:.2%} is below threshold {threshold:.2%}. "
        f"Hits: {hits}/{len(pairs)}. See output above for per-pair breakdown."
    )
