"""
Test whether the configured embedding provider supports MRL (Matryoshka) dimension truncation.

Run BEFORE changing embedding_dim in config or running any migration.
If this returns 512-dim vectors, replace T5 halfvec with T5-mrl (much simpler migration).
If it fails or returns 1536-dim, proceed with T5 halfvec as planned.

Usage:
    .venv/Scripts/python.exe scripts/test_mrl_dimensions.py
Requires: EMBEDDING_API_KEY (or OPENROUTER_API_KEY) set in .env
"""

import asyncio
import os
import sys

# Load .env before importing anything that reads config
from pathlib import Path
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    from openai import AsyncOpenAI
    from config import settings
    from config_overlay import get_setting, get_setting_int

    target_dim = 512
    model = get_setting("embedding_model", settings.embedding_model)
    base_url = get_setting("embedding_base_url", settings.embedding_base_url)
    api_key = get_setting("embedding_api_key", settings.effective_embedding_api_key)
    current_dim = get_setting_int("embedding_dim", settings.embedding_dim)

    print(f"Provider : {base_url}")
    print(f"Model    : {model}")
    print(f"Current  : {current_dim}-dim")
    print(f"Testing  : {target_dim}-dim (MRL truncation)")
    print()

    if not api_key:
        print("ERROR: EMBEDDING_API_KEY / OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
    try:
        response = await client.embeddings.create(
            model=model,
            input=["test de dimensiones MRL"],
            dimensions=target_dim,
        )
        actual = len(response.data[0].embedding)
    except Exception as e:
        print(f"FAIL: API error — {e}")
        print()
        print("Decision: proceed with T5 halfvec (two-phase Alembic migration).")
        sys.exit(1)
    finally:
        await client.close()

    if actual == target_dim:
        print(f"PASS: got {actual}-dim vectors — MRL supported.")
        print()
        print("Decision: use T5-mrl instead of T5 halfvec.")
        print("Deploy sequence:")
        print("  1. Alembic migration: vector(1536) -> vector(512) + rebuild HNSW index")
        print("  2. Change embedding_dim=512 in config.py -> redeploy")
        print("  3. validate_config() at startup verifies match automatically")
        print("  4. Re-upload all documents (covered by T6)")
    else:
        print(f"FAIL: got {actual}-dim instead of {target_dim}-dim — provider ignores dimensions param.")
        print()
        print("Decision: proceed with T5 halfvec (two-phase Alembic migration).")


if __name__ == "__main__":
    asyncio.run(main())
